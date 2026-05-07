// Pybind11 shim exposing Hailo's TAPPAS-derived YOLOv8/v5 instance segmentation
// reference postprocess to ara2-validator. Drives HailoRT directly with
// single-frame sync inference. Owns the full per-frame pipeline so the validator
// can compare end-to-end latency against HAL apples-to-apples:
//
//     OpenCV letterbox   |  preprocess_ms
//     HailoRT inference  |  inference_ms
//     TAPPAS decode+NMS+ |  postprocess_ms
//     mask matmul+sigmoid|
//
// The input contract is a raw BGR uint8 ndarray (cv2.imread default), matching
// Hailo's canonical reference. The shim does BGR->RGB and a proper
// aspect-preserving letterbox internally, then unmaps boxes+masks back to
// original-image coordinates so the caller can score against ground truth
// without further geometry work.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include "instance_seg_postprocess.hpp"
#include "general/hailo_objects.hpp"

namespace py = pybind11;
using namespace hailort;
using clk = std::chrono::steady_clock;

namespace {

double ms_since(const clk::time_point &t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// hailo-converter (>=0.1.x) appends a ZIP archive (edgefirst.json + labels.txt)
// to the end of every produced .hef. HailoRT validates the file's total length
// against the encoded protobuf size, so the trailing ZIP bytes have to be
// trimmed before handing the buffer to ``VDevice::create_infer_model``.
//
// We parse the ZIP End-Of-Central-Directory record to locate the smallest
// local-file-header offset; that is the original protobuf payload size.
// Returns ``buffer.size()`` when no recognizable EOCD is present (e.g. a HEF
// that was already stripped or never had a trailer).
size_t locate_hef_payload_size(const std::vector<uint8_t> &buf) {
    constexpr size_t EOCD_FIXED_SIZE = 22;
    constexpr uint32_t EOCD_SIG = 0x06054b50;  // "PK\x05\x06"
    constexpr uint32_t CD_SIG   = 0x02014b50;  // "PK\x01\x02"
    if (buf.size() < EOCD_FIXED_SIZE) return buf.size();

    // EOCD comment field is 0..65535 bytes, so the signature lies within
    // the last ~64KB. Scan that window for the EOCD magic.
    const size_t max_scan = std::min(buf.size(), size_t(EOCD_FIXED_SIZE + 65535));
    for (size_t back = EOCD_FIXED_SIZE; back <= max_scan; ++back) {
        const size_t i = buf.size() - back;
        uint32_t sig;
        std::memcpy(&sig, buf.data() + i, 4);
        if (sig != EOCD_SIG) continue;

        // EOCD layout: at +16 is offset of central directory (4 bytes LE).
        uint32_t cd_offset;
        std::memcpy(&cd_offset, buf.data() + i + 16, 4);
        if (cd_offset == 0 || cd_offset > buf.size() - 4) return buf.size();
        uint32_t cd_sig;
        std::memcpy(&cd_sig, buf.data() + cd_offset, 4);
        if (cd_sig != CD_SIG) return buf.size();

        // First central directory entry: at +42 is the relative offset of the
        // first local file header (4 bytes LE). That offset == start of the
        // appended ZIP region == end of the HEF protobuf payload.
        if (cd_offset + 46 > buf.size()) return buf.size();
        uint32_t local_header_offset;
        std::memcpy(&local_header_offset, buf.data() + cd_offset + 42, 4);
        if (local_header_offset == 0 || local_header_offset >= buf.size())
            return buf.size();
        return local_header_offset;
    }
    return buf.size();
}

// Aspect-preserving letterbox: scale to fit (longest side matches model dim),
// pad bottom and right with zeros. Matches the upstream
// `make_model_space_canvas` semantics (instance_seg_postprocess.cpp:120).
//
// We deliberately do NOT use the upstream sample's `pad_crop_to_target` from
// toolbox.cpp:919 because it is no-resize: it only pads (if input is smaller
// than model) or top-left crops (if larger), which silently destroys content
// for typical input sizes. Real Hailo deployments use a proper letterbox; this
// is the function the upstream code itself uses to undo the geometry on the
// postprocess side, so using it on both sides keeps preprocess and postprocess
// consistent.
//
// `LetterboxMap` is defined in instance_seg_postprocess.hpp (we use the
// upstream's struct rather than redeclaring our own).
cv::Mat letterbox_for_model(const cv::Mat &src,
                            int model_w, int model_h,
                            LetterboxMap &map)
{
    const float fh = static_cast<float>(src.rows) / static_cast<float>(model_h);
    const float fw = static_cast<float>(src.cols) / static_cast<float>(model_w);
    map.factor = std::max(fh, fw);

    cv::Mat resized;
    cv::resize(src, resized,
               cv::Size(static_cast<int>(std::round(src.cols / map.factor)),
                        static_cast<int>(std::round(src.rows / map.factor))),
               0, 0, cv::INTER_AREA);

    map.crop_h = resized.rows;
    map.crop_w = resized.cols;
    map.pad_h = std::max(0, model_h - resized.rows);
    map.pad_w = std::max(0, model_w - resized.cols);

    cv::Mat canvas;
    cv::copyMakeBorder(resized, canvas,
                       /*top*/ 0, /*bottom*/ map.pad_h,
                       /*left*/ 0, /*right*/ map.pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return canvas;
}

// Per-detection mask unmapping: TAPPAS's `filter()` produces masks at model
// space when called with (model_h, model_w). Crop the valid region (top-left,
// dims = crop_h x crop_w from the LetterboxMap) and resize to original image
// dims. This is the per-mask analogue of upstream's `map_model_to_frame`
// (instance_seg_postprocess.cpp:146) which operates on a whole canvas.
cv::Mat unmap_mask(const cv::Mat &mask_model_space,
                   const LetterboxMap &map,
                   int org_h, int org_w)
{
    const int cw = std::min(map.crop_w, mask_model_space.cols);
    const int ch = std::min(map.crop_h, mask_model_space.rows);
    cv::Rect roi(0, 0, cw, ch);
    cv::Mat cropped = mask_model_space(roi).clone();
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(org_w, org_h), 0, 0, cv::INTER_LINEAR);
    return resized;
}

}  // namespace

class HailoTappasBackend {
public:
    explicit HailoTappasBackend(const std::string &hef_path) {
        // Slurp the file and trim the hailo-converter ZIP trailer (if any)
        // before handing the buffer to HailoRT. We keep the trimmed copy
        // alive as a member because create_infer_model's MemoryView overload
        // does not take ownership of the bytes it parses.
        {
            std::ifstream f(hef_path, std::ios::binary);
            if (!f) throw std::runtime_error("cannot open HEF: " + hef_path);
            f.seekg(0, std::ios::end);
            const auto sz = static_cast<std::streamoff>(f.tellg());
            f.seekg(0, std::ios::beg);
            std::vector<uint8_t> raw(static_cast<size_t>(sz));
            f.read(reinterpret_cast<char *>(raw.data()), sz);
            if (!f) throw std::runtime_error("short read on HEF: " + hef_path);
            const size_t payload = locate_hef_payload_size(raw);
            raw.resize(payload);
            hef_buffer_ = std::move(raw);
        }

        auto vd = VDevice::create();
        if (!vd) throw std::runtime_error(
            "VDevice::create failed: " + std::to_string(vd.status()));
        vdevice_ = std::move(vd.value());

        auto im = vdevice_->create_infer_model(
            MemoryView(hef_buffer_.data(), hef_buffer_.size()));
        if (!im) throw std::runtime_error(
            "create_infer_model failed: " + std::to_string(im.status()));
        infer_model_ = im.value();
        infer_model_->set_batch_size(1);

        auto out_infos = infer_model_->hef().get_output_vstream_infos();
        if (!out_infos) throw std::runtime_error(
            "get_output_vstream_infos failed: " + std::to_string(out_infos.status()));
        for (const auto &info : out_infos.value()) {
            output_vsinfo_[std::string(info.name)] = info;
        }

        auto in_infos = infer_model_->hef().get_input_vstream_infos();
        if (!in_infos || in_infos->empty())
            throw std::runtime_error("model has no input vstream");
        const auto &shape = in_infos.value()[0].shape;
        model_h_ = static_cast<int>(shape.height);
        model_w_ = static_cast<int>(shape.width);
        model_c_ = static_cast<int>(shape.features);
        if (model_c_ != 3)
            throw std::runtime_error(
                "expected 3-channel model input, got channels=" + std::to_string(model_c_));
        input_name_ = infer_model_->get_input_names().at(0);

        auto cfg = infer_model_->configure();
        if (!cfg) throw std::runtime_error(
            "configure failed: " + std::to_string(cfg.status()));
        cfg_ = std::make_unique<ConfiguredInferModel>(std::move(cfg.value()));
    }

    py::dict infer(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image_bgr)
    {
        auto buf = image_bgr.request();
        if (buf.ndim != 3 || buf.shape[2] != 3)
            throw std::runtime_error(
                "input must be a contiguous (H, W, 3) uint8 BGR ndarray");
        const int org_h = static_cast<int>(buf.shape[0]);
        const int org_w = static_cast<int>(buf.shape[1]);

        double pre_ms = 0.0, infer_ms = 0.0, post_ms = 0.0;
        std::vector<HailoDetectionPtr> dets;
        std::vector<cv::Mat> masks_model_space;
        LetterboxMap lbmap{};

        {
            // Heavy compute under released GIL.
            py::gil_scoped_release release;

            // ----- 1. Preprocess: BGR view -> RGB -> letterbox -----
            //
            // Each step (cvtColor, resize, copyMakeBorder) is documented as
            // eager, but OpenCV's fast paths can skip the actual write under
            // certain conditions (e.g. resize with src.size == dst.size, or
            // a stride-shifted output that aliases the input). To prevent a
            // timer that stops before the data has actually landed, we force
            // a final clone(). It's an O(N) memcpy so it both materializes
            // any deferred kernel work and guarantees HailoRT receives a
            // fresh, contiguous buffer it can DMA directly.
            auto t_pre = clk::now();
            cv::Mat src_bgr(org_h, org_w, CV_8UC3, buf.ptr);   // zero-copy view
            cv::Mat rgb;
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
            cv::Mat letterboxed = letterbox_for_model(rgb, model_w_, model_h_, lbmap);
            letterboxed = letterboxed.clone();   // materialization barrier
            pre_ms = ms_since(t_pre);

            // ----- 2. Bind buffers + run sync inference -----
            std::map<std::string, std::vector<uint8_t>> output_buffers;
            for (const auto &name : infer_model_->get_output_names()) {
                output_buffers[name].resize(infer_model_->output(name)->get_frame_size());
            }

            auto bindings_exp = cfg_->create_bindings();
            if (!bindings_exp)
                throw std::runtime_error("create_bindings failed: " +
                                         std::to_string(bindings_exp.status()));
            auto bindings = std::move(bindings_exp.value());

            const size_t in_sz = infer_model_->input(input_name_)->get_frame_size();
            if (in_sz != static_cast<size_t>(letterboxed.total() * letterboxed.elemSize()))
                throw std::runtime_error(
                    "letterboxed buffer size " +
                    std::to_string(letterboxed.total() * letterboxed.elemSize()) +
                    " != model frame size " + std::to_string(in_sz));
            auto in_status = bindings.input(input_name_)->set_buffer(
                MemoryView(letterboxed.data, in_sz));
            if (HAILO_SUCCESS != in_status)
                throw std::runtime_error("input set_buffer failed: " +
                                         std::to_string(in_status));

            for (auto &kv : output_buffers) {
                auto st = bindings.output(kv.first)->set_buffer(
                    MemoryView(kv.second.data(), kv.second.size()));
                if (HAILO_SUCCESS != st)
                    throw std::runtime_error(
                        "output set_buffer failed for '" + kv.first + "': " +
                        std::to_string(st));
            }

            auto t_inf = clk::now();
            auto wait_ready = cfg_->wait_for_async_ready(
                std::chrono::milliseconds(5000), 1);
            if (HAILO_SUCCESS != wait_ready)
                throw std::runtime_error("wait_for_async_ready failed: " +
                                         std::to_string(wait_ready));
            auto job_exp = cfg_->run_async(bindings);
            if (!job_exp)
                throw std::runtime_error("run_async failed: " +
                                         std::to_string(job_exp.status()));
            auto wait_status = job_exp->wait(std::chrono::seconds(5));
            if (HAILO_SUCCESS != wait_status)
                throw std::runtime_error("inference wait failed: " +
                                         std::to_string(wait_status));
            infer_ms = ms_since(t_inf);

            // ----- 3. TAPPAS postprocess at MODEL space -----
            std::vector<std::pair<uint8_t *, hailo_vstream_info_t>> outputs;
            outputs.reserve(output_buffers.size());
            for (auto &kv : output_buffers)
                outputs.emplace_back(kv.second.data(), output_vsinfo_.at(kv.first));

            auto t_post = clk::now();
            auto roi = build_roi_from_outputs(outputs);
            // Pass model dims (NOT org dims) so TAPPAS produces masks in
            // letterbox space; we unmap to org coords below.
            masks_model_space = filter(roi, model_h_, model_w_);
            dets = get_detections_from_roi(roi);
            post_ms = ms_since(t_post);
        }
        // GIL re-acquired. Marshaling + per-mask unmap below.

        const py::ssize_t N = static_cast<py::ssize_t>(dets.size());
        py::array_t<float> boxes({N, py::ssize_t(4)});
        py::array_t<float> scores(N);
        py::array_t<int32_t> classes(N);

        auto bm = boxes.mutable_unchecked<2>();
        auto sm = scores.mutable_unchecked<1>();
        auto cm = classes.mutable_unchecked<1>();

        // Letterbox -> org coordinate mapping for boxes:
        // bb is normalized [0,1] in letterbox space. The valid region of the
        // letterboxed canvas spans [0, crop_w/model_w] x [0, crop_h/model_h];
        // outside that is padding. Multiplying by (model_w / crop_w) rescales
        // to [0,1] in org space and clamps any padding-region predictions.
        const float sx = static_cast<float>(model_w_) /
                         std::max(1, lbmap.crop_w);
        const float sy = static_cast<float>(model_h_) /
                         std::max(1, lbmap.crop_h);

        py::list mask_list;
        for (py::ssize_t i = 0; i < N; ++i) {
            const auto &det = dets[i];
            const auto &bb = det->get_bbox();
            float x1 = bb.xmin() * sx;
            float y1 = bb.ymin() * sy;
            float x2 = (bb.xmin() + bb.width()) * sx;
            float y2 = (bb.ymin() + bb.height()) * sy;
            // Clamp to valid org-normalized range
            x1 = std::clamp(x1, 0.0f, 1.0f);
            y1 = std::clamp(y1, 0.0f, 1.0f);
            x2 = std::clamp(x2, 0.0f, 1.0f);
            y2 = std::clamp(y2, 0.0f, 1.0f);
            bm(i, 0) = x1;
            bm(i, 1) = y1;
            bm(i, 2) = x2;
            bm(i, 3) = y2;
            sm(i) = det->get_confidence();
            cm(i) = det->get_class_id();

            // Per-mask unmap to org resolution. (TAPPAS produced masks at
            // model space because we passed model_h/model_w to filter().)
            const cv::Mat &m = masks_model_space[i];
            if (m.type() != CV_32FC1)
                throw std::runtime_error("expected float32 mask from filter()");
            cv::Mat mask_org = unmap_mask(m, lbmap, org_h, org_w);
            py::array_t<float> mask_np({mask_org.rows, mask_org.cols});
            std::memcpy(mask_np.mutable_data(), mask_org.data,
                        static_cast<size_t>(mask_org.rows) *
                        static_cast<size_t>(mask_org.cols) * sizeof(float));
            mask_list.append(std::move(mask_np));
        }

        py::dict timings;
        timings["preprocess_ms"] = pre_ms;
        timings["inference_ms"] = infer_ms;
        timings["postprocess_ms"] = post_ms;

        py::dict result;
        result["boxes"] = boxes;       // (N, 4) xyxy normalized [0, 1] in org space
        result["scores"] = scores;     // (N,)
        result["classes"] = classes;   // (N,) 0-indexed
        result["masks"] = mask_list;   // list of float32 (org_h, org_w) post-sigmoid
        result["timings"] = timings;
        return result;
    }

    int model_width() const { return model_w_; }
    int model_height() const { return model_h_; }
    int model_channels() const { return model_c_; }

    std::vector<std::string> input_names() const {
        return infer_model_->get_input_names();
    }
    std::vector<std::string> output_names() const {
        return infer_model_->get_output_names();
    }

private:
    std::vector<uint8_t> hef_buffer_;     // owns bytes referenced by InferModel
    std::unique_ptr<VDevice> vdevice_;
    std::shared_ptr<InferModel> infer_model_;
    std::unique_ptr<ConfiguredInferModel> cfg_;
    std::map<std::string, hailo_vstream_info_t> output_vsinfo_;
    std::string input_name_;
    int model_h_{0}, model_w_{0}, model_c_{0};
};

PYBIND11_MODULE(hailo_tappas_baseline, m) {
    m.doc() = "Hailo TAPPAS YOLOv8/v5 instance segmentation reference backend";

    py::class_<HailoTappasBackend>(m, "HailoTappasBackend")
        .def(py::init<const std::string &>(), py::arg("hef_path"),
             "Open HEF and configure HailoRT InferModel for single-frame sync inference.")
        .def("infer", &HailoTappasBackend::infer, py::arg("image_bgr"),
             "Run the full per-frame pipeline (OpenCV letterbox + HailoRT + TAPPAS decode).\n\n"
             "Args:\n"
             "    image_bgr: uint8 BGR ndarray shaped (H, W, 3), as returned by cv2.imread.\n\n"
             "Returns dict with keys:\n"
             "    boxes (N, 4) float32 xyxy normalized [0, 1] in original-image coords\n"
             "    scores (N,) float32\n"
             "    classes (N,) int32 0-indexed COCO IDs\n"
             "    masks list[float32 (H, W)] post-sigmoid, bbox-cropped, original resolution\n"
             "    timings {preprocess_ms, inference_ms, postprocess_ms}")
        .def_property_readonly("model_width", &HailoTappasBackend::model_width)
        .def_property_readonly("model_height", &HailoTappasBackend::model_height)
        .def_property_readonly("model_channels", &HailoTappasBackend::model_channels)
        .def_property_readonly("input_names", &HailoTappasBackend::input_names)
        .def_property_readonly("output_names", &HailoTappasBackend::output_names);
}
