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

// Shim-local letterbox descriptor. Replaces upstream's letterbox-map struct
// (which only carries factor + crop_w/h + pad_w/h, sufficient for top-left
// letterbox) so we can carry pad_left/pad_top for centered letterbox.
// Upstream's struct stays untouched in instance_seg_postprocess.hpp; we just
// stop using it inside the shim.
struct ShimLetterbox {
    float scale{1.0f};   // dst / src; src*scale == new_w/new_h
    int new_w{0};        // resized image width
    int new_h{0};        // resized image height
    int pad_left{0};     // top-left X offset of resized region in canvas
    int pad_top{0};      // top-left Y offset of resized region in canvas
};

// Ultralytics-faithful letterbox: scale by min ratio, centered pad with
// (114,114,114) gray, bilinear interpolation. Matches LetterBox(center=True)
// in ultralytics/data/augment.py:1542 and ara2_validator/preprocess.py
// letterbox(), so the TAPPAS reference pipeline consumes the same model-input
// bytes as the HAL preprocess path for the same image.
//
// Asymmetric pad rounding: integer split (model_w - new_w) / 2 for left/top
// paired with (model_w - new_w - pad_left) for right/bottom is exactly
// equivalent to Ultralytics' round(dw - 0.1) / round(dw + 0.1) rule for
// every non-negative integer total pad (even or odd). On odd totals the
// bottom/right side gets the extra pixel. Total pad >= 0 because
// scale = min(model_w/w, model_h/h) guarantees new_w <= model_w. (If
// scaleup=False is ever added per spec Section 9 future work, total pad can
// go negative for small images and C++ integer division semantics diverge
// from Python //; equivalence breaks in that regime.)
cv::Mat letterbox_for_model(const cv::Mat &src,
                            int model_w, int model_h,
                            ShimLetterbox &lb)
{
    lb.scale = std::min(static_cast<float>(model_w) / src.cols,
                        static_cast<float>(model_h) / src.rows);
    lb.new_w = static_cast<int>(std::round(src.cols * lb.scale));
    lb.new_h = static_cast<int>(std::round(src.rows * lb.scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(lb.new_w, lb.new_h),
               0, 0, cv::INTER_LINEAR);

    lb.pad_left = (model_w - lb.new_w) / 2;
    lb.pad_top  = (model_h - lb.new_h) / 2;
    const int pad_right  = model_w - lb.new_w - lb.pad_left;
    const int pad_bottom = model_h - lb.new_h - lb.pad_top;

    cv::Mat canvas;
    cv::copyMakeBorder(resized, canvas,
                       lb.pad_top, pad_bottom,
                       lb.pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return canvas;
}

// Per-detection mask unmap. TAPPAS produced this mask at model space (640x640)
// because we passed (model_h, model_w) to filter(). After our centered
// letterbox, the valid region is the (new_w x new_h) sub-rectangle starting
// at (pad_left, pad_top). The integer pad_left and new_w used here are the
// same values that constructed the canvas in letterbox_for_model, so there
// is no rounding drift between the canvas the model saw and the ROI we cut.
// Defensive clamps below are paranoia against future invariant violations.
cv::Mat unmap_mask(const cv::Mat &mask_model_space,
                   const ShimLetterbox &lb,
                   int org_h, int org_w)
{
    const int x = std::max(0, std::min(lb.pad_left, mask_model_space.cols - 1));
    const int y = std::max(0, std::min(lb.pad_top,  mask_model_space.rows - 1));
    const int w = std::min(lb.new_w, mask_model_space.cols - x);
    const int h = std::min(lb.new_h, mask_model_space.rows - y);
    cv::Mat cropped = mask_model_space(cv::Rect(x, y, w, h)).clone();
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(org_w, org_h),
               0, 0, cv::INTER_LINEAR);
    return resized;
}

// Test-only accessor: runs cvtColor + letterbox_for_model and returns the
// canvas. Bound at module level so a Python pytest can verify the shim's
// preprocess geometry matches the validator's preprocess.letterbox()
// without needing a HEF or HailoRT.
py::array_t<uint8_t> letterbox_for_test(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image_bgr,
    int model_w, int model_h)
{
    auto buf = image_bgr.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("input must be (H, W, 3) uint8 BGR");
    const int h = static_cast<int>(buf.shape[0]);
    const int w = static_cast<int>(buf.shape[1]);
    cv::Mat src_bgr(h, w, CV_8UC3, buf.ptr);
    cv::Mat rgb;
    cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
    ShimLetterbox lb{};
    cv::Mat canvas = letterbox_for_model(rgb, model_w, model_h, lb);
    py::array_t<uint8_t> out({canvas.rows, canvas.cols, 3});
    std::memcpy(out.mutable_data(), canvas.data,
                static_cast<size_t>(canvas.rows) * canvas.cols * 3);
    return out;
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

    py::dict infer(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image_bgr,
                   float score_threshold = 0.6f,
                   float iou_threshold = 0.7f)
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
        ShimLetterbox lb{};

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
            cv::Mat letterboxed = letterbox_for_model(rgb, model_w_, model_h_, lb);
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
            masks_model_space = filter(roi, model_h_, model_w_,
                                       score_threshold, iou_threshold);
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

        // bb is normalized [0,1] in letterbox space. Convert to org-normalized
        // via (bb - pad/model) * (model/new). pad/model rebases the origin to
        // the start of the valid region; (model/new) rescales the valid span
        // to [0,1]. Clamp matches Ultralytics' clip_boxes (utils/ops.py:152) —
        // detections outside the valid region collapse to zero-area at the
        // edge or get truncated to the image boundary.
        const float pad_lx_norm = static_cast<float>(lb.pad_left) / model_w_;
        const float pad_ty_norm = static_cast<float>(lb.pad_top)  / model_h_;
        const float sx = static_cast<float>(model_w_) / std::max(1, lb.new_w);
        const float sy = static_cast<float>(model_h_) / std::max(1, lb.new_h);

        py::list mask_list;
        for (py::ssize_t i = 0; i < N; ++i) {
            const auto &det = dets[i];
            const auto &bb = det->get_bbox();
            float x1 = (bb.xmin() - pad_lx_norm) * sx;
            float y1 = (bb.ymin() - pad_ty_norm) * sy;
            float x2 = (bb.xmin() + bb.width()  - pad_lx_norm) * sx;
            float y2 = (bb.ymin() + bb.height() - pad_ty_norm) * sy;
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
            cv::Mat mask_org = unmap_mask(m, lb, org_h, org_w);
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
        .def("infer", &HailoTappasBackend::infer,
             py::arg("image_bgr"),
             py::arg("score_threshold") = 0.6f,
             py::arg("iou_threshold") = 0.7f,
             "Run the full per-frame pipeline (OpenCV letterbox + HailoRT + TAPPAS decode).\n\n"
             "Args:\n"
             "    image_bgr: uint8 BGR ndarray shaped (H, W, 3), as returned by cv2.imread.\n"
             "    score_threshold: confidence threshold applied during box decode (default 0.6,\n"
             "        the upstream TAPPAS sample default; use 0.001 for Ultralytics-style val).\n"
             "    iou_threshold: NMS IoU threshold (default 0.7).\n\n"
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

    m.def("letterbox_for_test", &letterbox_for_test,
          py::arg("image_bgr"), py::arg("model_w") = 640, py::arg("model_h") = 640,
          "Run cvtColor + the shim's letterbox_for_model on an input BGR image.\n"
          "Returns the (model_h, model_w, 3) uint8 RGB canvas. Test-only; the\n"
          "production path is HailoTappasBackend.infer().");
}
