// Pybind11 shim exposing Hailo's TAPPAS-derived YOLOv8/v5 instance segmentation
// reference postprocess to ara2-validator. Drives HailoRT directly (sync, no
// GStreamer) and routes raw output buffers through the upstream `filter()` /
// `build_roi_from_outputs()` / `get_detections_from_roi()` entrypoints.
//
// Returns NumPy arrays plus a list of float32 masks at original-image
// resolution, matching the contract `ara2_validator.hailo_pipeline` will
// expect from a baseline backend.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstring>
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
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

class HailoTappasBackend {
public:
    explicit HailoTappasBackend(const std::string &hef_path) {
        auto vd = VDevice::create();
        if (!vd) throw std::runtime_error(
            "VDevice::create failed: " + std::to_string(vd.status()));
        vdevice_ = std::move(vd.value());

        auto im = vdevice_->create_infer_model(hef_path);
        if (!im) throw std::runtime_error(
            "create_infer_model failed: " + std::to_string(im.status()));
        infer_model_ = im.value();
        infer_model_->set_batch_size(1);

        // Cache output vstream infos by name (postprocess needs them)
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
        input_name_ = infer_model_->get_input_names().at(0);

        auto cfg = infer_model_->configure();
        if (!cfg) throw std::runtime_error(
            "configure failed: " + std::to_string(cfg.status()));
        cfg_ = std::make_unique<ConfiguredInferModel>(std::move(cfg.value()));
    }

    py::dict infer(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> input,
                   int org_height, int org_width)
    {
        auto buf = input.request();
        if (buf.ndim != 3 ||
            buf.shape[0] != model_h_ ||
            buf.shape[1] != model_w_ ||
            buf.shape[2] != model_c_)
        {
            throw std::runtime_error(
                "input shape mismatch: expected (" +
                std::to_string(model_h_) + ", " +
                std::to_string(model_w_) + ", " +
                std::to_string(model_c_) + "), got " +
                std::to_string(buf.shape[0]) + "x" +
                std::to_string(buf.shape[1]) + "x" +
                std::to_string(buf.shape[2]));
        }

        // ----- Allocate output buffers -----
        std::map<std::string, std::vector<uint8_t>> output_buffers;
        for (const auto &name : infer_model_->get_output_names()) {
            size_t sz = infer_model_->output(name)->get_frame_size();
            output_buffers[name].resize(sz);
        }

        double infer_ms = 0.0;
        double post_ms = 0.0;
        std::vector<HailoDetectionPtr> dets;
        std::vector<cv::Mat> masks;
        {
            // Release GIL during the heavy compute (HailoRT async wait + xtensor postprocess).
            py::gil_scoped_release release;

            // ----- Bind buffers (one Bindings, single-frame) -----
            auto bindings_exp = cfg_->create_bindings();
            if (!bindings_exp)
                throw std::runtime_error("create_bindings failed: " +
                                         std::to_string(bindings_exp.status()));
            auto bindings = std::move(bindings_exp.value());

            size_t in_sz = infer_model_->input(input_name_)->get_frame_size();
            auto in_status = bindings.input(input_name_)->set_buffer(
                MemoryView(buf.ptr, in_sz));
            if (HAILO_SUCCESS != in_status)
                throw std::runtime_error("input set_buffer failed: " +
                                         std::to_string(in_status));

            for (auto &kv : output_buffers) {
                auto out_status = bindings.output(kv.first)->set_buffer(
                    MemoryView(kv.second.data(), kv.second.size()));
                if (HAILO_SUCCESS != out_status)
                    throw std::runtime_error("output set_buffer failed for '" +
                                             kv.first + "': " +
                                             std::to_string(out_status));
            }

            // ----- Sync inference (run_async + wait, single bindings) -----
            auto t0 = clk::now();
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
            infer_ms = ms_since(t0);

            // ----- TAPPAS postprocess on raw outputs -----
            std::vector<std::pair<uint8_t *, hailo_vstream_info_t>> outputs;
            outputs.reserve(output_buffers.size());
            for (auto &kv : output_buffers) {
                outputs.emplace_back(kv.second.data(), output_vsinfo_.at(kv.first));
            }

            auto t1 = clk::now();
            auto roi = build_roi_from_outputs(outputs);
            masks = filter(roi, org_height, org_width);
            dets = get_detections_from_roi(roi);
            post_ms = ms_since(t1);
        }
        // GIL re-acquired here.

        // ----- Marshal to Python -----
        const py::ssize_t N = static_cast<py::ssize_t>(dets.size());
        py::array_t<float> boxes({N, py::ssize_t(4)});
        py::array_t<float> scores(N);
        py::array_t<int32_t> classes(N);

        auto bm = boxes.mutable_unchecked<2>();
        auto sm = scores.mutable_unchecked<1>();
        auto cm = classes.mutable_unchecked<1>();

        py::list mask_list;
        for (py::ssize_t i = 0; i < N; ++i) {
            const auto &det = dets[i];
            const auto &bb = det->get_bbox();
            // HailoBBox is (xmin, ymin, w, h) normalized to [0,1].
            // Emit xyxy normalized for compatibility with HAL's contract.
            const float x1 = bb.xmin();
            const float y1 = bb.ymin();
            const float x2 = bb.xmin() + bb.width();
            const float y2 = bb.ymin() + bb.height();
            bm(i, 0) = x1;
            bm(i, 1) = y1;
            bm(i, 2) = x2;
            bm(i, 3) = y2;
            sm(i) = det->get_confidence();
            cm(i) = det->get_class_id();

            // Mask: float32 single-channel, contiguous, (org_h, org_w).
            const cv::Mat &m = masks[i];
            if (m.type() != CV_32FC1)
                throw std::runtime_error("expected float32 mask from filter()");
            py::array_t<float> mask_np({m.rows, m.cols});
            std::memcpy(mask_np.mutable_data(), m.data,
                        static_cast<size_t>(m.rows) *
                        static_cast<size_t>(m.cols) * sizeof(float));
            mask_list.append(std::move(mask_np));
        }

        py::dict timings;
        timings["inference_ms"] = infer_ms;
        timings["postprocess_ms"] = post_ms;

        py::dict result;
        result["boxes"] = boxes;       // (N, 4) xyxy normalized [0, 1]
        result["scores"] = scores;     // (N,)
        result["classes"] = classes;   // (N,) 0-indexed
        result["masks"] = mask_list;   // list of float32 (org_h, org_w)
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
             py::arg("input"), py::arg("org_height"), py::arg("org_width"),
             "Run inference + TAPPAS postprocess.\n\n"
             "Args:\n"
             "    input: uint8 NHWC array shaped (model_h, model_w, model_c)\n"
             "        — already letterboxed and quantized to model input format.\n"
             "    org_height, org_width: original image size; masks are returned\n"
             "        at this resolution.\n\n"
             "Returns dict with keys:\n"
             "    boxes (N,4) float32 xyxy normalized [0,1]\n"
             "    scores (N,) float32\n"
             "    classes (N,) int32 (0-indexed)\n"
             "    masks list[float32 (org_h, org_w)] post-sigmoid, bbox-cropped\n"
             "    timings {inference_ms, postprocess_ms}")
        .def_property_readonly("model_width", &HailoTappasBackend::model_width)
        .def_property_readonly("model_height", &HailoTappasBackend::model_height)
        .def_property_readonly("model_channels", &HailoTappasBackend::model_channels)
        .def_property_readonly("input_names", &HailoTappasBackend::input_names)
        .def_property_readonly("output_names", &HailoTappasBackend::output_names);
}
