# pybind11 shim — `hailo_tappas_baseline`

Thin Python module exposing the upstream TAPPAS YOLOv8/v5-seg reference
postprocess as a single class, `HailoTappasBackend`, that owns the full
per-frame pipeline:

| Stage | Implementation | Reported as |
|---|---|---|
| Letterbox + BGR→RGB | OpenCV (`cv::cvtColor` + aspect-preserving resize/pad) | `preprocess_ms` |
| NPU inference | HailoRT direct, single-frame sync | `inference_ms` |
| Decode + NMS + mask matmul + sigmoid | Vendored TAPPAS (`build_roi_from_outputs` + `filter()`) | `postprocess_ms` |

This is the canonical "OpenCV preprocess + Hailo NPU + TAPPAS postprocess"
reference path, intended for head-to-head accuracy/perf comparison against
the EdgeFirst HAL pipeline. Imported by `ara2_validator.hailo_pipeline`.

## Build (RPi5 or any aarch64 host with HailoRT)

```bash
sudo apt install gcc-11 g++-11 cmake libopencv-dev pybind11-dev python3-dev
sudo dpkg -i hailort_*_arm64.deb hailort-pcie-driver_*_all.deb

cd third_party/hailo_tappas_baseline/pybind11
cmake -S . -B build -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11
cmake --build build -j
cmake --install build   # places .so next to source for in-tree imports
```

After install the module sits at
`third_party/hailo_tappas_baseline/pybind11/hailo_tappas_baseline*.so`.

## Module API

```python
import cv2
from hailo_tappas_baseline import HailoTappasBackend

backend = HailoTappasBackend("yolov8n-seg.hef")
backend.model_width, backend.model_height, backend.model_channels  # e.g. 640, 640, 3
backend.input_names, backend.output_names                          # introspection

# Input contract: raw BGR uint8 ndarray, native resolution.
# The shim does cvtColor, letterbox, inference, postprocess, and unmaps results
# back to the input image's coordinates.
image_bgr = cv2.imread("frame.jpg")
result = backend.infer(image_bgr)

result["boxes"]    # (N, 4) float32 xyxy normalized [0, 1] in original-image space
result["scores"]   # (N,)   float32
result["classes"]  # (N,)   int32 0-indexed COCO class IDs
result["masks"]    # list[float32 (H, W)] post-sigmoid, bbox-cropped, at input H/W
result["timings"]  # {"preprocess_ms": float, "inference_ms": float, "postprocess_ms": float}
```

The GIL is released around preprocess + inference + postprocess. NumPy
marshaling and per-mask unmapping run with the GIL held, but the heavy compute
(OpenCV resize, HailoRT wait, xtensor proto×coef matmul) does not block other
Python threads.

## Letterbox: deviation from upstream sample

Upstream's `runtime/cpp/instance_segmentation/instance_segmentation.cpp` calls
`pad_crop_to_target` (`common/toolbox.cpp:919`) for inference-side preprocessing
— a no-resize pad-or-crop that just zero-pads if the input is smaller than the
model and **top-left crops** if larger. That is fine for inputs already at
640×640 but silently destroys content for typical native-resolution images
(e.g. a 1920×1080 frame becomes its top-left 640×640 corner).

The shim instead uses `make_model_space_canvas` semantics
(`instance_seg_postprocess.cpp:120`) — aspect-preserving resize + pad
bottom/right with zeros. This matches:

- The function the upstream code itself uses to **undo** the geometry on the
  postprocess side.
- The conventional YOLO letterbox.
- What real Hailo deployments (TAPPAS GStreamer pipelines using `videoscale` +
  `videobox`) do in production.

The pre/post symmetry matters: TAPPAS's `decode_masks` resizes proto masks to
whatever dims you pass to `filter()`. The shim passes model dims so masks come
out in letterbox space, then per-mask unmap (`unmap_mask`) applies the inverse
LetterboxMap to recover original-image coordinates. Boxes get the same
treatment via a scale/clamp.

## Hard-coded thresholds (upstream)

The vendored postprocess source bakes:

- `SCORE_THRESHOLD = 0.6f` (`instance_seg_postprocess.cpp:11`)
- `IOU_THRESHOLD = 0.7f` (`instance_seg_postprocess.cpp:12`)
- `NUM_CLASSES = 80` (`instance_seg_postprocess.cpp:13`)
- `regression_length = 15` (DFL bins − 1 → 64ch boxes; `instance_seg_postprocess.cpp:614`)
- `strides = {8, 16, 32}`, `network_dims = {640, 640}` (`instance_seg_postprocess.cpp:615-616`)

These are baked because the comparison target *is* this exact code path. If we
later want CLI parity with HAL's `--score-threshold` / `--iou-threshold`, the
clean route is to patch `instance_seg_postprocess.cpp` to take the values as
parameters and update `../README.md` (currently "verbatim, no modifications").

## HEF compatibility caveat

The upstream postprocess detects "HEF with HailoRT-Postprocess" mode by
`get_output_vstream_infos_size() == 1` and switches to a packed-NMS fast path
(`get_detections` / `draw_detections_and_mask` in the `.cpp`). Our
hailo-converter v0.3.0 emits 10 raw tensors, so the shim falls into the
`segmentation_postprocess` raw-heads path — which is exactly the comparison we
want for host-side decode against HAL.

The raw-heads path also assumes specific tensor characteristics:
- Protos identified by `features() == 32 && height() == 160 && width() == 160`
  (`pop_proto`, `instance_seg_postprocess.cpp:516`).
- 3 detection scales with the per-scale triple ordering
  `[boxes_s, scores_s, mask_coefs_s]`.

If our v0.3.0 HEF tensor names/orderings don't match, the shim throws during
`filter()` rather than producing silently-wrong results — verify by checking
`backend.output_names` against the upstream's expectations on first use against
both the Hailo Model Zoo reference HEF and our re-converted HEF.
