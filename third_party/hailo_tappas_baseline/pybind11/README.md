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

The shim's `letterbox_for_model` (and the `letterbox_for_test` accessor that
wraps it) does an **Ultralytics-faithful YOLO letterbox** — aspect-preserving
resize with `cv::INTER_LINEAR`, centered padding split evenly between
top/bottom and left/right (with the bottom/right side taking the extra pixel
on odd-total pads), and `(114, 114, 114)` gray fill. This matches the
`LetterBox(center=True)` class in `ultralytics/data/augment.py:1542` (the
default for Ultralytics' YOLO validation) and the validator's
`ara2_validator/preprocess.letterbox()`.

It deviates from upstream's preprocessing in two places. The upstream
`runtime/cpp/instance_segmentation/instance_segmentation.cpp` calls
`pad_crop_to_target` (`common/toolbox.cpp:919`) for inference-side
preprocessing — a no-resize pad-or-crop that zero-pads small inputs and
**top-left crops** larger ones, fine for inputs already at 640×640 but
silently destroys content for typical native-resolution images (e.g. a
1920×1080 frame becomes its top-left 640×640 corner). Upstream's helper
`make_model_space_canvas` (`instance_seg_postprocess.cpp:120`), which is what
the upstream code uses to undo the letterbox on the postprocess side, is
closer to correct but still differs from Ultralytics on three counts:
interpolation (`cv::INTER_AREA` instead of `cv::INTER_LINEAR`), padding
placement (bottom/right anchored instead of centered), and fill colour
(`(0,0,0)` black instead of `(114,114,114)` gray).

The shim therefore does its own letterbox that is symmetric with the
postprocess unmap math (also living in the shim, in the box-unmap loop
inside `infer()` and in `unmap_mask`). The vendored `upstream/` tree is
not modified — these edits live entirely in `pybind11/hailo_pybind.cpp`.

Mask materialization in the shim path is "retina-mode" (continuous-sigmoid
masks bilinear-upsampled to original-image resolution before any threshold)
by construction:
upstream's `decode_masks` produces continuous-sigmoid masks at proto
resolution, bilinear-upsamples them to model space (640×640) via the
`(org_image_width, org_image_height)` argument we pass as `(model_w, model_h)`,
then bbox-crops in normalized letterbox coords (the bbox is letterbox-
normalized at this point because we haven't unmapped yet). The shim's
`unmap_mask` then crops the centered valid region from the 640×640 mask
and bilinear-resizes to original-image dimensions.

## Hard-coded model parameters (upstream)

The vendored postprocess still hard-codes assumptions about the model itself,
which is fine for a yolov8n/m-seg baseline but worth flagging:

- `NUM_CLASSES = 80` (`instance_seg_postprocess.cpp:13`)
- `regression_length = 15` (DFL bins − 1 → 64ch boxes; `instance_seg_postprocess.cpp:614`)
- `strides = {8, 16, 32}`, `network_dims = {640, 640}` (`instance_seg_postprocess.cpp:615-616`)

Score and IoU thresholds **are** runtime-tunable; the shim's `infer()` accepts
`score_threshold` and `iou_threshold` keyword arguments which forward to a
local 5-arg overload of upstream's `filter()`. See `../README.md` for the
diff against upstream.

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
