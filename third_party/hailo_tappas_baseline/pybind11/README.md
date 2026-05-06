# pybind11 shim — `hailo_tappas_baseline`

Thin Python module exposing the upstream TAPPAS YOLOv8/v5-seg postprocess as a
single class, `HailoTappasBackend`, that drives HailoRT directly (sync, no
GStreamer) and returns NumPy arrays of detections + masks. Imported by
`ara2_validator.hailo_pipeline`.

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
`third_party/hailo_tappas_baseline/pybind11/hailo_tappas_baseline*.so` so
`ara2_validator/hailo_pipeline.py` can import it once the directory is on
`sys.path`.

## Module API

```python
from hailo_tappas_baseline import HailoTappasBackend

backend = HailoTappasBackend("yolov8n-seg.hef")
backend.model_width, backend.model_height, backend.model_channels  # e.g. 640, 640, 3
backend.input_names, backend.output_names                          # introspection

# input: uint8 NHWC array, already letterboxed to (model_h, model_w, model_c).
# org_height / org_width: original image dims; masks are returned at this size.
result = backend.infer(input_array, org_height=720, org_width=1280)

result["boxes"]    # (N, 4) float32, xyxy normalized to [0, 1]
result["scores"]   # (N,)   float32
result["classes"]  # (N,)   int32, 0-indexed COCO IDs
result["masks"]    # list[float32 (org_h, org_w)] post-sigmoid, bbox-cropped
result["timings"]  # {"inference_ms": float, "postprocess_ms": float}
```

The GIL is released around the HailoRT inference and the xtensor postprocess so
multiple Python threads can drive the backend in parallel if needed.

## Hard-coded thresholds (upstream)

The vendored postprocess uses fixed thresholds:

- `SCORE_THRESHOLD = 0.6f` (instance_seg_postprocess.cpp:11)
- `IOU_THRESHOLD = 0.7f` (instance_seg_postprocess.cpp:12)
- `regression_length = 15` (DFL bins − 1, so 64ch boxes; instance_seg_postprocess.cpp:614)
- `strides = {8, 16, 32}` and `network_dims = {640, 640}` (instance_seg_postprocess.cpp:615-616)

For now these are baked in. If we need configurable thresholds for parity with
HAL's `--score-threshold` / `--iou-threshold` flags, the cleanest path is to
patch the upstream source with thresholds-as-parameters and document the diff
in `../README.md` (currently "verbatim, no modifications").

## HEF compatibility caveat

The upstream postprocess detects "HEF with HailoRT-Postprocess" mode by
`infer_model.get_output_vstream_infos_size() == 1` and switches to a packed-NMS
fast path (`get_detections` / `draw_detections_and_mask` in the .cpp).
hailo-converter v0.3.0 emits 10 raw tensors, so the shim falls into the
`segmentation_postprocess` raw-heads path — which is what we want for a fair
host-side decode comparison against HAL.

The raw-heads path also assumes specific tensor characteristics:
- protos identified by `features() == 32 && height() == 160 && width() == 160`
  (see `pop_proto`, line 516)
- 3 detection scales (3 boxes + 3 scores + 3 mask_coefs tensors) with the
  layout `[boxes_s, scores_s, mask_coefs_s]` order in the per-scale triple

If our v0.3.0 HEF tensor names/orderings don't match, the shim will throw
during `filter()` rather than producing wrong results — verify by checking
`backend.output_names` against the upstream's expectations on first use.
