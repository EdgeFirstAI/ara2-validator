# Changelog

All notable changes to `ara2-validator` are documented here. Format is loosely based on [Keep a Changelog](https://keepachangelog.com/) and this project follows [Semantic Versioning](https://semver.org/) once a first tagged release is cut.

## [Unreleased] — 2026-05-01

This is the first publication-grade revision. The validator has been rewritten end-to-end against the EdgeFirst HAL Optimization Guide, matched to the Ultralytics YOLOv8-seg reference, and benchmarked on imx8mp-frdm against both ONNX FP32 and the NXP dvapi reference implementation.

### Added

- **`HalPipeline` class** (`hal_pipeline.py`) — one-stop HAL setup encapsulating `ImageProcessor`, `Decoder`, and DMA-BUF output tensor wrapping. Encodes the construct-once / reuse-many pattern that HAL Optimization Guide Rules 1, 4, and 5 expect; customers porting to EdgeFirst HAL can copy the class verbatim.
- **`MaskMode` enum** (`postprocess_hal.MaskMode`) — typed selector for `RETINA` (`MaskResolution.Scaled`, default) vs `FAST` (`MaskResolution.Proto` + `cv2.resize`).
- **`MaskTimings` / `MaskResult` dataclasses** — replaces the dict-shaped `timings` mutate-in-place arg with structured per-stage timings (`materialize_hal_ms`, `paste_ms`, `rle_ms`).
- **`materialize_masks_for_coco_rle()` / `_arrays()`** — split the return-value polymorphism in `materialize_masks_for_coco()` into two explicit entry points. The RLE variant reuses one `(orig_h, orig_w)` canvas across detections and encodes inline, removing the per-detection ~1 MB allocation that hurt latency on high-resolution inputs.
- **`--fast-masks` flag** on `edgefirst.py`, `reference.py`, `onnx_reference.py`, and `__main__.py` — explicit opt-in to the lower-accuracy / lower-latency mask path. Default (no flag) is the retina path.
- **Shared `_stats` module** — `trimmed_stats()`, `fmt_row()`, `summary_header()` hoisted out of the three entry points so the per-stage timing summaries stay byte-for-byte identical across scripts.
- **EdgeFirst HAL version banner** at startup of the HAL backend so benchmark logs disambiguate the materialiser path (scalar vs batched-GEMM landed in HAL 0.18.0; rayon-parallel + per-detection logit-precompute restoration landed in 0.18.1, which is this validator's floor).
- **`logging` adoption** across all entry points — replaces ad-hoc `print(..., file=sys.stderr)` so users can silence per-image lines with `--log-level WARNING`.
- **Regression tests** (`tests/test_hal_path.py`) for the new `MaskMode` API surface, the split helpers, the back-compat shim's `DeprecationWarning`, the `HalPipeline` import, and the `unletterbox_box_array` round-trip.
- **Deployment-style benchmark** (`tests/bench_draw_masks.py`) that exercises HAL's fused `ImageProcessor.draw_masks` (decode + materialize + composite in one Rust/GPU call) over the same coco128 set the validator uses. Measures the mask path a live pipeline actually takes, distinct from the validation-only `materialize_masks` + RLE path.

### Changed

- **NPU is now referred to as Ara240, vendor as NXP** throughout user-facing documentation and source docstrings. Library and package names (`ara2-validator`, `ara2-rs`, `edgefirst-ara2`) are unchanged.
- **`MaskResolution.Scaled` is the default** for HAL mask materialisation (retina, matches Ultralytics `process_mask_native`). Closes a previously-observed −13.8 pp mask mAP gap from the buggy threshold-then-upsample-binary path.
- **Postprocess timing now includes the mask decode** in `reference.py` so it can be compared apples-to-apples with the edgefirst HAL postprocess. RLE encode is reported separately as output formatting.
- **`HalPipeline.__init__` always calls `model.allocate_tensors("dma")`** — closes the silent zero-copy break in `__main__.py --backend hal`, where `from_fd()` would previously wrap a non-DMA tensor.
- **`__main__.py` and `edgefirst.py` share `HalPipeline`** — eliminates ~120 lines of duplicated decoder / output-tensor setup.
- **README** rewritten with results-first ordering, EdgeFirst Studio scope note, Mermaid architecture diagram, deployment-style timing example at `--score-threshold 0.5`, GitHub admonitions for the dvapi limitation and validation-threshold caveat, NPU sub-timing table with measured-roundtrip total, and an explanation of how the EdgeFirst HAL path uses the dvproxy DMA-BUF surface.

### Removed

- **`postprocess_hal.postprocess_hal()`, `_bridge_tensors()`, `_build_quant_list()`, `build_hal_metadata()`** — dead code from an earlier refactor.
- **`--retina-masks` and `--mask-mode` flags** on `edgefirst.py` — collapsed into a single `--fast-masks` opt-out (retina by default).
- **`--offset` flag** on `reference.py` — the dvapi.py + libaraclient stack is unstable past ~100 sequential inferences in a single process; chunking around it added complexity without solving the underlying issue. The script now documents the limitation in the `--max-images` help text and recommends the HAL path for full val5k runs.
- **`reference.decode_outputs()`, `dequantize`, `xcycwh_to_xyxy`, `unletterbox_boxes`, `trimmed_stats`, `fmt_row`** — unused functions superseded by `postprocess_numpy` and `_stats`.

### Performance

- **75× speedup on `materialize_hal`** (569 ms → 7.5 ms on imx8mp-frdm, N=119) via the HAL batched-GEMM kernel, rayon-parallel + per-detection logit-precompute path (0.18.1), and further mask decoding + NMS decode optimizations (0.18.2).
- **End-to-end on imx8mp-frdm coco128** drops from 452 ms (`reference` retina) to 32 ms (`edgefirst` retina), with mask mAP within 0.6 pp of the dvapi reference.
- **~10–30 ms saved per image** by reusing the `(orig_h, orig_w)` mask canvas instead of allocating per-detection.

### Validation

- **coco-val5k accuracy** measured against ONNX FP32 ceiling on imx8mp-frdm: −5.37 pp box mAP, −0.45 pp mask mAP, +0.55 pp mAR@100. Box drift is the int8 quantisation cost on the bbox path; mask drift is int8 plus a small numerical-precision offset. mAR@100 is essentially identical (within noise).
- **coco128 timing** unified table on imx8mp-frdm covers ONNX FP32, dvapi reference (retina + fast), and HAL edgefirst (retina + fast) — all three on-target so timings are directly comparable.
- **Deployment-style timing** measured via HAL's fused `ImageProcessor.draw_masks` at `--score-threshold 0.5` on the same coco128 set: 41 ms end-to-end on imx8mp-frdm, 38 ms on imx95-frdm. Reference live pipeline: `ara2-rs/examples/yolov8_live.py`.

### Dataset

- **`coco128-seg.zip`** published at `https://repo.edgefirst.ai/datasets/coco128-seg.zip` — same 128 train2017 images as the upstream Ultralytics coco128.zip but bundled with the COCO segmentation annotations (filtered from `instances_train2017.json`) so the dataset can be used directly with `pycocotools.COCOeval(iou_type="segm")`.

### Known issues

- **dvapi.py heap corruption** past ~100 sequential inferences in a single process. Documented in `reference.py` and the README; HAL path is unaffected and is the recommended route for full val5k.
- **No mask-edge fidelity match with Ultralytics** — int8 quantisation produces protos that are near-identical but not bit-identical to the FP32 reference, so per-mask IoU is in the 0.95–0.99 range rather than 1.0. This is reflected in the −0.45 pp mask mAP drift figure.
