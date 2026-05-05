# Ara240 Validator

> **YOLOv8-seg COCO validation on the NXP Ara240 NPU, end-to-end.** Three reference inference paths (FP32 ONNX host, dvapi+numpy on-target, EdgeFirst HAL on-target) producing pycocotools-evaluated COCO JSON, with per-stage timings on each.

| | What it shows | Where it runs |
|---|---|---|
| **`onnx_reference`** | FP32 ONNX accuracy ceiling | any CPU/CUDA host, or on-target |
| **`reference`** | int8 NPU baseline using NXP dvapi.py | imx8mp-frdm, imx95-frdm |
| **`edgefirst`** | int8 NPU + GPU letterbox + HAL Decoder | imx8mp-frdm, imx95-frdm |

This validator augments the NXP-supplied dvapi reference with EdgeFirst HAL acceleration: GPU letterbox preprocessing rendered directly into the NPU input buffer, detection decode and mask materialisation in compiled Rust, all sharing the dvproxy DMA-BUF surface for zero-copy data movement. It is an **offline benchmarking + validation tool**, not a production streaming application — each pipeline stage runs serially so per-stage cost is captured cleanly. For live camera/video inference with EdgeFirst, see the [ara2-rs examples](https://github.com/EdgeFirstAI/ara2-rs/tree/main/examples).

## Scope of this demonstration

This repository validates **a single model (YOLOv8n-seg) on a single NPU (NXP Ara240) for a single task (COCO instance segmentation)**.

For multi-model, multi-sensor, multi-platform productisation — data labelling, training, deployment, and OTA at fleet scale — see **[EdgeFirst Studio](https://edgefirst.studio/)**. Studio ships these workflows end-to-end and supports target hardware beyond Ara240 (i.MX 8M Plus, i.MX 95, Raspberry Pi + Hailo, NVIDIA Jetson) with first-class support for radar, lidar, and camera fusion.

## Results

### coco-val5k (5000 images, imx8mp-frdm)

The full COCO val2017 set. Drift columns are absolute **percentage points** (`mAP_a − mAP_b`).

| Run | Box mAP | Box mAP@50 | Mask mAP | Mask mAP@50 | Mask mAR@100 |
|---|---:|---:|---:|---:|---:|
| `onnx_reference` retina (FP32, host CPU) | **0.3605** | 0.5119 | **0.2801** | 0.4716 | 0.4115 |
| `edgefirst` retina (HAL int8) | 0.3068 | 0.4745 | 0.2756 | 0.4549 | 0.4170 |
| **drift (pp)** | **−5.37** | −3.74 | **−0.45** | −1.67 | +0.55 |

> [!NOTE]
> The dvapi.py reference is excluded from the val5k row because the underlying `libaraclient_aarch64.so` ctypes wrapper hits a glibc `double free or corruption` abort after roughly 100–200 sequential inferences in a single process. coco128 (128 images) generally completes; coco-val5k (5000) does not. The Rust-backed `edgefirst-ara2` wrapper used by `edgefirst.py` is unaffected and is the route used for the val5k row.

### coco128 — per-stage timing (128 images)

All scripts run on-target so the per-stage timings are directly comparable. Two tables are shown: the first uses a **validation** score threshold of `0.001` (required for correct mAP computation — see note below); the second uses a **deployment** threshold of `0.5`, representative of real-time inference.

> [!IMPORTANT]
> **Decode and Mask timings are strongly threshold-dependent.** At the validation threshold of `0.001`, the model retains ~100–300 candidate detections per frame for decode and ~5–15 surviving detections for mask materialisation. At a deployment threshold of `0.5`, only ~2–5 high-confidence detections survive, and both Decode and Mask drop accordingly. The validation table represents a worst-case timing bound; the deployment table represents real-world performance.

> [!NOTE]
> **Why `0.001`?** pycocotools computes mAP by integrating the precision-recall curve over 101 recall levels. This requires submitting *all* detections that might contribute to any recall level — including low-confidence true positives that only matter at high-recall operating points. A threshold of `0.5` truncates the P-R curve early, causing mAP to be underestimated (see the mAP columns below). The `0.001` threshold is the Ultralytics `val` default and is required for comparable, standards-compliant COCO evaluation.

#### Validation threshold (`--score-threshold 0.001`)

| Script | Mask | Box mAP | Mask mAP | Pre | Inf | Decode | Mask | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **imx8mp-frdm** | | | | | | | | |
| `onnx_reference` (FP32, on-target CPU) | retina | **0.4533** | **0.3444** | 1.7 ms | 46.3 ms | — | 47.6 ms | 95.7 ms |
| `onnx_reference` (FP32, on-target CPU) | fast | 0.4533 | 0.3337 | 1.8 ms | 44.6 ms | — | 47.8 ms | 94.1 ms |
| `reference` (dvapi+numpy) | retina | 0.3941 | 0.3278 | 21.1 ms | 18.0 ms | — | 412.9 ms | 452.2 ms |
| `reference` (dvapi+numpy) | fast | 0.3941 | 0.3221 | 20.6 ms | 18.2 ms | — | 293.4 ms | 332.3 ms |
| `edgefirst` (HAL) | retina | 0.3955 | 0.3218 | 6.2 ms | 13.5 ms | 2.6 ms | 9.7 ms | **32.0 ms** |
| `edgefirst` (HAL) | fast | 0.3955 | 0.3095 | 6.2 ms | 13.5 ms | 2.6 ms | 3.8 ms | **26.1 ms** |
| **imx95-frdm** | | | | | | | | |
| `edgefirst` (HAL) | retina | 0.3966 | 0.3231 | 4.1 ms | 11.3 ms | 2.6 ms | 7.6 ms | **25.5 ms** |
| `edgefirst` (HAL) | fast | 0.3966 | 0.3103 | 4.1 ms | 11.3 ms | 2.5 ms | 2.9 ms | **20.8 ms** |

#### Deployment threshold (`--score-threshold 0.5`)

| Script | Mask | Box mAP | Mask mAP | Pre | Inf | Decode | Mask | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **imx8mp-frdm** | | | | | | | | |
| `edgefirst` (HAL) | retina | 0.2778 | 0.2434 | 6.1 ms | 13.4 ms | 2.2 ms | 3.5 ms | **25.2 ms** |
| `edgefirst` (HAL) | fast | 0.2778 | 0.2364 | 6.2 ms | 13.4 ms | 2.3 ms | 1.8 ms | **23.6 ms** |
| **imx95-frdm** | | | | | | | | |
| `edgefirst` (HAL) | retina | 0.2793 | 0.2475 | 3.9 ms | 11.3 ms | 2.1 ms | 3.3 ms | **20.5 ms** |
| `edgefirst` (HAL) | fast | 0.2793 | 0.2395 | 3.9 ms | 11.3 ms | 2.2 ms | 1.5 ms | **18.9 ms** |

The `0.5` threshold reduces mAP by ~30% (0.40 → 0.28 Box, 0.32 → 0.24 Mask) because pycocotools can no longer integrate the full precision-recall curve — high-recall operating points are unreachable when low-confidence true positives are discarded before submission. This is an evaluation artefact, not a model quality difference: the model's actual detection capability is unchanged, only the evaluator's ability to measure it is truncated.

`Decode` = HAL `nms_decode` (dequantisation, top-K filtering, box decode, class-aware NMS, proto extraction). `Mask` = mask materialisation (proto matmul, sigmoid, crop, resize). `End-to-end` = Pre + Inf + Decode + Mask. The `reference` and `onnx_reference` rows don't split Decode from Mask because Ultralytics performs both in a single fused `postprocess()` call. The dvapi+numpy reference is omitted from the imx95 block because the dvproxy / `libaraclient` stack on that board needs intermittent restarts.

#### NPU inference breakdown

The `Inf` column above is the wall-clock time `model.run()` takes — input-in, NPU compute, output-out, plus whatever host-side staging the backend requires. The NPU driver also reports per-stage sub-timings the firmware actually spent on the device. Since the NPU runs the same model the same way for both backends, the firmware sub-timings are essentially identical; the difference between the two `Inf` numbers is host-side staging cost, which is what the DMA-BUF zero-copy path eliminates.

| | DMA host→device | NPU compute | DMA device→host | Driver subtotal | Wall-clock roundtrip | Host overhead |
|---|---:|---:|---:|---:|---:|---:|
| `reference` (dvapi) on imx8mp | 2.17 ms | 4.40 ms | 4.66 ms | 11.23 ms | **17.94 ms** | 6.71 ms |
| `edgefirst` (ara2-rs) on imx8mp | 2.21 ms | 4.40 ms | 4.83 ms | 11.44 ms | **13.33 ms** | 1.89 ms |
| `edgefirst` (ara2-rs) on imx95 | 1.96 ms | 4.39 ms | 2.96 ms | 9.31 ms | **11.22 ms** | 1.91 ms |

The ~4.6 ms host-overhead delta between dvapi and edgefirst on imx8mp is the DMA-BUF benefit. The dvapi path moves the input tensor through a host buffer (numpy → libaraclient ctypes → dvproxy) and the output tensor back the same way; the edgefirst path uses `Tensor.from_fd` to wrap the dvproxy DMA-BUF directly, so the GPU letterbox writes the input DMA-BUF in place and HAL's Decoder reads the output DMA-BUF in place — the host never touches either buffer. The `HalPipeline` class is what wires this up; see its docstring for the construct-once / reuse-many invariants HAL Optimization Guide Rules 1, 4, and 5 require to keep the path zero-copy.

#### Post-processing breakdown (Decode + Mask)

The `Decode` and `Mask` columns in the timing tables represent the two phases of post-processing after NPU inference completes. Both operate entirely on the CPU (Cortex-A53 on imx8mp, Cortex-A55 on imx95) against the NPU output DMA-BUF.

All sub-step timings below were measured on the **imx95-frdm** (Cortex-A55 @ 1.8 GHz)
using HAL's built-in Perfetto tracing instrumentation (`hal.Tracing()` context manager).

**Decode** (`decode` in HAL) transforms the raw quantized NPU output grid into a filtered list of detections:

| Sub-step | What it does | Time (0.5 thr) | Time (0.001 thr) |
|----------|-------------|:--------------:|:-----------------:|
| Score filter + argmax | Dequantize 8400×80 int8 score grid, find per-anchor class argmax, threshold filter | 0.81 ms | 0.86 ms |
| Top-K pre-filter | Keep top 300 highest-scoring candidates (O(N) partial sort) | 0.009 ms | 0.014 ms |
| Class-aware NMS | Per-class IoU suppression (greedy, threshold 0.7) | 0.009 ms | 0.109 ms |
| Box dequantisation | Dequantize int8 box coords for surviving candidates | 0.002 ms | 0.003 ms |
| Proto extraction | Copy mask coefficients + 819 KB proto tensor (NCHW, no transpose) | 0.72 ms | 0.95 ms |
| **Total decode** | | **1.65 ms** | **2.05 ms** |

The score filter dominates decode cost — it must scan all 8400 anchors × 80 classes regardless of threshold. NMS cost scales quadratically in per-class survivors but is negligible when top-K caps candidates at 300.

**Mask Materialize** (`materialize_masks` in HAL) produces per-detection binary masks from the prototype tensor and per-detection mask coefficients:

| Sub-step | What it does | Cost driver |
|----------|-------------|-------------|
| Proto matmul | For each detection: dot product of 32 mask coefficients × 32-channel prototype at each output pixel | Linear in detections × output resolution |
| Sigmoid + crop | Apply σ(x), zero pixels outside detection bbox | Negligible |
| Bilinear resize | Upscale from 160×160 ROI to output dimensions (**scaled** mode only) | Proportional to output mask area |

Measured mask timings (imx95-frdm, **scaled** mode, 0.5 threshold):
- 1 detection: **1.0–2.6 ms** (median 2.6 ms)
- 2 detections: **1.8–5.2 ms** (median 2.4 ms)
- 4 detections: **5.3 ms**

At validation threshold (0.001, ~5–15 detections): mean **7.8 ms**, max **19.6 ms**.
At deployment threshold (0.5, ~1–4 detections): mean **3.7 ms**, max **8.7 ms**.

> **Measurement method**: Sub-step times captured using HAL's built-in Perfetto
> tracing (`hal.Tracing()` context manager) which writes Chrome JSON trace files
> viewable at [ui.perfetto.dev](https://ui.perfetto.dev/). Tracing adds < 1%
> overhead (single atomic load per span site when no subscriber is active).

### What "end-to-end" measures

`End-to-end` covers the per-frame work after image acquisition:

1. **preprocess** — letterbox resize, normalise, quantise, lay out as the NPU's input tensor.
2. **inference** — submit input, run NPU, retrieve output.
3. **decode** — dequantise the detection grid, apply top-K pre-filtering, decode box coordinates, run class-aware NMS, extract proto data.
4. **mask materialisation** — per-detection proto coefficient × prototype matmul, sigmoid activation, crop to bbox, resize to original resolution.

Stages excluded from end-to-end because they are validation-only or environment-specific:

- **image decode** — JPEG → RGB buffer. In a real pipeline the frame arrives from a camera or video decoder, not from disk.
- **output formatting** — pycocotools RLE encoding for the COCO results JSON. A deployment would consume the binary masks directly.
- **COCO evaluation** — pycocotools per-image mAP computation. Strictly an offline validation step.

### Fused draw_masks timing

The tables above measure *validation* mask cost: the validator calls HAL's `materialize_masks` per detection and then RLE-encodes each binary mask for pycocotools. A live deployment doesn't do that — it composites masks straight onto a display canvas via HAL's fused `ImageProcessor.draw_masks`, which decodes, materialises, and blends in one Rust/GPU call without ever round-tripping per-detection masks through Python.

`tests/bench_draw_masks.py` runs the same 128-image set through that fused path at `--score-threshold 0.5`:

| Stage | imx8mp-frdm | imx95-frdm |
|---|---:|---:|
| preprocess (GPU) | 5.85 ms | 3.66 ms |
| inference (wall) | 13.28 ms | 11.28 ms |
| draw_masks (decode + materialize + render) | 9.51 ms | 9.51 ms |
| **end-to-end (pre + inf + draw)** | **28.65 ms** | **24.45 ms** |

For the canonical live-pipeline pattern (camera → NPU → fused draw_masks → Wayland compositor) see [`ara2-rs/examples/yolov8_live.py`](https://github.com/EdgeFirstAI/ara2-rs/blob/main/examples/yolov8_live.py); for the API surface see HAL's `ImageProcessor.draw_masks` and `ImageProcessor.draw_decoded_masks`. The *EdgeFirst for i.MX Reference Manual* documents the full `edgefirstoverlay` per-stage timing methodology that this script mirrors.

## Serial vs concurrent pipelines

The end-to-end numbers above come from a **serial** execution model: each stage runs to completion before the next begins, one frame at a time. That is what makes the per-stage breakdown additive — the total is exactly the sum of the parts, with no overlap to disentangle, which is the entire point of a benchmarking + validation tool.

![Serial pipeline timeline](assets/diagram_serial_execution.png)

Drawn as a waterfall, the headroom becomes obvious: while any single stage runs, every other stage is idle. The GPU sits unused during NPU inference; the NPU sits unused during preprocess; the host sits unused during DMA transfers. With the Decode/Mask split visible, inference at 13 ms is the longest single stage; Decode (2.6 ms) and Mask (7.6 ms) are independent of the GPU and NPU and could overlap with the next frame's preprocess or inference in a pipelined deployment.

![Pipeline waterfall](assets/diagram_pipeline.waterfall.png)

A throughput-tuned deployment can overlap stages across consecutive frames — start preprocessing frame *N+1* the moment frame *N*'s preprocess finishes, regardless of whether frame *N*'s NPU inference is still running. With four pipeline stages (Pre 6 ms | Inf 13 ms | Decode 2.6 ms | Mask 9.7 ms), once the pipeline is filled the throughput *period* collapses to the **slowest single stage** (13.5 ms, inference → 74 FPS). Decode and Mask together (12.3 ms) fit inside the inference window, so they are fully hidden in the concurrent case.

![Concurrent pipeline](assets/diagram_pipeline_overlap.png)

Per-frame **latency** is unchanged. Any one frame still has to walk through every stage in order, and the end-to-end clock from sensor capture to rendered output is identical to the serial case. Concurrent pipelining is therefore a *throughput* optimisation; it does not make any individual frame complete faster, which is why latency-critical paths (closed-loop control, AR overlays) cannot escape the per-stage budget by adding parallelism. Implementing the concurrent pipeline is out of scope for this validator — overlap would obscure the per-stage attribution this tool exists to capture — but the pattern is documented in the EdgeFirst HAL and `ara2-rs` examples.

### How the validator pipeline mirrors deployment

Execution scheduling aside, the validator's data path is the same path a live camera pipeline uses on this hardware. Frames are loaded from disk rather than from a sensor, but they are loaded straight into **DMA-BUF-backed EdgeFirst tensors** — the same tensor type `edgefirst-ara2` produces from a V4L2 capture. From the GPU letterbox onward, the validator and a live pipeline see identical surfaces: the GPU writes the NPU input DMA-BUF in place, the NPU produces an output DMA-BUF, HAL's Decoder reads it without staging, and HAL's mask materialisation runs against the same proto tensor. Swapping the disk loader for a camera capture is the only change needed to repurpose `HalPipeline` as the inner loop of a live application — and that is exactly the shape of [`ara2-rs/examples/yolov8_live.py`](https://github.com/EdgeFirstAI/ara2-rs/blob/main/examples/yolov8_live.py).

One caveat is worth calling out: **both Decode and Mask stages are threshold-dependent.** The validation table (threshold `0.001`) retains many more candidate detections through NMS and materialises more masks than a deployment would. Compare the two tables above: at `0.5`, Decode drops from 2.6 ms to 2.2 ms (fewer candidates to sort) and Mask drops from 9.7 ms to 3.5 ms (fewer surviving detections to materialise). For the fused `draw_masks` deployment path, see [Fused draw_masks timing](#fused-draw_masks-timing).

## Installation

The validator and its HAL extra install in one step from this repository.

### On-target (imx8mp-frdm or imx95-frdm)

```bash
ssh imx8mp-frdm
python3 -m venv /root/venv
source /root/venv/bin/activate
pip install --upgrade pip
pip install 'ara2-validator[hal] @ git+https://github.com/EdgeFirstAI/ara2-validator.git'
```

The `[hal]` extra pulls [`edgefirst-hal>=0.18.2`](https://github.com/EdgeFirstAI/hal) and `edgefirst-ara2`. The `>=0.18.2` floor is required: 0.17.x ships a scalar `materialize_masks` (~569 ms / image at N=119 detections), 0.18.0 regressed rayon parallelism (PR #51), 0.18.1 restored it (~33 ms / image), and 0.18.2 adds detection decode, NCHW proto layout elimination, and mask materialisation optimizations (Decode 2.6 ms + Mask 9.7 ms on imx8mp-frdm retina at validation threshold).

### Host (FP32 ONNX baseline)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install 'ara2-validator[onnx] @ git+https://github.com/EdgeFirstAI/ara2-validator.git'
```

### Dataset

`coco128-seg` (128 train2017 images plus filtered COCO segmentation annotations) is hosted at:

```bash
wget https://repo.edgefirst.ai/datasets/coco128-seg.zip
unzip coco128-seg.zip -d /root/
```

For coco-val5k:

```bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip -d /root/
unzip annotations_trainval2017.zip -d /root/
```

## Quick start

```bash
python -m ara2_validator.edgefirst \
    --model yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128-seg/images/ \
    --gt /root/coco128-seg/annotations/instances_train2017.json \
    --with-masks
```

This completes in about 40 s on imx8mp-frdm (warmup + 128 inferences + COCO evaluation). Add `--fast-masks` for the latency-tuned mask path.

## Running validation benchmarks

This section documents the full workflow for running coco128-seg validation on the target boards with a local HAL wheel build.

### Board prerequisites

| Board | SSH alias | Python | NPU service | Venv |
|-------|-----------|--------|-------------|------|
| imx8mp-frdm | `ssh imx8mp-frdm` | 3.13 | `ara2.service` | `/root/venv` (plain) |
| imx95-frdm | `ssh imx95-frdm` | 3.13 | `ara2.service` | `/root/venv` (`--system-site-packages`) |

Both boards must have the NPU service running (`systemctl is-active ara2.service` → `active`). The imx95-frdm frequently requires multiple reboots (3–5) to bring the NPU online due to PCIe link instability; if `ara2.service` shows `device handle error`, reboot and retry.

### Deploying a local HAL wheel

The validator cannot be `pip install`-ed directly on-target (setuptools flat-layout error from the `assets/` directory). Instead, rsync the code and install the HAL wheel separately:

```bash
# Sync validator source to both boards
rsync -a --delete --exclude='*.pyc' --exclude='__pycache__' \
    ara2_validator/ imx8mp-frdm:/root/ara2-validator/ara2_validator/
rsync -a --delete --exclude='*.pyc' --exclude='__pycache__' \
    ara2_validator/ imx95-frdm:/root/ara2-validator/ara2_validator/

# Deploy HAL wheel (cp38-abi3 is compatible with Python 3.8+)
HAL_WHEEL=~/software/hal/target/wheels/edgefirst_hal-0.18.1-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

scp "$HAL_WHEEL" imx8mp-frdm:/tmp/
ssh imx8mp-frdm '/root/venv/bin/pip install --force-reinstall --no-deps --no-cache-dir /tmp/edgefirst_hal-*.whl'

scp "$HAL_WHEEL" imx95-frdm:/tmp/
ssh imx95-frdm '/root/venv/bin/pip install --force-reinstall --no-deps --no-cache-dir /tmp/edgefirst_hal-*.whl'
```

> [!WARNING]
> The imx95-frdm venv uses `--system-site-packages`. If an older HAL version is installed system-wide at `/usr/lib/python3.13/site-packages/edgefirst_hal/`, Python will prefer a version-specific `.cpython-313-*.so` over the venv's `.abi3.so`. Remove any system-wide copy before benchmarking:
> ```bash
> ssh imx95-frdm 'rm -rf /usr/lib/python3.13/site-packages/edgefirst_hal*'
> ```

### Verifying the installed wheel

```bash
# Confirm the correct .so is loaded from the venv
ssh imx8mp-frdm '/root/venv/bin/python3 -c "import edgefirst_hal; print(edgefirst_hal.__file__)"'
# Expected: /root/venv/lib/python3.13/site-packages/edgefirst_hal/__init__.py

# Compare installed .so against wheel contents
ssh imx95-frdm 'md5sum /root/venv/lib/python3.13/site-packages/edgefirst_hal/edgefirst_hal.abi3.so'
```

### Running the 4 benchmark configurations

All runs use `PYTHONPATH` to pick up the rsynced validator code:

```bash
# imx8mp retina (full-resolution masks)
ssh imx8mp-frdm 'cd /root/ara2-validator && PYTHONPATH=/root/ara2-validator \
    /root/venv/bin/python3 -m ara2_validator.edgefirst \
    --model /root/yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128/images/ \
    --gt /root/coco128/annotations/instances_train2017.json \
    --with-masks'

# imx8mp fast (proto-resolution masks)
ssh imx8mp-frdm 'cd /root/ara2-validator && PYTHONPATH=/root/ara2-validator \
    /root/venv/bin/python3 -m ara2_validator.edgefirst \
    --model /root/yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128/images/ \
    --gt /root/coco128/annotations/instances_train2017.json \
    --with-masks --fast-masks'

# imx95 retina
ssh imx95-frdm 'cd /root/ara2-validator && PYTHONPATH=/root/ara2-validator \
    /root/venv/bin/python3 -m ara2_validator.edgefirst \
    --model /root/yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128/images/ \
    --gt /root/coco128/annotations/instances_train2017.json \
    --with-masks'

# imx95 fast
ssh imx95-frdm 'cd /root/ara2-validator && PYTHONPATH=/root/ara2-validator \
    /root/venv/bin/python3 -m ara2_validator.edgefirst \
    --model /root/yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128/images/ \
    --gt /root/coco128/annotations/instances_train2017.json \
    --with-masks --fast-masks'
```

### Running the deployment-style benchmark

The fused `draw_masks` benchmark uses a deployment score threshold (`0.5`) and measures the path a live pipeline takes:

```bash
ssh imx8mp-frdm 'cd /root/ara2-validator && PYTHONPATH=/root/ara2-validator \
    /root/venv/bin/python3 -m tests.bench_draw_masks \
    --model /root/yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128/images/ \
    --score-threshold 0.5'

ssh imx95-frdm 'cd /root/ara2-validator && PYTHONPATH=/root/ara2-validator \
    /root/venv/bin/python3 -m tests.bench_draw_masks \
    --model /root/yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128/images/ \
    --score-threshold 0.5'
```

### Interpreting output

Each run prints a timing table:

```
Stage                               Mean      Min      Max  ms
preprocess (GPU)                    4.11     3.34     4.79
inference (wall)                   11.25    11.03    11.52
  dma input (host→device)             1.96     1.95     2.03
  dma output (device→host)            2.96     2.94     3.01
postprocess (decode+mask)          10.12     3.66    24.60
  decode (HAL)                        2.56     2.19     2.98
  mask materialize                    7.55     1.33    21.97
  Model-path (pre+inf+post, excl. output): 25.47 ms (39.3 FPS)
```

Key metrics to compare:
- **decode** — detection decode (dequantisation + NMS + proto extraction) in compiled Rust
- **mask materialize** — mask materialisation (retina ~7–10 ms, fast ~2–3 ms)
- **Model-path** — end-to-end excluding output formatting (the number reported in the timing table above)

Accuracy is reported below the timing table as Box mAP@0.50:0.95 and Mask mAP@0.50:0.95.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `HardwareError: Ara2 error: 520` | NPU not ready. Run `systemctl restart ara2.service`, wait 20s. If it fails, reboot. |
| `ara2.service` stuck in `activating` | Reboot the board. imx95 may need 3–5 reboots. |
| Inference wall time >20 ms on imx95 | PCIe/thermal issue. Reboot for clean DMA state. |
| `ModuleNotFoundError: edgefirst_ara2` | On imx95: venv must use `--system-site-packages` (edgefirst-ara2 is system-installed). |
| Wrong .so loaded (stale results) | Remove system-wide copy: `rm -rf /usr/lib/python3.13/site-packages/edgefirst_hal*` |
| `pip install` of validator fails with flat-layout error | Use the rsync + PYTHONPATH workflow above instead. |

## Architecture

```mermaid
flowchart TD
    classDef cpu fill:#e8e8e8,stroke:#888,color:#333
    classDef gpu fill:#cfe9ff,stroke:#4a8fbf,color:#1a3a55
    classDef npu fill:#f5d0c5,stroke:#c97155,color:#5a2a18
    classDef host fill:#d5e8d4,stroke:#82b366,color:#2a4a2a

    subgraph host_path[ONNX reference]
        H1[cv2 letterbox<br/>FP32 / 255]:::host
        H2[ONNX Runtime<br/>CPU/CUDA]:::host
        H3[numpy NMS<br/>retina mask decode]:::host
    end

    subgraph dvapi_path[dvapi reference — on-target]
        D1[cv2 letterbox<br/>+ int8 quantize]:::cpu
        D2[dvapi.py → libaraclient<br/>via /var/run/ara2.sock]:::npu
        D3[numpy NMS + dequant<br/>retina mask decode]:::cpu
    end

    subgraph hal_path[EdgeFirst HAL — on-target]
        E1[HAL GPU letterbox<br/>writes NPU input DMA-BUF]:::gpu
        E2[edgefirst-ara2 → libaraclient<br/>via /var/run/ara2.sock]:::npu
        E3[HAL Decoder + dequant<br/>HAL materialize_masks]:::gpu
    end

    H1 --> H2 --> H3 --> COCO[COCO JSON<br/>bbox + segm]:::host
    D1 --> D2 --> D3 --> COCO
    E1 --> E2 --> E3 --> COCO
```

The `edgefirst` path differs from the `reference` path in three places:

- **Preprocess** runs on the GPU and writes its result directly into the DMA-BUF surface that dvproxy will hand to the NPU as its input tensor — no host-side copy between letterbox output and NPU input.
- **Decode (dequantisation + NMS)** runs in compiled Rust against the dvproxy output DMA-BUF surface, so the HAL Decoder reads the NPU output without staging through host memory.
- **Mask materialisation** runs in HAL's batched-GEMM kernel (rayon-parallel) instead of a per-detection numpy `cv2.resize`.

### Capability matrix

| | `onnx_reference` | `reference` | `edgefirst` |
|---|---|---|---|
| Device | PC / any CPU, or on-target | imx8mp-frdm, imx95-frdm | imx8mp-frdm, imx95-frdm |
| Runtime | ONNX Runtime | dvapi → libaraclient | edgefirst-ara2 → libaraclient |
| Preprocess | cv2 (CPU) | cv2 + int8 quant (CPU) | HAL GPU (DMA-BUF) |
| Decode | numpy (CPU) | numpy + dequant (CPU) | HAL Decoder (Rust) |
| Mask decode | numpy retina or process_mask | numpy retina or process_mask | HAL `materialize_masks` (Scaled or Proto) |
| Box eval | ✅ bbox | ✅ bbox | ✅ bbox |
| Mask eval | ✅ segm | ✅ segm | ✅ segm |
| NPU sub-timings | — | ✅ DMA + NPU | ✅ DMA + NPU |

### Model format

The ONNX reference uses the standard Ultralytics export with a single concatenated output `[1, 116, 8400]` (4 box + 80 class + 32 mask coeff) plus prototype tensor `[1, 32, 160, 160]`.

The Ara240 DVM (`yolov8n-seg-kinara-1.2.1.dvm`) uses a **split-decoder** output format compiled by the NXP toolchain:

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `protos` | [1, 32, 160, 160] | int8 | Mask prototype spatial basis |
| `mask_coefs` | [1, 32, 8400] | int8 | Mask prototype weights |
| `scores` | [1, 80, 8400] | uint8 | Per-class confidence |
| `boxes` | [1, 4, 8400] | int16 | Box coordinates (xcycwh) |

Other DVMs (e.g. from EdgeFirst Studio sessions) may split boxes into separate xy `[1, 2, 8400]` and wh `[1, 2, 8400]` tensors with independent quantisation parameters. Both formats are handled automatically.

## Mask materialisation modes

Each script defaults to the **retina** mask path — Ultralytics `ops.process_mask_native()`: bilinear-upsample the continuous f32 logit plane to original-image resolution, then threshold at `logit > 0`. This is the right mode for COCO evaluation and for visualisation, and is what `MaskResolution.Scaled(orig_w, orig_h)` selects in the HAL backend.

The `--fast-masks` flag is an opt-in to a lower-latency path:

- **`edgefirst`** — `MaskResolution.Proto` returns continuous-sigmoid u8 tiles at proto resolution and the caller upsamples each tile via `cv2.resize`. About 12% faster end-to-end on imx8mp-frdm, ~1.2 pp lower mask mAP.
- **`reference` and `onnx_reference`** — Ultralytics `ops.process_mask()`: threshold logits at proto resolution, bilinear-upsample the binary mask, re-threshold. Same speed as retina on these stacks (per-detection upsample dominates either way), ~0.6 pp lower mask mAP.

## Usage

### `onnx_reference` — FP32 ONNX (host or on-target)

```bash
python -m ara2_validator.onnx_reference \
    --model yolov8n-seg.onnx \
    --images /root/coco128-seg/images/ \
    --gt /root/coco128-seg/annotations/instances_train2017.json \
    --with-masks \
    --provider cpu
```

### `reference` — Ara240 dvapi (imx8mp-frdm, imx95-frdm)

```bash
python -m ara2_validator.reference \
    --model yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128-seg/images/ \
    --gt /root/coco128-seg/annotations/instances_train2017.json \
    --with-masks
```

### `edgefirst` — EdgeFirst HAL (imx8mp-frdm, imx95-frdm)

```bash
python -m ara2_validator.edgefirst \
    --model yolov8n-seg-kinara-1.2.1.dvm \
    --images /root/coco128-seg/images/ \
    --gt /root/coco128-seg/annotations/instances_train2017.json \
    --with-masks
```

### Common CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Model file (.onnx or .dvm) |
| `--images` | (required) | Image directory |
| `--output` | per-script | COCO results JSON output path |
| `--score-threshold` | 0.001 | NMS score threshold (Ultralytics `val` default) |
| `--iou-threshold` | 0.7 | NMS IoU threshold (Ultralytics `val` default) |
| `--max-detections` | 300 | Max detections per image (Ultralytics `max_det`) |
| `--max-images` | all | Limit number of images processed |
| `--warmup` | 3 | Warmup iterations (excluded from timing) |
| `--gt` | none | COCO GT JSON for evaluation |
| `--log-level` | INFO | One of DEBUG, INFO, WARNING, ERROR |

### Script-specific options

| Flag | Script | Description |
|------|--------|-------------|
| `--with-masks` | all | Enable segmentation mask decode + eval |
| `--fast-masks` | all | Use the lower-accuracy / lower-latency mask decode path |
| `--provider cpu\|cuda` | `onnx_reference` | ONNX Runtime execution provider |
| `--labels` | `reference`, `edgefirst`, `__main__` | Path to labels.txt |
| `--socket` | `reference`, `edgefirst`, `__main__` | dvproxy socket (default: `/var/run/ara2.sock`) |
| `--backend numpy\|hal` | `__main__` | Backend selector for the side-by-side A/B harness |

`python -m ara2_validator` is a side-by-side A/B harness that wraps both backends behind one entry point; `edgefirst.py` is the canonical HAL entry point.

## Benchmark methodology

**Image decode is reported separately from end-to-end.** The decode stage (JPEG → RGB buffer) simulates loading from disk and is not representative of a live pipeline where frames arrive from a camera or video decoder. The end-to-end timing (preprocess + inference + postprocess) represents the per-frame cost after image acquisition.

**Postprocess includes the mask decode.** For both validators "postprocess" is `nms_decode + mask_decode` (or `nms_decode + materialize_hal` on the HAL path). RLE-encoding the binary masks for the COCO output JSON is reported separately as **output formatting** because it is identical work in every script and is not part of the inference pipeline.

**Validation threshold inflates postprocess.** `--score-threshold 0.001` matches Ultralytics `val` defaults and produces a complete precision-recall curve; a deployment would use `~0.50` and see substantially fewer surviving detections per frame. See [Deployment-style timing](#deployment-style-timing).

**Serial pipeline.** Each stage runs serially to capture accurate per-stage timing. An optimised deployment can overlap preprocessing of frame *N+1* with inference of frame *N*, increasing throughput (gated by the slowest stage) without reducing per-frame latency. The validator does not implement this.

**Outlier filter.** All timing values are reported as min/mean/max after filtering the top and bottom 1% outliers (one sample on each side of a 128-image run). On-target benchmarks see occasional spikes from cgroup CPU pressure, dvproxy stalls, and GPU driver warmup — trimming suppresses those without losing the mean.

**Sequential, never parallel.** Benchmarks against the same target are run sequentially; running two validator processes in parallel on the same host invalidates the per-stage timings via shared GPU/CPU contention.

## Known limitations

### dvapi.py heap corruption past ~100 inferences/process

The `dvapi.py` ctypes wrapper around `libaraclient_aarch64.so` hits a glibc `double free or corruption` abort after roughly 100–200 sequential inferences in a single process. coco128 (128 images) generally completes; coco-val5k (5000 images) does not. The Rust-backed `edgefirst-ara2` wrapper used by `edgefirst.py` is unaffected and is the recommended route for full val5k runs.

### Mask edges are not bit-identical to Ultralytics

int8 quantisation produces protos that are near-identical but not bit-identical to the FP32 reference, so per-mask IoU is in the 0.95–0.99 range rather than 1.0. This is reflected in the −0.45 pp mask mAP drift figure on val5k.

## Dependencies

### All scripts

- `numpy`, `opencv-python`, `Pillow` — preprocessing and image I/O
- `pycocotools` — COCO evaluation and RLE mask encoding

### `onnx_reference`

- `onnxruntime` — FP32 inference (`pip install onnxruntime`)

### `reference`

- `libaraclient.so` — NXP Ara240 client library (part of the Ara240 runtime SDK)
- `dvapi.py` — ctypes wrapper (vendored in this repo, patched for imx8mp)
- `dvproxy` daemon running with `/var/run/ara2.sock` present

### `edgefirst`

- `edgefirst-hal>=0.18.2` — GPU-accelerated preprocessing and postprocessing (the floor is required: 0.17.x ships a scalar `materialize_masks` ~75× slower; 0.18.0 dropped the rayon parallelism the kernel needs; 0.18.1 restored it; 0.18.2 adds detection decode and mask materialisation optimizations)
- `edgefirst-ara2` — Rust/pyo3 wrapper around libaraclient with DMA-BUF tensor mapping
- Same `dvproxy` + `libaraclient.so` requirements as `reference`

## Project structure

```
ara2_validator/
├── __main__.py          # Side-by-side A/B harness (--backend numpy|hal)
├── onnx_reference.py    # FP32 ONNX reference
├── reference.py         # Ara240 dvapi + numpy/OpenCV
├── edgefirst.py         # EdgeFirst HAL — canonical HAL entry point
├── hal_pipeline.py      # HalPipeline class — copy-paste-ready HAL setup
├── postprocess_hal.py   # MaskMode enum + materialize_masks_for_coco_*
├── postprocess_numpy.py # NumPy decode: dequant → boxes → NMS → masks
├── preprocess.py        # Letterbox resize, quantise, HalPreprocessor
├── nms.py               # Pure numpy NMS (class-aware, Ultralytics algorithm)
├── coco_output.py       # COCO JSON serialisation with RLE masks
├── coco_eval.py         # pycocotools evaluation wrapper
├── _stats.py            # Shared trimmed-mean + per-stage formatter
├── model.py             # Ara240 model loading and metadata
├── dvapi.py             # Vendored NXP Ara240 SDK ctypes wrapper
└── tests/
    ├── test_hal_path.py     # Regression guards for the HAL contract
    └── bench_draw_masks.py  # Deployment-style fused draw_masks bench
```

## See also

- **[EdgeFirst Studio](https://edgefirst.studio/)** — multi-model, multi-sensor, multi-platform productisation platform (this repository is a single-model demonstration).
- **[EdgeFirst HAL](https://github.com/EdgeFirstAI/hal)** — hardware abstraction layer (Rust + Python bindings); see the Optimization Guide in the HAL README for the rules `HalPipeline` encodes.
- **[ara2-rs examples](https://github.com/EdgeFirstAI/ara2-rs/tree/main/examples)** — live camera/video inference using `edgefirst-ara2`. In particular `yolov8_live.py` is the canonical pattern for the camera → NPU → fused `draw_masks` → Wayland compositor pipeline that `tests/bench_draw_masks.py` mirrors. This validator is the offline / batched counterpart.
- **EdgeFirst for i.MX Reference Manual** — documents the `edgefirstoverlay` instrumented per-stage timing methodology and platform-by-platform mask-rendering benchmarks that complement the on-device numbers reported here.
