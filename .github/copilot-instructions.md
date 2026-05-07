# Repository guide — `ara2-validator` (the HAL validator)

This repo is a **rudimentary validation harness** for Ultralytics YOLO segmentation
models, designed to compare **vendor-/upstream-faithful reference pipelines** against
**EdgeFirst HAL-optimized pipelines** on the same model and the same hardware.

It exists to answer one question per platform: *"if you adopt EdgeFirst HAL on this
target, what changes — in measured per-stage latency, and in pycocotools mAP?"*
Every design decision in this repo is in service of producing a clean,
defensible answer to that question.

The historical name `ara2-validator` is a fossil from when the only target was the
NXP Ara240; we now also benchmark against Hailo-8/8L on RPi5, and the architecture
is meant to scale to additional NPUs as backends are added. Treat the repo's
conceptual role as **"the HAL validator"** even though the import package is still
`ara2_validator`.

## The one rule everything else depends on

**Pipelines run serially. One stage at a time, one frame at a time.**

This is not a limitation we will eventually fix. It is the load-bearing invariant
of the whole project.

Every modern inference framework — HailoRT, TAPPAS GStreamer, ONNX Runtime,
EdgeFirst HAL itself — supports concurrent pipelining: starting frame *N+1*'s
preprocess while frame *N*'s NPU inference runs, etc. Concurrent pipelining
**increases throughput** (frames per second) by hiding stages behind the slowest
one. It **does not reduce per-frame latency**, and in many cases it *increases*
end-to-end latency by a few milliseconds (extra queueing, scheduler jitter,
deeper buffer chains).

If you measure a pipelined system, your "preprocess time" silently includes time
the GPU spent waiting for an inference batch to drain, your "inference time"
silently includes time stalled on the previous frame's postprocess copy, and
the per-stage attribution becomes meaningless. The whole point of this repo —
*"can the user trust the per-stage breakdown?"* — collapses.

So: **never** add async/await, threading, double-buffering, batch inference,
or pipelined dispatch to any benchmark path here. If a contributor proposes it,
push back. If a vendor's reference pipeline naturally pipelines (TAPPAS GStreamer
does), wrap it in a synchronous shim that runs one frame to completion before
returning, and document the wrapping. The numbers in the README are what users
believe; making them wrong is the worst thing this repo can do.

## Three concepts, three terms — use them consistently

The README, the docstrings, and these instructions all use the same three-tier
language. **Don't invent synonyms.** Every PR description and every commit
message should pick from this vocabulary verbatim:

- **Reference model** — the Ultralytics ONNX FP32 export. Treated as
  effectively lossless from the original PyTorch checkpoint. Defines the
  accuracy ceiling; absolute mAP values from this row are what every NPU
  number is reported relative to. Run on a host CPU/CUDA box, not on-target.
  **Per-stage timing for the reference model is not reported** — the question
  this repo asks is on-target latency, and host CPU latency would only confuse
  the comparison. The `onnx_reference` script produces this row.
- **Reference pipeline** — the NPU run via the vendor's documented
  integration. The shape is **NPU + vendor SDK (e.g. TAPPAS for Hailo, dvapi
  for Ara240) + OpenCV/NumPy** for whatever the SDK doesn't cover (preprocess
  on every target; postprocess on Ara240, where the SDK ships no postprocess
  library). One reference pipeline per target — the customer-facing "what you
  get out of the box" path. Backends: `numpy` (Ara240 dvapi + opencv), `hailo`
  (Hailo TAPPAS C++ postprocess + opencv preprocess).
- **EdgeFirst pipeline** — the *same NPU model* run through `edgefirst-hal` for
  preprocessing, detection decode, NMS, and mask materialisation. Same NPU
  bytes, same compiled graph, same int8 quantisation; only the host-side
  software around the NPU changes. Backends: `hal` (Ara240), `hal-hailo`
  (Hailo).

| Backend        | Concept            | Pre        | Inference         | Post                |
|----------------|--------------------|------------|-------------------|---------------------|
| `onnx_reference` | reference model    | OpenCV CPU | ONNX Runtime      | Ultralytics         |
| `numpy`        | reference pipeline | OpenCV CPU | dvapi NPU         | NumPy + cv2         |
| `hailo`        | reference pipeline | OpenCV CPU | HailoRT NPU       | TAPPAS C++          |
| `hal`          | EdgeFirst pipeline | HAL GPU    | edgefirst-ara2 NPU| HAL Decoder + masks |
| `hal-hailo`    | EdgeFirst pipeline | HAL GPU    | HailoRT NPU       | HAL Decoder + masks |

All five paths use the same Ultralytics-faithful letterbox geometry, the same
NMS parameters (score 0.001 / IoU 0.7 / max_det 300 for `val`; 0.5 / 0.7 / 300
for deploy), and the same pycocotools eval harness. The two EdgeFirst pipelines
share the **same** HAL Decoder (Rust per-scale subsystem) and **same** mask
materialiser; only the inference engine differs.

### Drift columns are vs the reference model, always

When adding a new backend or platform, the absolute mAP column goes next to a
"Δ pp vs FP32" drift number. The drift is computed against the reference-model
row at the top of the same table — never against another NPU pipeline, never
against another platform. Within-target accuracy comparison is then a quick
read of two drift numbers (reference pipeline vs EdgeFirst pipeline on the same
NPU).

### HAL preprocess uses the GPU, OpenCV uses the CPU

The 1–2 ms wall-clock difference between OpenCV preprocess and HAL preprocess
at 640×640 input is not a HAL slowdown — HAL runs on the platform GPU (V3D on
RPi5, Vivante on i.MX) so the CPU is free for the host-side decode + NMS that
follows. OpenCV runs on the CPU and contends with the same cores the
postprocess will need. At larger camera resolutions (1080p input → 640×640
model) HAL beats OpenCV on wall-clock as well. We report only the canonical
EdgeFirst pipeline (HAL pre + HAL post) and the canonical reference pipeline
(vendor pre + vendor post) — not the cross combinations. A user who wants HAL
post with OpenCV pre, or vice versa, can wire it from
`ara2_validator/preprocess.py` + `ara2_validator/postprocess_hal.py`.

## Per-stage timings vs pycocotools mAP, together

Every benchmark must report **both** the per-stage timing breakdown **and** the
pycocotools mAP. They are not independently meaningful:

- A pipeline that drops postprocess time by skipping NMS at low scores is
  *faster* and *worse*; only paired numbers expose the trade.
- A pipeline that improves mask mAP by switching from `process_mask` to
  `process_mask_native` doubles the mask-decode cost; only paired numbers expose
  the trade.
- A pipeline that gets fast by quantizing more aggressively loses accuracy; only
  paired numbers expose the trade.

If a change is shown in isolation — *"X is now 5 ms faster"* without an mAP
column — it is not a complete benchmark. Add the pycocotools eval before
declaring victory. The validator's main entry point already runs both in one
process; new backends should preserve that pairing.

## Score thresholds: validation vs deployment

Two threshold regimes, both reported, never confused:

- **`--score-threshold 0.001`** (validation) — Ultralytics `val` convention.
  Submits all candidate detections so pycocotools can integrate the full
  precision-recall curve over 101 recall levels. **This is the only threshold
  at which mAP is comparable across runs.** A `0.5` threshold would silently
  truncate the P-R curve and under-report mAP by ~30%.
- **`--score-threshold 0.5`** (deployment) — what a real-time application would
  use. Reports realistic per-frame latency under typical detection counts (1-5
  surviving boxes) and demonstrates how postprocess cost (especially mask
  materialization) collapses when low-confidence candidates are filtered.

When adding a new benchmark row, run **both** thresholds and report each in its
own table. Postprocess and mask costs are dominated by detection count, so a
single threshold gives a misleading picture in either direction.

## Backends and dispatch

`ara2_validator/__main__.py` is the unified harness. The `--backend` flag
selects the pipeline:

```text
--backend numpy       NXP dvapi reference (Ara240, on-target)
--backend hal         HAL on Ara240
--backend hailo       Hailo TAPPAS reference (RPi5 + Hailo)
--backend hal-hailo   HAL on Hailo-8/8L (RPi5 + Hailo)
```

Each backend has its own `_run_<name>` function and `_print_summary_<name>`
helper. New backends should follow the same shape — extract the per-stage
timings, print them under a consistent header, and write the COCO results JSON
with the existing `build_coco_results` + `save_coco_results` helpers.

`ara2_validator/onnx_reference.py` is a separate entry point because it runs on
a host (CPU/CUDA), not on-target.

## Critical files

| Path                                | Role                                                                  |
|-------------------------------------|-----------------------------------------------------------------------|
| `ara2_validator/__main__.py`        | Unified dispatch + per-stage summary tables                           |
| `ara2_validator/hal_pipeline.py`    | HAL on Ara240 — DMA-BUF zero-copy, Ultralytics letterbox parity        |
| `ara2_validator/hal_hailo_pipeline.py` | HAL on Hailo — GPU letterbox + HailoRT inference + HAL decode      |
| `ara2_validator/hailo_pipeline.py`  | TAPPAS reference wrapper around the vendored C++ shim                  |
| `ara2_validator/reference.py`       | NXP dvapi reference (Ara240 vendor path)                              |
| `ara2_validator/onnx_reference.py`  | Ultralytics FP32 host baseline                                        |
| `ara2_validator/postprocess_hal.py` | `MaskMode.RETINA` / `FAST` and `materialize_masks_for_coco_rle`       |
| `ara2_validator/preprocess.py`      | Letterbox geometry + `HalPreprocessor` (matches Ultralytics rounding) |
| `ara2_validator/coco_eval.py`       | pycocotools eval wrapper — bbox + segm                                |
| `third_party/hailo_tappas_baseline/`| Vendored Hailo Application Code Examples (MIT) + pybind11 shim         |

## Things to avoid

- **Don't call vendor "reference" anything that doesn't run an unmodified
  upstream codepath.** The point of `numpy` / `hailo` backends is that they
  can be reproduced with no EdgeFirst dependencies. Reaching into HAL from
  the reference path defeats the comparison.
- **Don't swap thresholds without re-running.** Numbers in the README are
  paired with their threshold; mixing them produces misleading deltas.
- **Don't silently change the letterbox.** Ultralytics uses a specific
  rounding convention (`round(x - 0.1)` for left/top, `round(x + 0.1)` for
  right/bottom). HAL matches this. The reference paths must too.
- **Don't add `async`, `torch.compile()`, batched inference, multi-stream
  HailoRT, or any pipelining flag.** See "the one rule" above.
- **Don't add a "HAL fast path" that diverges from the schema-driven
  Decoder.** HAL's per-scale subsystem reads the embedded `edgefirst.json`
  schema-v2 metadata; bypassing it loses the cross-NPU comparability that
  makes the postprocess column comparable.
- **Don't commit `.claude/plans/*` — those are scratch.** Per the
  user's global guidance, plans live locally and never reach git.

## Common workflow shortcuts

```bash
# Activate venv first (per global CLAUDE.md instruction)
source venv/bin/activate

# Validation run on Hailo (val convention)
python -m ara2_validator \
    --model /path/to/yolov8n-seg.hef --backend hal-hailo \
    --images /path/to/val2017 \
    --output /tmp/results.json \
    --score-threshold 0.001 --warmup 5

# Deployment run on Hailo
python -m ara2_validator \
    --model /path/to/yolov8n-seg.hef --backend hal-hailo \
    --images /path/to/coco128-seg/images \
    --output /tmp/results_deploy.json \
    --score-threshold 0.5 --warmup 3

# Eval (paired bbox + segm)
python -m ara2_validator.coco_eval \
    --gt /path/to/instances_val2017.json \
    --results /tmp/results.json --type bbox
python -m ara2_validator.coco_eval \
    --gt /path/to/instances_val2017.json \
    --results /tmp/results.json --type segm
```

When extending the validator with a new target, follow this checklist:

1. Add a `<target>_pipeline.py` that owns the full per-frame pipeline and
   reports `decode`, `preprocess`, `inference`, `nms_decode`, `materialize_hal`
   (or equivalent) timings — synchronous, single frame in flight.
2. Wire `--backend <name>` in `__main__.py` with both a `_run_<name>` and a
   `_print_summary_<name>`.
3. Run **paired** validation (`0.001`) and deployment (`0.5`) benchmarks.
4. Run pycocotools eval against the same GT, both `bbox` and `segm`.
5. Update the README results table with the new row, threshold-paired.
6. Add a row to the capability matrix.

That's the full pattern. Don't ship a backend without all five.
