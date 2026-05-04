#!/usr/bin/env python3
"""EdgeFirst HAL-accelerated YOLOv8-seg inference on NXP Ara240.

This is the canonical entry point for the EdgeFirst HAL pipeline:
zero-copy DMA-BUF preprocessing on the GPU, NPU inference via the
Rust-backed ``edgefirst-ara2`` wrapper, and mask materialisation on
HAL's batched-GEMM kernel.

Pipeline overview
-----------------

::

    JPEG  ─┬─►  HAL GPU letterbox  ──►  NPU input DMA-BUF
           │   (DMA-to-DMA, zero-copy)
           │
    NPU output DMA-BUF  ──►  HAL Decoder NMS  ──►  HAL materialize_masks
                            (compiled Rust)         (rayon-parallel)

The HAL setup is encapsulated in :class:`HalPipeline` (see
``hal_pipeline.py``); a customer adopting EdgeFirst HAL can copy that
class verbatim into their own application — it encodes the patterns
the HAL Optimization Guide expects (one ``ImageProcessor`` per
pipeline thread, ``Tensor.from_fd`` once at startup, ``Decoder`` built
once and reused).

Timing breakdown
----------------

The script reports per-stage means after a 1 % trim:

* ``decode``         — JPEG → DMA-backed HAL tensor.
* ``preprocess``     — GPU letterbox-resize (DMA-to-DMA).
* ``inference``      — NPU compute including PCIe DMA transfers
  (sub-timings shown below this row).
* ``postprocess``    — ``nms_decode`` (HAL Decoder) +
  ``materialize_hal`` (HAL ``materialize_masks`` call). This is the
  HAL-side mask work; ``paste`` and ``rle`` are reported separately
  under "output formatting" because they are COCO-output formatting
  (binary canvas paste + ``pycocotools.mask.encode``), not part of
  the inference pipeline.
* ``model-path``     — ``preprocess + inference + postprocess``,
  excluding output formatting. This is what would run after image
  acquisition in a real camera/video pipeline.

For the rationale behind excluding paste + RLE from the model-path,
see the README "Benchmark notes" section.

Mask modes
----------

* Default: retina (:class:`MaskMode.RETINA` — HAL
  ``MaskResolution.Scaled``). Highest accuracy. Matches Ultralytics
  ``ops.process_mask_native()``.
* ``--fast-masks``: fast path (:class:`MaskMode.FAST` — HAL
  ``MaskResolution.Proto`` + caller ``cv2.resize``). About 1.5×
  faster end-to-end on imx8mp-frdm at the cost of about 0.8 pp mask
  mAP. Use for latency-sensitive deployments, not for COCO eval.

Usage
-----

::

    python -m ara2_validator.edgefirst \\
        --model yolov8n-seg-kinara-1.2.1.dvm \\
        --images /root/coco128/images/ \\
        --gt /root/coco128/annotations/instances_train2017.json \\
        --output results_edgefirst.json \\
        --with-masks
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import sys
from pathlib import Path

import numpy as np

from .coco_output import build_coco_results, save_coco_results
from .hal_pipeline import HalPipeline, unletterbox_box_array
from .model import load_model
from .postprocess_hal import MaskMode, materialize_masks_for_coco_rle
from ._stats import fmt_row, summary_header, trimmed_stats

log = logging.getLogger("ara2_validator.edgefirst")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Args shared with reference / onnx_reference."""
    parser.add_argument("--model", required=True,
                        help="Path to .dvm model file")
    parser.add_argument("--images", required=True,
                        help="Path to image directory")
    parser.add_argument("--output", default="results_edgefirst.json",
                        help="Output COCO results JSON path")
    parser.add_argument("--labels", default=None,
                        help="Path to labels.txt (fallback if not in DVM)")
    parser.add_argument("--score-threshold", type=float, default=0.001,
                        help="NMS score threshold (default 0.001 — Ultralytics val convention)")
    parser.add_argument("--iou-threshold", type=float, default=0.7,
                        help="NMS IoU threshold (default 0.7 — Ultralytics val convention)")
    parser.add_argument("--max-detections", type=int, default=300,
                        help="Maximum detections per image (default 300 — Ultralytics max_det)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process")
    parser.add_argument("--socket", default="/var/run/ara2.sock",
                        help="Ara240 dvproxy Unix socket path")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations before timing")
    parser.add_argument("--gt", default=None,
                        help="Path to COCO GT JSON for mAP evaluation")
    parser.add_argument("--with-masks", action="store_true", default=False,
                        help="Enable mask decode and segm evaluation")
    parser.add_argument("--fast-masks", action="store_true", default=False,
                        help="Use the lower-accuracy / lower-latency mask "
                             "path (HAL MaskResolution.Proto + caller "
                             "cv2.resize). About 1.5× faster end-to-end "
                             "on imx8mp-frdm at the cost of ~0.8 pp mask "
                             "mAP. Default (without this flag) is the "
                             "high-accuracy retina path "
                             "(MaskResolution.Scaled, matches Ultralytics "
                             "process_mask_native).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default INFO)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EdgeFirst HAL-accelerated Ara240 YOLOv8-seg inference"
    )
    _add_common_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        stream=sys.stderr,
    )

    # Print HAL version so benchmark logs disambiguate the materialiser
    # path. The batched-GEMM materializer + squeeze_padding_dims
    # schema-v2 fix landed in HAL 0.18.0 (PR #54); the rayon-parallel
    # restoration + per-detection logit-precompute path landed in
    # 0.18.1; further mask decoding optimizations landed in 0.18.2,
    # which is the validator's floor.
    try:
        hal_ver = importlib.metadata.version("edgefirst_hal")
    except Exception:  # noqa: BLE001 — version lookup must not fail the run
        hal_ver = "unknown"

    mode = MaskMode.FAST if args.fast_masks else MaskMode.RETINA
    log.info("HAL:     edgefirst_hal %s", hal_ver)
    log.info("Mask:    %s", "fast (Proto+cv2)" if args.fast_masks else "retina (Scaled)")

    # ── Collect images ───────────────────────────────────────────────
    img_dir = Path(args.images)
    if not img_dir.is_dir():
        log.error("Error: %s is not a directory", args.images)
        return 1
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts
    )
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        log.error("Error: no images found in %s", args.images)
        return 1

    # ── Load model + build HAL pipeline ──────────────────────────────
    log.info("Model:   %s", args.model)
    log.info("Images:  %d from %s", len(image_paths), args.images)
    log.info("Backend: edgefirst (HAL GPU + HAL Decoder)")
    log.info("Output:  %s", args.output)
    model, spec = load_model(
        args.model, labels_path=args.labels, socket_path=args.socket
    )

    # HalPipeline owns all the HAL state (preprocessor, decoder,
    # output tensors). Construction is one-shot per the HAL
    # Optimization Guide; the per-image loop below only calls
    # `infer_one` and the mask helper.
    pipeline = HalPipeline(
        model, spec,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )
    log.info(
        "Input:   %d×%d×%d, qn=%.6f, offset=%d, signed=%s",
        spec.input_shape[0], spec.input_shape[1], spec.input_shape[2],
        spec.input_qn, spec.input_offset, spec.input_signed,
    )
    log.info(
        "Outputs: %s",
        ", ".join(f"{k}={v.shape}" for k, v in spec.outputs.items()),
    )
    log.info("Labels:  %d classes\n", len(spec.labels))

    # ── Warmup ───────────────────────────────────────────────────────
    if args.warmup > 0:
        log.info("Warmup: %d iterations...", args.warmup)
        pipeline.warmup(image_paths, n=args.warmup)

    # ── Process images ───────────────────────────────────────────────
    timings: dict[str, list[float]] = {
        "decode": [], "preprocess": [], "inference": [],
        "nms_decode": [], "materialize_hal": [],
        "paste": [], "rle": [],
        "postprocess": [], "total": [],
        "dma_input": [], "npu_compute": [], "dma_output": [],
    }
    all_results: list[dict] = []
    mask_failures = 0

    import time as _time

    for idx, img_path in enumerate(image_paths):
        t_total = _time.perf_counter()

        # 1-4: preprocess + inference + nms_decode (one HAL pipeline call).
        result = pipeline.infer_one(img_path)

        # 5: mask materialisation (HAL → RLE), measured separately.
        mask_result = None
        if args.with_masks and result.proto_data is not None and len(result.scores) > 0:
            try:
                mask_result = materialize_masks_for_coco_rle(
                    pipeline.preprocessor.processor,
                    result.boxes, result.scores, result.classes,
                    result.proto_data, result.lb_info,
                    pipeline.input_w, pipeline.input_h,
                    mode=mode,
                )
            except Exception as e:  # noqa: BLE001 — degrade per-frame, log loudly
                log.warning(
                    "[HAL] materialize_masks failed on %s: %s: %s",
                    Path(img_path).name, type(e).__name__, e,
                )
                mask_failures += 1

        # Record timings.
        for k, v in result.timings.items():
            timings[k].append(v)
        if mask_result is not None:
            timings["materialize_hal"].append(mask_result.timings.materialize_hal_ms)
            timings["paste"].append(mask_result.timings.paste_ms)
            timings["rle"].append(mask_result.timings.rle_ms)
        else:
            timings["materialize_hal"].append(0.0)
            timings["paste"].append(0.0)
            timings["rle"].append(0.0)
        timings["postprocess"].append(
            timings["nms_decode"][-1] + timings["materialize_hal"][-1]
        )

        # Build COCO results entries (un-letterboxes boxes, attaches RLEs).
        boxes_orig = unletterbox_box_array(
            result.boxes, result.lb_info, pipeline.input_w, pipeline.input_h,
        )
        scores_np = np.asarray(result.scores, dtype=np.float32)
        classes_np = np.asarray(result.classes, dtype=np.uint64)
        rles = mask_result.rles if mask_result is not None else None
        all_results.extend(
            build_coco_results(img_path, boxes_orig, scores_np, classes_np, rles)
        )

        timings["total"].append((_time.perf_counter() - t_total) * 1e3)

        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(image_paths) - 1:
            avg = np.mean(timings["total"][-10:])
            log.info(
                "  [%4d/%d] %20s  %3d dets  %6.1f ms/img",
                idx + 1, len(image_paths), Path(img_path).name,
                len(result.scores), avg,
            )

    if mask_failures:
        log.warning(
            "[HAL] %d/%d frames had mask materialisation failures (see warnings above)",
            mask_failures, len(image_paths),
        )

    # ── Timing summary ───────────────────────────────────────────────
    print()
    hdr = summary_header()
    print(hdr)
    print("─" * len(hdr))
    print(fmt_row("image decode", trimmed_stats(timings["decode"])))
    print(fmt_row("preprocess (GPU)", trimmed_stats(timings["preprocess"])))
    print(fmt_row("inference (wall)", trimmed_stats(timings["inference"])))
    print(fmt_row("dma input (host→device)", trimmed_stats(timings["dma_input"]), indent=1))
    print(fmt_row("npu compute", trimmed_stats(timings["npu_compute"]), indent=1))
    print(fmt_row("dma output (device→host)", trimmed_stats(timings["dma_output"]), indent=1))
    print(fmt_row("postprocess (nms+mask)", trimmed_stats(timings["postprocess"])))
    print(fmt_row("nms_decode (HAL)", trimmed_stats(timings["nms_decode"]), indent=1))
    print(fmt_row("materialize_hal", trimmed_stats(timings["materialize_hal"]), indent=1))
    if any(t > 0 for t in timings["paste"]):
        # Output formatting — NOT part of postprocess / model-path.
        # Reported separately so the inference cost is comparable
        # apples-to-apples with the dvapi reference.
        of_total = [timings["paste"][i] + timings["rle"][i]
                    for i in range(len(timings["paste"]))]
        print(fmt_row("output formatting", trimmed_stats(of_total)))
        print(fmt_row("paste", trimmed_stats(timings["paste"]), indent=1))
        print(fmt_row("rle encode", trimmed_stats(timings["rle"]), indent=1))
    print("─" * len(hdr))
    s = trimmed_stats(timings["total"])
    print(fmt_row("TOTAL (incl. output)", s))
    if s["mean"] > 0:
        print(f"  Throughput: {1000 / s['mean']:.1f} FPS")

    model_path = [
        timings["preprocess"][i] + timings["inference"][i]
        + timings["postprocess"][i]
        for i in range(len(timings["total"]))
    ]
    mp = trimmed_stats(model_path)
    print(
        f"\n  Model-path (pre+inf+post, excl. output): {mp['mean']:.2f} ms "
        f"({1000/mp['mean']:.1f} FPS)"
    )
    print(
        f"  Outlier filter: 1% trimmed ({s['n']}/{len(timings['total'])} samples)"
    )
    print(f"  Total detections: {len(all_results)}")

    save_coco_results(all_results, args.output)
    log.info("  Saved to: %s", args.output)

    # ── COCO evaluation ──────────────────────────────────────────────
    if args.gt:
        from .coco_eval import evaluate_coco
        print()
        metrics = evaluate_coco(args.gt, args.output, iou_type="bbox")
        if metrics:
            print(f"\n  Box  mAP@0.50:0.95 = {metrics['AP']:.4f}")
            print(f"  Box  mAP@0.50      = {metrics['AP50']:.4f}")
            print(f"  Box  mAR@100       = {metrics['AR100']:.4f}")
        if args.with_masks and any("segmentation" in r for r in all_results):
            segm = evaluate_coco(args.gt, args.output, iou_type="segm")
            if segm:
                print(f"\n  Mask mAP@0.50:0.95 = {segm['AP']:.4f}")
                print(f"  Mask mAP@0.50      = {segm['AP50']:.4f}")
                print(f"  Mask mAR@100       = {segm['AR100']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
