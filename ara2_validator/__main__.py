"""CLI entry point for the Ara240 standalone validator.

Runs YOLOv8-seg inference on a set of images via the Ara240 NPU and
produces COCO-format annotation JSON suitable for offline evaluation
with pycocotools.

Two modes:
  --backend numpy  : CPU preprocess + CPU postprocess (baseline)
  --backend hal    : EdgeFirst HAL — GPU letterbox-resize via DMA-BUF
                     zero-copy, NPU inference, and HAL Decoder NMS +
                     ``materialize_masks`` (rayon-parallel, batched-GEMM).

The HAL path delegates all setup to :class:`HalPipeline` (defined in
``hal_pipeline.py``); see :mod:`ara2_validator.edgefirst` for the
canonical HAL entry point. This module exists for the side-by-side
A/B comparison: a single command that can switch between the numpy
reference and the HAL path with ``--backend``.

Usage
-----

::

    python -m ara2_validator \\
        --model yolov8n-seg-kinara-1.2.1.dvm \\
        --images /root/coco128/images/ \\
        --output results.json \\
        --backend hal
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ._stats import fmt_row, summary_header, trimmed_stats
from .coco_output import build_coco_results, save_coco_results

log = logging.getLogger("ara2_validator")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ara240 DVM inference → COCO JSON annotations",
    )
    parser.add_argument("--model", required=True,
                        help="Path to .dvm model file")
    parser.add_argument("--images", required=True,
                        help="Path to image directory")
    parser.add_argument("--output", default="results.json",
                        help="Output COCO results JSON path")
    parser.add_argument("--labels", default=None,
                        help="Path to labels.txt (fallback if not in DVM)")
    parser.add_argument("--backend", choices=["numpy", "hal", "hailo"],
                        default="hal",
                        help="Preprocessing + postprocessing backend "
                             "(default: hal). 'hailo' expects --model to "
                             "point at a .hef file and uses the vendored "
                             "TAPPAS reference postprocess for head-to-head "
                             "comparison against HAL.")
    parser.add_argument("--score-threshold", type=float, default=0.001,
                        help="NMS score threshold (default 0.001 — "
                             "Ultralytics val convention)")
    parser.add_argument("--iou-threshold", type=float, default=0.7,
                        help="NMS IoU threshold (default 0.7 — "
                             "Ultralytics val convention)")
    parser.add_argument("--max-detections", type=int, default=300,
                        help="Maximum detections per image "
                             "(default 300 — Ultralytics max_det)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process")
    parser.add_argument("--socket", default="/var/run/ara2.sock",
                        help="Ara240 dvproxy Unix socket path")
    parser.add_argument("--no-masks", action="store_true",
                        help="Skip mask decoding (detection only)")
    parser.add_argument("--fast-masks", action="store_true",
                        help="HAL backend: use the lower-accuracy / "
                             "lower-latency mask path "
                             "(MaskResolution.Proto + cv2.resize). "
                             "Numpy backend: use Ultralytics "
                             "process_mask (non-retina). Default is the "
                             "high-accuracy retina path.")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations before timing")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default INFO)")
    return parser


def _collect_images(images_dir: str, max_images: int | None) -> list[str]:
    img_dir = Path(images_dir)
    if not img_dir.is_dir():
        log.error("Error: %s is not a directory", images_dir)
        sys.exit(1)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts
    )
    if max_images:
        image_paths = image_paths[:max_images]
    if not image_paths:
        log.error("Error: no images found in %s", images_dir)
        sys.exit(1)
    return image_paths


def _hal_version() -> str:
    try:
        return importlib.metadata.version("edgefirst_hal")
    except Exception:  # noqa: BLE001 — version lookup must not fail the run
        return "unknown"


def _run_hal(args, image_paths: list[str], spec, model) -> list[dict]:
    """HAL backend: HalPipeline + materialize_masks_for_coco_rle."""
    from .hal_pipeline import HalPipeline, unletterbox_box_array
    from .postprocess_hal import MaskMode, materialize_masks_for_coco_rle

    log.info("HAL:     edgefirst_hal %s", _hal_version())
    log.info(
        "Mask:    %s",
        "fast (Proto+cv2)" if args.fast_masks else "retina (Scaled)",
    )

    # HalPipeline owns ImageProcessor + Decoder + DMA-BUF output tensors
    # for the lifetime of the run — see hal_pipeline.py for the full
    # rationale (HAL Optimization Guide rules 1, 4, 5).
    pipeline = HalPipeline(
        model, spec,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )
    mode = MaskMode.FAST if args.fast_masks else MaskMode.RETINA

    if args.warmup > 0:
        log.info("Warmup: %d iterations...", args.warmup)
        pipeline.warmup(image_paths, n=args.warmup)

    timings: dict[str, list[float]] = {
        "decode": [], "preprocess": [], "inference": [],
        "dma_input": [], "npu_compute": [], "dma_output": [],
        "nms_decode": [], "materialize_hal": [],
        "paste": [], "rle": [],
        "postprocess": [], "total": [],
    }
    all_results: list[dict] = []
    mask_failures = 0

    for idx, img_path in enumerate(image_paths):
        t_total = time.perf_counter()

        result = pipeline.infer_one(img_path)

        mask_result = None
        if (not args.no_masks
                and result.proto_data is not None
                and len(result.scores) > 0):
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

        boxes_orig = unletterbox_box_array(
            result.boxes, result.lb_info,
            pipeline.input_w, pipeline.input_h,
        )
        scores_np = np.asarray(result.scores, dtype=np.float32)
        classes_np = np.asarray(result.classes, dtype=np.uint64)
        rles = mask_result.rles if mask_result is not None else None
        all_results.extend(
            build_coco_results(img_path, boxes_orig, scores_np, classes_np, rles)
        )

        timings["total"].append((time.perf_counter() - t_total) * 1e3)

        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(image_paths) - 1:
            avg = np.mean(timings["total"][-10:])
            log.info(
                "  [%4d/%d] %20s  %3d dets  %6.1f ms/img",
                idx + 1, len(image_paths), Path(img_path).name,
                len(result.scores), avg,
            )

    if mask_failures:
        log.warning(
            "[HAL] %d/%d frames had mask materialisation failures",
            mask_failures, len(image_paths),
        )

    _print_summary_hal(timings)
    return all_results


def _run_numpy(args, image_paths: list[str], spec, model) -> list[dict]:
    """OpenCV backend: CPU preprocess + opencv postprocess (baseline)."""
    from .model import _strip_trailing_ones
    from .postprocess_opencv import (
        decode_masks, decode_masks_retina,
        postprocess_opencv, unletterbox_boxes, unletterbox_masks,
    )
    from .preprocess import LetterboxInfo, letterbox

    c, input_h, input_w = spec.input_shape

    if args.warmup > 0:
        log.info("Warmup: %d iterations...", args.warmup)
        for i in range(min(args.warmup, len(image_paths))):
            bgr = cv2.imread(image_paths[i])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            padded, _, _, _ = letterbox(rgb, input_h, input_w)
            normalized = padded.astype(np.float32) / 255.0
            quantized = np.round(normalized / spec.input_qn + spec.input_offset)
            dt = np.int8 if spec.input_signed else np.uint8
            ii = np.iinfo(dt)
            quantized = np.clip(quantized, ii.min, ii.max).astype(dt)
            tensor = np.transpose(quantized, (2, 0, 1))[np.newaxis]
            model.set_input_tensor(0, tensor.flatten())
            model.run()

    timings: dict[str, list[float]] = {
        "decode": [], "preprocess": [], "inference": [],
        "postprocess": [], "total": [],
        "dma_input": [], "npu_compute": [], "dma_output": [],
    }
    all_results: list[dict] = []

    for idx, img_path in enumerate(image_paths):
        t_total = time.perf_counter()

        # 1. Image decode
        t0 = time.perf_counter()
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]
        timings["decode"].append((time.perf_counter() - t0) * 1e3)

        # 2. CPU preprocess (letterbox + normalize + quantize + CHW)
        t0 = time.perf_counter()
        padded, scale_f, pad_x, pad_y = letterbox(rgb, input_h, input_w)
        normalized = padded.astype(np.float32) / 255.0
        quantized = np.round(normalized / spec.input_qn + spec.input_offset)
        dtype = np.int8 if spec.input_signed else np.uint8
        iinfo = np.iinfo(dtype)
        quantized = np.clip(quantized, iinfo.min, iinfo.max).astype(dtype)
        tensor = np.transpose(quantized, (2, 0, 1))[np.newaxis]
        lb_info = LetterboxInfo(
            scale=scale_f, pad_x=pad_x, pad_y=pad_y,
            orig_w=orig_w, orig_h=orig_h,
        )
        timings["preprocess"].append((time.perf_counter() - t0) * 1e3)

        # 3. NPU inference
        t0 = time.perf_counter()
        model.set_input_tensor(0, tensor.flatten())
        timing = model.run()
        timings["inference"].append((time.perf_counter() - t0) * 1e3)
        timings["dma_input"].append(timing.input_time_us / 1000.0)
        timings["npu_compute"].append(timing.run_time_us / 1000.0)
        timings["dma_output"].append(timing.output_time_us / 1000.0)

        # Fetch outputs and dequantize
        outputs = {}
        for role, ts in spec.outputs.items():
            t = model.get_output_tensor(ts.index).reshape(ts.shape)
            stripped = _strip_trailing_ones(list(t.shape))
            if stripped != list(t.shape):
                t = t.reshape(stripped)
            outputs[role] = (t.astype(np.float32) - ts.offset) * ts.qn
        if "boxes_xy" in outputs and "boxes_wh" in outputs:
            xy = outputs.pop("boxes_xy")
            wh = outputs.pop("boxes_wh")
            if xy.ndim == 2:
                xy = xy[np.newaxis]
            if wh.ndim == 2:
                wh = wh[np.newaxis]
            outputs["boxes"] = np.concatenate([xy, wh], axis=1)

        # 4. CPU postprocess (decode + NMS + masks)
        t0 = time.perf_counter()
        boxes_np, scores_np, classes_np, mask_logits = postprocess_opencv(
            outputs, input_h, input_w,
            args.score_threshold, args.iou_threshold,
            args.max_detections, with_masks=not args.no_masks,
        )
        boxes_orig = unletterbox_boxes(boxes_np, lb_info)
        orig_masks = None
        if mask_logits is not None and len(mask_logits) > 0:
            if args.fast_masks:
                binary = decode_masks(mask_logits, boxes_np, input_h, input_w)
                orig_masks = unletterbox_masks(
                    binary, lb_info, input_h, input_w,
                )
            else:
                orig_masks = decode_masks_retina(
                    mask_logits, boxes_orig, lb_info, input_h, input_w,
                )
        timings["postprocess"].append((time.perf_counter() - t0) * 1e3)

        all_results.extend(
            build_coco_results(img_path, boxes_orig, scores_np, classes_np, orig_masks)
        )
        timings["total"].append((time.perf_counter() - t_total) * 1e3)

        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(image_paths) - 1:
            avg = np.mean(timings["total"][-10:])
            log.info(
                "  [%4d/%d] %20s  %3d dets  %6.1f ms/img",
                idx + 1, len(image_paths), Path(img_path).name,
                len(boxes_orig), avg,
            )

    _print_summary_numpy(timings)
    return all_results


def _run_hailo(args, image_paths: list[str]) -> list[dict]:
    """Hailo backend: vendored TAPPAS reference postprocess.

    Drives a Hailo HEF directly via :class:`HailoTappasPipeline`, which
    wraps the C++ shim in ``third_party/hailo_tappas_baseline``. Unlike
    HAL or numpy, this path doesn't go through the Ara240 dvproxy stack
    — it operates on a ``.hef`` file passed via ``--model``.
    """
    from .hailo_pipeline import (
        HailoTappasPipeline,
        boxes_normalized_to_pixels,
        threshold_masks_to_uint8,
    )

    log.info("HEF:     %s", args.model)

    pipeline = HailoTappasPipeline(
        args.model,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )
    log.info(
        "Model input: %d×%d×%d",
        pipeline.input_h, pipeline.input_w, pipeline.backend.model_channels,
    )
    log.info(
        "Outputs: %s",
        ", ".join(pipeline.backend.output_names),
    )

    if args.warmup > 0:
        log.info("Warmup: %d iterations...", args.warmup)
        pipeline.warmup(image_paths, n=args.warmup)

    timings: dict[str, list[float]] = {
        "decode": [], "preprocess": [], "inference": [],
        "postprocess": [], "total": [],
    }
    all_results: list[dict] = []

    for idx, img_path in enumerate(image_paths):
        t_total = time.perf_counter()

        result = pipeline.infer_one(img_path)

        for k in ("decode", "preprocess", "inference", "postprocess"):
            timings[k].append(result.timings[k])

        # Read original-image dims for normalized → pixel box conversion.
        # cv2.imread is the same call the pipeline made internally; on a
        # warm filesystem the second read is essentially free, but we
        # could carry orig_w/orig_h through the pipeline if it shows up.
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        orig_h, orig_w = bgr.shape[:2]

        boxes_px = boxes_normalized_to_pixels(result.boxes, orig_w, orig_h)
        masks = (
            None
            if args.no_masks or len(result.masks_float) == 0
            else threshold_masks_to_uint8(result.masks_float)
        )
        all_results.extend(
            build_coco_results(
                img_path, boxes_px, result.scores, result.classes, masks,
            )
        )

        timings["total"].append((time.perf_counter() - t_total) * 1e3)

        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(image_paths) - 1:
            avg = float(np.mean(timings["total"][-10:]))
            log.info(
                "  [%4d/%d] %20s  %3d dets  %6.1f ms/img",
                idx + 1, len(image_paths), Path(img_path).name,
                len(result.scores), avg,
            )

    _print_summary_hailo(timings)
    return all_results


def _print_summary_hal(timings: dict[str, list[float]]) -> None:
    print()
    hdr = summary_header()
    print(hdr)
    print("─" * len(hdr))
    print(fmt_row("image decode", trimmed_stats(timings["decode"])))
    print(fmt_row("preprocess (GPU)", trimmed_stats(timings["preprocess"])))
    print(fmt_row("inference (wall)", trimmed_stats(timings["inference"])))
    print(fmt_row("dma input (host→device)",
                  trimmed_stats(timings["dma_input"]), indent=1))
    print(fmt_row("npu compute",
                  trimmed_stats(timings["npu_compute"]), indent=1))
    print(fmt_row("dma output (device→host)",
                  trimmed_stats(timings["dma_output"]), indent=1))
    print(fmt_row("postprocess (nms+mask)",
                  trimmed_stats(timings["postprocess"])))
    print(fmt_row("nms_decode (HAL)",
                  trimmed_stats(timings["nms_decode"]), indent=1))
    print(fmt_row("materialize_hal",
                  trimmed_stats(timings["materialize_hal"]), indent=1))
    if any(t > 0 for t in timings["paste"]):
        of_total = [
            timings["paste"][i] + timings["rle"][i]
            for i in range(len(timings["paste"]))
        ]
        print(fmt_row("output formatting", trimmed_stats(of_total)))
        print(fmt_row("paste", trimmed_stats(timings["paste"]), indent=1))
        print(fmt_row("rle encode", trimmed_stats(timings["rle"]), indent=1))
    _print_total(timings, hdr_len=len(hdr))


def _print_summary_hailo(timings: dict[str, list[float]]) -> None:
    """Hailo backend timing summary.

    No DMA breakdown: HailoRT does report HW-only latency separately,
    but we deliberately measure full-frame host wall-clock to keep the
    comparison against HAL apples-to-apples. The decode/preprocess/
    inference/postprocess split mirrors HAL's structure so the rows
    line up visually when running both backends back to back.
    """
    print()
    hdr = summary_header()
    print(hdr)
    print("─" * len(hdr))
    print(fmt_row("image decode", trimmed_stats(timings["decode"])))
    print(fmt_row("preprocess (OpenCV)",
                  trimmed_stats(timings["preprocess"])))
    print(fmt_row("inference (wall)",
                  trimmed_stats(timings["inference"])))
    print(fmt_row("postprocess (TAPPAS)",
                  trimmed_stats(timings["postprocess"])))
    _print_total(timings, hdr_len=len(hdr))


def _print_summary_numpy(timings: dict[str, list[float]]) -> None:
    print()
    hdr = summary_header()
    print(hdr)
    print("─" * len(hdr))
    print(fmt_row("image decode", trimmed_stats(timings["decode"])))
    print(fmt_row("preprocess", trimmed_stats(timings["preprocess"])))
    print(fmt_row("inference (wall)", trimmed_stats(timings["inference"])))
    print(fmt_row("dma input (host→device)",
                  trimmed_stats(timings["dma_input"]), indent=1))
    print(fmt_row("npu compute",
                  trimmed_stats(timings["npu_compute"]), indent=1))
    print(fmt_row("dma output (device→host)",
                  trimmed_stats(timings["dma_output"]), indent=1))
    print(fmt_row("postprocess", trimmed_stats(timings["postprocess"])))
    _print_total(timings, hdr_len=len(hdr))


def _print_total(timings: dict[str, list[float]], hdr_len: int) -> None:
    print("─" * hdr_len)
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
        f"  Outlier filter: 1% trimmed "
        f"({s['n']}/{len(timings['total'])} samples)"
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        stream=sys.stderr,
    )

    image_paths = _collect_images(args.images, args.max_images)

    log.info("Model:   %s", args.model)
    log.info("Images:  %d from %s", len(image_paths), args.images)
    log.info("Backend: %s", args.backend)
    log.info("Output:  %s", args.output)

    if args.backend == "hailo":
        # Hailo path drives a .hef directly via HailoRT — no dvproxy,
        # no Ara240-specific tensor spec. The pipeline introspects the
        # HEF for input dims and output names itself.
        all_results = _run_hailo(args, image_paths)
    else:
        from .model import load_model
        model, spec = load_model(
            args.model, labels_path=args.labels, socket_path=args.socket,
        )
        c, input_h, input_w = spec.input_shape
        log.info(
            "Input:   %d×%d×%d, qn=%.6f, offset=%d, signed=%s",
            c, input_h, input_w,
            spec.input_qn, spec.input_offset, spec.input_signed,
        )
        log.info(
            "Outputs: %s",
            ", ".join(f"{k}={v.shape}" for k, v in spec.outputs.items()),
        )
        log.info("Labels:  %d classes", len(spec.labels))

        if args.backend == "hal":
            all_results = _run_hal(args, image_paths, spec, model)
        else:
            all_results = _run_numpy(args, image_paths, spec, model)

    log.info("Total detections: %d", len(all_results))
    save_coco_results(all_results, args.output)
    log.info("Saved to: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
