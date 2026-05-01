#!/usr/bin/env python3
"""FP32 ONNX reference validation for YOLOv8-seg using ONNX Runtime.

This script establishes the accuracy ceiling for a YOLOv8-seg model by
running FP32 inference on the original ONNX export. It is intended to
run on a PC (or any CPU/CUDA host) and produces COCO-format results
evaluated against ground truth via pycocotools.

The results from this script serve as the reference target: int8 NPU
implementations (Ara240 DVM, TFLite, HEF) should be within a few mAP
points of this FP32 baseline, and the HAL vs numpy decode paths should
match this baseline to within <0.5%.

Pipeline overview
-----------------
The preprocessing and postprocessing are written to match Ultralytics'
ONNX inference path as closely as possible so that any mAP difference
from the published Ultralytics numbers is due to minor numerical
differences (float32 rounding, NMS tie-breaking), not algorithmic
divergence.

  Ultralytics (torch)              This script (numpy)
  ─────────────────────            ───────────────────────
  cv2.imread → BGR                 cv2.imread → BGR→RGB
  LetterBox(center=True)           letterbox() with cv2.copyMakeBorder
  BGR→RGB, HWC→BCHW, /255          HWC→BCHW, /255
  ort.InferenceSession.run()       ort.InferenceSession.run()
  transpose(1,2) → [N,116]         transpose → [N,116]
  xywh2xyxy, split box/cls/mc      xywh2xyxy, split box/cls/mc
  torchvision.ops.nms (C++)        _greedy_nms (pure numpy)
  coeffs @ protos → crop → >0      coeffs @ protos → crop → >0

Key differences from reference.py (Ara240 DVM path):
  - No quantization: input is float32, output is float32
  - No Ara240 dvapi: uses ONNX Runtime CPU provider
  - Standard YOLO output layout: [1, 116, 8400] concat tensor
    (vs split decoder outputs on Ara240)
  - No PCIe/NPU timing breakdown (just wall-clock per stage)

Ultralytics published numbers (COCO val2017, 5000 images):
  YOLOv8n-seg: box mAP@0.50:0.95 = 36.7, mask mAP@0.50:0.95 = 30.5
  https://docs.ultralytics.com/models/yolov8/

Usage
-----
    python -m ara2_validator.onnx_reference \\
        --model yolov8n-seg.onnx \\
        --images coco128/images/ \\
        --gt coco128/annotations/instances_train2017.json \\
        --output results_onnx_fp32.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ._stats import fmt_row, summary_header, trimmed_stats
from .coco_output import build_coco_results, save_coco_results
from .nms import nms_class_aware, nms_with_masks

log = logging.getLogger("ara2_validator.onnx_reference")


# ── Preprocessing ─────────────────────────────────────────────────────────────
#
# Matches Ultralytics augment.py::LetterBox.__call__() exactly:
#   1. scale = min(target/src_h, target/src_w)
#   2. cv2.resize(INTER_LINEAR)
#   3. cv2.copyMakeBorder(value=(114,114,114), centered)
#   4. BGR→RGB (we load as RGB directly)
#   5. HWC→CHW, float32 / 255.0
#
# No quantization — stays FP32 end-to-end, matching Ultralytics ONNX path.

def letterbox(img, target_h, target_w, color=(114, 114, 114)):
    """Letterbox resize matching Ultralytics LetterBox(center=True).

    Returns (padded_img, scale, pad_x, pad_y).
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    right = target_w - new_w - pad_x
    bottom = target_h - new_h - pad_y
    padded = cv2.copyMakeBorder(
        resized, pad_y, bottom, pad_x, right,
        cv2.BORDER_CONSTANT, value=color)
    return padded, scale, pad_x, pad_y


def preprocess(rgb, input_h, input_w):
    """Letterbox → float32 / 255 → HWC→BCHW.

    Matches Ultralytics predictor.preprocess():
        im = im[..., ::-1].transpose((0,3,1,2))  # BGR→RGB, BHWC→BCHW
        im = im.astype(np.float32) / 255.0

    Returns (tensor_bchw, scale, pad_x, pad_y, orig_w, orig_h).
    """
    orig_h, orig_w = rgb.shape[:2]
    padded, scale, pad_x, pad_y = letterbox(rgb, input_h, input_w)

    tensor = padded.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]
    tensor = np.ascontiguousarray(tensor)

    return tensor, scale, pad_x, pad_y, orig_w, orig_h


# ── Postprocessing ────────────────────────────────────────────────────────────
#
# Standard YOLO output: output0 = [1, 116, 8400], output1 = [1, 32, 160, 160]
#
# output0 layout (116 = 4 + 80 + 32):
#   [0:4,  :]  → box coordinates (xc, yc, w, h) in pixel space
#   [4:84, :]  → 80 class scores (NOT softmaxed — raw logits, but already
#                post-DFL so they behave as confidences after the model's
#                internal sigmoid)
#   [84:116,:] → 32 mask prototype coefficients
#
# Ultralytics decode (non_max_suppression in utils/ops.py):
#   1. prediction.transpose(-1, -2)  → [1, 8400, 116]
#   2. xywh2xyxy on [:, :4]
#   3. Confidence filter, class-aware NMS via torchvision.ops.nms
#   4. Separate mask coefficients for post-NMS detections
#
# Mask decode (process_mask in utils/ops.py):
#   1. coeffs @ protos.reshape(32, -1) → [M, 160*160]
#   2. reshape → [M, 160, 160]
#   3. crop_mask to bounding box region
#   4. Threshold > 0 (equivalent to sigmoid > 0.5)

def xywh2xyxy(boxes):
    """Convert [xc, yc, w, h] → [x1, y1, x2, y2].

    Matches ultralytics.utils.ops.xywh2xyxy().
    """
    xy = boxes[:, :2]
    wh = boxes[:, 2:4]
    return np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)


def decode_yolo_output(
    output0: np.ndarray,
    output1: np.ndarray,
    input_h: int,
    input_w: int,
    score_threshold: float,
    iou_threshold: float,
    max_det: int,
    with_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Decode standard YOLO [1,116,8400] + [1,32,160,160] outputs.

    Returns (boxes_xyxy, scores, classes, masks_or_None) in model-input
    pixel coordinates (640×640 letterboxed space).
    """
    pred = output0[0].T  # [1, 116, 8400] → [8400, 116]

    num_classes = pred.shape[1] - 4 - 32
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:4 + num_classes]
    mask_coeffs = pred[:, 4 + num_classes:]

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    if with_masks:
        boxes_nms, scores_nms, classes_nms, mc_nms = nms_with_masks(
            boxes_xyxy, class_scores, mask_coeffs,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_det,
            class_aware=True,
        )
    else:
        boxes_nms, scores_nms, classes_nms, _ = nms_class_aware(
            boxes_xyxy, class_scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_det,
        )
        mc_nms = None

    mask_logits = None
    if mc_nms is not None and len(mc_nms) > 0:
        protos = output1[0]  # [32, 160, 160]
        mask_logits = compute_mask_logits(mc_nms, protos)

    return boxes_nms, scores_nms, classes_nms, mask_logits


def compute_mask_logits(mask_coeffs, protos):
    """Compute raw mask logits at prototype resolution (160×160).

    Returns float logits — thresholding is deferred until after upsampling
    to original image resolution (retina-style, see decode_masks_retina).

    Ultralytics equivalent (process_mask_native, ops.py:520-521):
      masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    """
    c, ph, pw = protos.shape
    logits = mask_coeffs @ protos.reshape(c, -1)
    return logits.reshape(-1, ph, pw)


def unletterbox_boxes(boxes, scale, pad_x, pad_y, orig_w, orig_h):
    """Map xyxy boxes from 640×640 letterboxed space to original image.

    Matches ultralytics.utils.ops.scale_boxes().
    """
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - pad_x) / scale, 0, orig_w)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - pad_y) / scale, 0, orig_h)
    return out


def decode_masks_retina(mask_logits, boxes_orig, scale, pad_x, pad_y,
                       orig_w, orig_h, input_h, input_w):
    """Retina-style mask decode: upsample logits to original size, then crop + threshold.

    Matches ultralytics.utils.ops.process_mask_native() (ops.py:508-524):
      1. masks = masks_in @ protos  → (M, 160, 160) logits
      2. scale_masks() → remove letterbox padding, bilinear to (orig_h, orig_w)
      3. crop_mask(masks, bboxes)   → zero outside bbox at original coords
      4. .gt_(0.0)                  → threshold (equivalent to sigmoid > 0.5)

    Thresholding at original resolution preserves edge detail that is lost
    when thresholding at proto (160×160) resolution.
    """
    if mask_logits is None or len(mask_logits) == 0:
        return []

    _, ph, pw = mask_logits.shape

    # scale_masks: proto (160×160) → remove letterbox → original image size
    # Ultralytics ops.py:549-560
    gain_h = ph / orig_h
    gain_w = pw / orig_w
    gain = min(gain_h, gain_w)
    pad_w_proto = (pw - orig_w * gain) / 2
    pad_h_proto = (ph - orig_h * gain) / 2
    top = round(pad_h_proto - 0.1)
    left = round(pad_w_proto - 0.1)
    bottom = ph - round(pad_h_proto + 0.1)
    right = pw - round(pad_w_proto + 0.1)

    # Reuse one (orig_h, orig_w) crop mask across detections — fill(0)
    # per iteration is microseconds; np.zeros() per detection on a
    # 1080p image would cost ~1 ms × N detections.
    crop = np.zeros((orig_h, orig_w), dtype=np.float32)
    result = []
    for i, logit in enumerate(mask_logits):
        valid = logit[top:bottom, left:right]
        upsampled = cv2.resize(valid, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)

        x1 = int(max(0, boxes_orig[i, 0]))
        y1 = int(max(0, boxes_orig[i, 1]))
        x2 = int(min(orig_w, boxes_orig[i, 2] + 0.5))
        y2 = int(min(orig_h, boxes_orig[i, 3] + 0.5))
        crop.fill(0)
        crop[y1:y2, x1:x2] = 1.0
        upsampled *= crop

        # > 0 on logits ≡ sigmoid > 0.5
        result.append((upsampled > 0.0).astype(np.uint8))

    return result


def crop_and_threshold(mask_logits, boxes_model, input_h, input_w):
    """Non-retina mask decode: crop at proto resolution, then threshold.

    Matches ultralytics.utils.ops.process_mask() (non-retina path).
    Returns binary masks at proto resolution (e.g. 160×160).
    """
    _, ph, pw = mask_logits.shape
    scale_x = pw / input_w
    scale_y = ph / input_h

    # Reuse one (ph, pw) crop mask — fill(0) per iteration is much
    # cheaper than np.zeros() for the 300-detection NMS limit.
    logits = mask_logits.copy()
    crop = np.zeros((ph, pw), dtype=np.float32)
    for i in range(len(logits)):
        x1 = int(max(0, boxes_model[i, 0] * scale_x))
        y1 = int(max(0, boxes_model[i, 1] * scale_y))
        x2 = int(min(pw, boxes_model[i, 2] * scale_x + 0.5))
        y2 = int(min(ph, boxes_model[i, 3] * scale_y + 0.5))
        crop.fill(0)
        crop[y1:y2, x1:x2] = 1.0
        logits[i] *= crop

    return (logits > 0.0).astype(np.uint8)


def upsample_masks(masks_binary, scale, pad_x, pad_y,
                   orig_w, orig_h, input_h, input_w):
    """Upsample proto-resolution binary masks to original image size.

    Removes letterbox padding and resizes to original dimensions.
    """
    if len(masks_binary) == 0:
        return []
    result = []
    for mask in masks_binary:
        full = cv2.resize(mask.astype(np.float32), (input_w, input_h),
                          interpolation=cv2.INTER_LINEAR)
        valid_w = int(round(orig_w * scale))
        valid_h = int(round(orig_h * scale))
        cropped = full[pad_y:pad_y + valid_h, pad_x:pad_x + valid_w]
        orig_mask = cv2.resize(cropped, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)
        result.append((orig_mask > 0.5).astype(np.uint8))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FP32 ONNX reference YOLOv8-seg validation (onnxruntime)")
    parser.add_argument("--model", required=True,
                        help="Path to YOLOv8-seg .onnx model")
    parser.add_argument("--images", required=True,
                        help="Path to image directory")
    parser.add_argument("--output", default="results_onnx_fp32.json",
                        help="Output COCO results JSON path")
    parser.add_argument("--score-threshold", type=float, default=0.001,
                        help="Score threshold for NMS (default: 0.001, "
                             "matches Ultralytics default)")
    parser.add_argument("--iou-threshold", type=float, default=0.7,
                        help="IoU threshold for NMS (default: 0.7, "
                             "matches Ultralytics default)")
    parser.add_argument("--max-detections", type=int, default=300,
                        help="Maximum detections per image (default: 300, "
                             "matches Ultralytics max_det)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations before timing")
    parser.add_argument("--gt", default=None,
                        help="Path to COCO GT JSON for mAP evaluation")
    parser.add_argument("--with-masks", action="store_true", default=False,
                        help="Enable mask decode and segm evaluation")
    parser.add_argument("--fast-masks", action="store_true", default=False,
                        help="Use the lower-accuracy / lower-latency non-"
                             "retina mask decode path (Ultralytics "
                             "`ops.process_mask`). Default (without this "
                             "flag) is the high-accuracy retina path "
                             "(`ops.process_mask_native`).")
    parser.add_argument("--provider", default="cpu",
                        choices=["cpu", "cuda"],
                        help="ONNX Runtime execution provider")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        stream=sys.stderr,
    )

    try:
        import onnxruntime as ort
    except ImportError:
        log.error("onnxruntime required. Install with: pip install onnxruntime")
        sys.exit(1)

    img_dir = Path(args.images)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        log.error("Error: no images found in %s", args.images)
        sys.exit(1)

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.provider == "cuda"
                 else ["CPUExecutionProvider"])
    log.info("Loading model: %s", args.model)
    session = ort.InferenceSession(args.model, providers=providers)
    active_provider = session.get_providers()[0]

    input_info = session.get_inputs()[0]
    _, channels, input_h, input_w = input_info.shape
    input_name = input_info.name

    outputs_info = session.get_outputs()
    if len(outputs_info) < 2:
        log.warning(
            "expected 2 outputs (pred + protos), got %d. Masks unavailable.",
            len(outputs_info),
        )
        args.with_masks = False

    log.info("Model:    %s", args.model)
    log.info("Provider: %s", active_provider)
    log.info("Images:   %d from %s", len(image_paths), args.images)
    log.info("Input:    %s [1, %d, %d, %d] float32",
             input_name, channels, input_h, input_w)
    for o in outputs_info:
        log.info("  %-12s: %s %s", o.name, o.shape, o.type)
    log.info("Output:   %s", args.output)
    log.info("Masks:    %s", "enabled" if args.with_masks else "disabled")

    if args.warmup > 0:
        log.info("Warmup: %d iterations...", args.warmup)
        for i in range(min(args.warmup, len(image_paths))):
            bgr = cv2.imread(image_paths[i])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor, *_ = preprocess(rgb, input_h, input_w)
            session.run(None, {input_name: tensor})

    timings: dict[str, list[float]] = {
        "decode": [], "preprocess": [], "inference": [],
        "postprocess": [], "total": [],
    }
    all_results: list[dict] = []

    for idx, img_path in enumerate(image_paths):
        t_total = time.perf_counter()
        filename = Path(img_path).name

        # 1. Image decode
        t0 = time.perf_counter()
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        timings["decode"].append((time.perf_counter() - t0) * 1e3)

        # 2. Preprocess (Letterbox → float32/255 → HWC→BCHW). NO quantization.
        t0 = time.perf_counter()
        tensor, scale_f, pad_x, pad_y, orig_w, orig_h = preprocess(
            rgb, input_h, input_w)
        timings["preprocess"].append((time.perf_counter() - t0) * 1e3)

        # 3. ONNX Runtime inference
        t0 = time.perf_counter()
        ort_outputs = session.run(None, {input_name: tensor})
        timings["inference"].append((time.perf_counter() - t0) * 1e3)

        # 4. Postprocess
        t0 = time.perf_counter()
        output0 = ort_outputs[0]
        output1 = ort_outputs[1] if len(ort_outputs) > 1 else None

        boxes_np, scores_np, classes_np, masks_proto = decode_yolo_output(
            output0, output1, input_h, input_w,
            args.score_threshold, args.iou_threshold,
            args.max_detections, with_masks=args.with_masks,
        )
        timings["postprocess"].append((time.perf_counter() - t0) * 1e3)

        boxes_orig = unletterbox_boxes(
            boxes_np, scale_f, pad_x, pad_y, orig_w, orig_h)

        masks_orig = None
        if masks_proto is not None:
            if not args.fast_masks:
                masks_orig = decode_masks_retina(
                    masks_proto, boxes_orig, scale_f, pad_x, pad_y,
                    orig_w, orig_h, input_h, input_w)
            else:
                masks_binary = crop_and_threshold(
                    masks_proto, boxes_np, input_h, input_w)
                masks_orig = upsample_masks(
                    masks_binary, scale_f, pad_x, pad_y,
                    orig_w, orig_h, input_h, input_w)

        timings["total"].append((time.perf_counter() - t_total) * 1e3)

        results = build_coco_results(
            img_path, boxes_orig, scores_np, classes_np, masks_orig)
        all_results.extend(results)

        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(image_paths) - 1:
            avg = np.mean(timings["total"][-10:])
            log.info(
                "  [%4d/%d] %20s  %3d dets  %6.1f ms/img",
                idx + 1, len(image_paths), filename, len(results), avg,
            )

    # ── Timing summary ───────────────────────────────────────────────────
    print()
    hdr = summary_header()
    print(hdr)
    print("─" * len(hdr))
    print(fmt_row("image decode", trimmed_stats(timings["decode"])))
    print(fmt_row("preprocess", trimmed_stats(timings["preprocess"])))
    print(fmt_row("inference", trimmed_stats(timings["inference"])))
    print(fmt_row("postprocess", trimmed_stats(timings["postprocess"])))
    print("─" * len(hdr))
    s = trimmed_stats(timings["total"])
    print(fmt_row("TOTAL", s))
    if s["mean"] > 0:
        print(f"  Throughput: {1000 / s['mean']:.1f} FPS")

    model_path = [
        timings["preprocess"][i] + timings["inference"][i]
        + timings["postprocess"][i]
        for i in range(len(timings["total"]))]
    mp = trimmed_stats(model_path)
    print(f"\n  Model-path (pre+inf+post): {mp['mean']:.2f} ms "
          f"({1000/mp['mean']:.1f} FPS)")
    print(f"  Outlier filter: 1% trimmed "
          f"({s['n']}/{len(timings['total'])} samples)")
    print(f"  Total detections: {len(all_results)}")

    save_coco_results(all_results, args.output)
    log.info("Saved to: %s", args.output)

    if args.gt:
        print()
        from .coco_eval import evaluate_coco

        metrics = evaluate_coco(args.gt, args.output, iou_type="bbox")
        if metrics:
            print(f"\n  Box  mAP@0.50:0.95 = {metrics['AP']:.4f}")
            print(f"  Box  mAP@0.50      = {metrics['AP50']:.4f}")
            print(f"  Box  mAR@100       = {metrics['AR100']:.4f}")

        if args.with_masks:
            with open(args.output) as f:
                sample = json.load(f)
            if sample and "segmentation" in sample[0]:
                seg_metrics = evaluate_coco(
                    args.gt, args.output, iou_type="segm")
                if seg_metrics:
                    print(f"\n  Mask mAP@0.50:0.95 = {seg_metrics['AP']:.4f}")
                    print(f"  Mask mAP@0.50      = {seg_metrics['AP50']:.4f}")
                    print(f"  Mask mAR@100       = {seg_metrics['AR100']:.4f}")


if __name__ == "__main__":
    main()
