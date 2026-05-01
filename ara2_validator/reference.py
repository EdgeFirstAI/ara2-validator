#!/usr/bin/env python3
"""Reference YOLOv8-seg inference on Ara240 using NXP dvapi + numpy/OpenCV.

This script is a self-contained reference implementation that demonstrates
how to run a YOLOv8-seg DVM model on the NXP Ara240 NPU using only:
  - NXP's dvapi.py (ctypes wrapper around libaraclient)
  - OpenCV for image loading/letterbox resize
  - NumPy for quantization, dequantization, NMS, and mask decode

No EdgeFirst tooling is required. The preprocessing and postprocessing
steps are documented inline with references to the equivalent Ultralytics
code paths for comparison.

Timing breakdown
----------------
The report separates image acquisition (decode) from the model pipeline:

  decode      — JPEG → numpy RGB buffer (cv2.imread + cvtColor)
  preprocess  — Letterbox resize, normalize [0,1], quantize to int8, HWC→CHW
  inference   — PCIe DMA host→device + NPU compute + PCIe DMA device→host
  postprocess — Dequantize outputs, box decode, class-aware NMS, mask matmul

The "model-path" aggregate (pre + inf + post) represents what would run
after image acquisition in a real camera/video pipeline.

Ara240 inference sub-timings come from the dvproxy stats reported after
each inference:
  pci input     — Host DRAM → Ara240 device DRAM (PCIe DMA write)
  npu compute   — Neural processor core execution time
  pci output    — Ara240 device DRAM → Host DRAM (PCIe DMA read)

Ultralytics comparison
----------------------
Ultralytics' ONNX predictor does:
  1. LetterBox(stride=32, center=True) with cv2.resize + cv2.copyMakeBorder
  2. BGR→RGB → BHWC→BCHW → float32 / 255.0 (NO quantization — stays FP32)
  3. ONNX Runtime inference (CPU or CUDA provider)
  4. Transpose BCN→BNC, xywh→xyxy, torchvision.ops.nms (compiled C++)
  5. Mask: coeffs @ protos → crop → bilinear upsample → threshold > 0

This reference mirrors steps 1, 4, and 5 exactly. Steps 2-3 differ
because the Ara240 NPU requires int8 quantized input and produces
quantized output that must be dequantized before decoding.

Usage
-----
    python -m ara2_validator.reference \\
        --model yolov8n-seg-kinara-1.2.1.dvm \\
        --images /root/coco128/images/ \\
        --output results_reference.json
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# dvapi.py is vendored from the NXP Ara240 runtime SDK
# (ara2-runtime-r1.2.1/art/linux/aarch64/include/dvapi.py) with a patch
# to _dvApiObj() so it can find libaraclient_aarch64.so installed at
# /usr/lib on the imx8mp-frdm board, not just at SDK-relative paths.
# ---------------------------------------------------------------------------
from ._stats import fmt_row, summary_header, trimmed_stats
from .coco_output import build_coco_results, save_coco_results
from .dvapi import DVSession, DVTensor, dv_status_code

log = logging.getLogger("ara2_validator.reference")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TensorMeta:
    """Metadata for one output tensor parsed from the model params."""
    index: int
    name: str
    role: str          # "protos", "mask_coefs", "scores", "boxes"
    shape: tuple       # (C, H, W) or (C, N) — as reported by model
    dtype: np.dtype
    qn: float          # quantization scale: float_val = (raw - offset) * qn
    offset: int        # quantization zero-point


@dataclass
class ModelInfo:
    """Parsed model specification."""
    input_c: int
    input_h: int
    input_w: int
    input_qn: float
    input_offset: int
    input_signed: bool
    input_bpp: int
    outputs: dict[str, TensorMeta] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_trailing_ones(shape: list[int]) -> list[int]:
    """Strip trailing dimensions of size 1 (Ara240 padding convention)."""
    while len(shape) > 2 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


def _infer_role(shape: list[int], num_protos: int | None = None,
                seen: set | None = None) -> str:
    """Guess output tensor role from its shape.

    Ara240 DVMs report shapes as [C, H, W] (channels-first) with an
    optional trailing padding dim of 1. This heuristic identifies:
      - protos: 3D spatial with 32 channels and large spatial dims
      - boxes: feature dim == 4
      - box_xy / box_wh: feature dim == 2 (split decoder)
      - scores: feature dim == 80 (COCO)
      - mask_coefs: feature dim == 32 (matching proto count)
    """
    s = _strip_trailing_ones(list(shape))
    if len(s) == 3 and 32 in s and max(s) > 100:
        return "protos"
    if len(s) >= 2:
        feat = min(s[0], s[-1]) if len(s) == 2 else s[0]
        if feat == 4:
            return "boxes"
        if feat == 2:
            # Split decoder: first (2, N) is box_xy, second is box_wh
            if seen and "box_xy" in seen:
                return "box_wh"
            return "box_xy"
        if feat == 80:
            return "scores"
        if feat == 32 or (num_protos and feat == num_protos):
            return "mask_coefs"
    return "unknown"


def _parse_model(dv_model) -> ModelInfo:
    """Extract input/output metadata from a loaded DVModel.

    The DVModel object (from dvapi.py) exposes:
      - input_param[i].preprocess_param[0] for quantization info
      - input_param[i].{nch, height, width, bpp} for tensor shape
      - output_param[j].postprocess_param[0] for dequant info
      - output_param[j].{nch, height, width, depth, bpp, size} for shape
    """
    ip = dv_model.input_param[0]
    pp = ip.preprocess_param
    info = ModelInfo(
        input_c=ip.nch,
        input_h=ip.height,
        input_w=ip.width,
        input_qn=pp.qn,
        input_offset=pp.offset,
        input_signed=pp.is_signed,
        input_bpp=ip.bpp,
    )

    raw_shapes = []
    for j in range(dv_model.num_outputs):
        op = dv_model.output_param[j]
        shape = _strip_trailing_ones([op.nch, op.height, op.width])
        if op.depth > 1:
            shape = [op.nch, op.depth, op.height, op.width]
            shape = _strip_trailing_ones(shape)
        raw_shapes.append(shape)

    proto_ch = None
    for s in raw_shapes:
        if len(s) == 3 and 32 in s and max(s) > 100:
            proto_ch = 32
            break

    seen_roles: set = set()
    for j in range(dv_model.num_outputs):
        op = dv_model.output_param[j]
        opp = op.postprocess_param
        shape = raw_shapes[j]
        role = _infer_role(shape, proto_ch, seen_roles)
        seen_roles.add(role)

        if op.bpp == 1:
            dtype = np.int8 if opp.is_signed else np.uint8
        elif op.bpp == 2:
            dtype = np.int16 if opp.is_signed else np.uint16
        else:
            dtype = np.float32

        info.outputs[role] = TensorMeta(
            index=j,
            name=op.layer_name,
            role=role,
            shape=tuple(shape),
            dtype=dtype,
            qn=opp.qn,
            offset=opp.offset,
        )

    return info


def _read_labels(dvm_path: str, fallback_path: str | None) -> list[str]:
    """Try reading labels from DVM ZIP trailer, then fallback file."""
    data = Path(dvm_path).read_bytes()
    for i in range(len(data) - 4):
        if data[i:i + 4] == b"PK\x03\x04":
            try:
                zf = zipfile.ZipFile(io.BytesIO(data[i:]))
                if "labels.txt" in zf.namelist():
                    return [ln for ln in
                            zf.read("labels.txt").decode().strip().splitlines()
                            if ln.strip()]
            except zipfile.BadZipFile:
                continue
    if fallback_path and Path(fallback_path).exists():
        return [ln for ln in Path(fallback_path).read_text().splitlines()
                if ln.strip()]
    return []


# ── Preprocessing ─────────────────────────────────────────────────────────────
#
# Ultralytics equivalent: ultralytics/data/augment.py::LetterBox.__call__()
#   1. Compute scale = min(dst/src_h, dst/src_w)
#   2. cv2.resize(img, (new_w, new_h), INTER_LINEAR)
#   3. cv2.copyMakeBorder with value=(114,114,114) centered
#   4. BGR→RGB (done in predictor.preprocess)
#   5. HWC→CHW transpose
#   6. float() / 255.0   ← stays FP32, no quantization
#
# Our reference differs at step 6: we quantize to int8 because the Ara240
# NPU only accepts quantized input. The quantization formula is:
#   quantized = round(pixel/255.0 / qn + offset)
# For this model (qn=0.003915, offset=-127): ≈ round(pixel*1.003 - 127)
# which is essentially a linear remap [0,255] → [-127,128] (standard
# full-range signed int8 affine quantization).

def letterbox(img, target_h, target_w, color=(114, 114, 114)):
    """Letterbox resize matching Ultralytics LetterBox(center=True).

    Uses cv2.copyMakeBorder (same as Ultralytics) rather than np.full +
    array assignment. Padding split follows Ultralytics' round(x-0.1) /
    round(x+0.1) convention for asymmetric odd-remainder cases.

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


def preprocess(rgb, input_h, input_w, qn, offset, signed):
    """Quantize and transpose an RGB image for Ara240 NPU input.

    Pipeline: letterbox → normalize [0,1] → affine quantize → HWC→CHW

    Returns (flat_int8_tensor, scale, pad_x, pad_y, orig_w, orig_h).
    """
    orig_h, orig_w = rgb.shape[:2]
    padded, scale, pad_x, pad_y = letterbox(rgb, input_h, input_w)

    normalized = padded.astype(np.float32) / 255.0
    quantized = np.round(normalized / qn + offset)
    dtype = np.int8 if signed else np.uint8
    iinfo = np.iinfo(dtype)
    quantized = np.clip(quantized, iinfo.min, iinfo.max).astype(dtype)

    chw = np.transpose(quantized, (2, 0, 1))
    flat = np.ascontiguousarray(chw).reshape(-1)

    return flat, scale, pad_x, pad_y, orig_w, orig_h


# ── Postprocessing helpers ────────────────────────────────────────────────────
#
# Box decode + NMS + mask decode are delegated to postprocess_numpy so
# this module stays thin. See postprocess_numpy.py for the per-step
# Ultralytics-equivalent commentary.

def _dequantize(raw: np.ndarray, qn: float, offset: int) -> np.ndarray:
    """Affine dequant: float = (raw_int - offset) * qn."""
    return (raw.astype(np.float32) - offset) * qn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reference Ara240 YOLOv8-seg inference (dvapi + numpy/OpenCV)")
    parser.add_argument("--model", required=True,
                        help="Path to .dvm model file")
    parser.add_argument("--images", required=True,
                        help="Path to image directory")
    parser.add_argument("--output", default="results_reference.json",
                        help="Output COCO results JSON path")
    parser.add_argument("--labels", default=None,
                        help="Path to labels.txt")
    parser.add_argument("--score-threshold", type=float, default=0.001,
                        help="Score threshold for NMS")
    parser.add_argument("--iou-threshold", type=float, default=0.7,
                        help="IoU threshold for NMS")
    parser.add_argument("--max-detections", type=int, default=300,
                        help="Maximum detections per image")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process. NOTE: the "
                             "dvapi.py / libaraclient stack underneath this "
                             "reference path is unstable past roughly 100 "
                             "consecutive inferences in a single process — "
                             "you will see a glibc 'double free or "
                             "corruption' abort on coco-val5k. coco128 (128 "
                             "images) generally completes; for a single-"
                             "process full-val5k run, prefer the "
                             "edgefirst.py path (which uses the Rust-backed "
                             "edgefirst-ara2 wrapper and is not affected).")
    parser.add_argument("--socket", default="/var/run/ara2.sock",
                        help="Ara240 dvproxy Unix socket path")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations before timing")
    parser.add_argument("--gt", default=None,
                        help="Path to COCO GT JSON for mAP evaluation")
    parser.add_argument("--with-masks", action="store_true", default=False,
                        help="Enable mask decode and segmentation evaluation")
    parser.add_argument("--fast-masks", action="store_true", default=False,
                        help="Use the lower-accuracy / lower-latency non-"
                             "retina mask decode path (Ultralytics "
                             "`ops.process_mask`): threshold logits at proto "
                             "resolution, bilinear-upsample binary mask, re-"
                             "threshold. Default (without this flag) is the "
                             "high-accuracy retina path "
                             "(`ops.process_mask_native`): bilinear-upsample "
                             "the f32 logit plane to original resolution, "
                             "then threshold. Retina wins by ~0.6 pp mask "
                             "mAP on coco128 (with this dvapi+numpy stack "
                             "the speed difference is also small).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
        stream=sys.stderr,
    )

    img_dir = Path(args.images)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        log.error("Error: no images found in %s", args.images)
        sys.exit(1)

    # ── Load model via dvapi ──────────────────────────────────────────────
    # dvapi.py provides the DVSession/DVModel high-level wrapper around
    # the NXP libaraclient C library. Connection is via Unix socket
    # to the dvproxy daemon running on the board.
    #
    # Ultralytics equivalent: onnxruntime.InferenceSession(model_path)
    log.info("Connecting to Ara240 proxy at %s...", args.socket)
    ret, session = DVSession.create_via_unix_socket(args.socket)
    if ret != dv_status_code.DV_SUCCESS:
        log.error("cannot connect to dvproxy: %s", ret)
        sys.exit(1)

    with session:
        ret, endpoints = session.get_endpoint_list()
        if ret != dv_status_code.DV_SUCCESS or not endpoints:
            log.error("no Ara240 endpoints available")
            sys.exit(1)

        log.info("Loading model: %s", args.model)
        ret, dv_model = session.load_model_from_file(
            endpoints[0], args.model)
        if ret != dv_status_code.DV_SUCCESS:
            log.error("model load failed: %s", ret)
            sys.exit(1)

        meta = _parse_model(dv_model)
        labels = _read_labels(args.model, args.labels)

        log.info("Model:   %s", args.model)
        log.info("Images:  %d from %s", len(image_paths), args.images)
        log.info("Backend: reference (dvapi + numpy/OpenCV)")
        log.info("Output:  %s", args.output)
        log.info(
            "Input:   %d×%d×%d, qn=%.6f, offset=%d, signed=%s",
            meta.input_c, meta.input_h, meta.input_w,
            meta.input_qn, meta.input_offset, meta.input_signed,
        )
        for role, tm in meta.outputs.items():
            log.info(
                "  %-12s: shape=%s, dtype=%s, qn=%.6f, offset=%d",
                role, tm.shape, tm.dtype, tm.qn, tm.offset,
            )
        log.info("Labels:  %d classes", len(labels))

        # DVModel._allocate_output_tensors() creates a contiguous numpy
        # buffer and returns DVTensor slices keyed by output index.
        # Each DVTensor.numpy_data is a view into int8 backing memory.
        output_tensors = dv_model._allocate_output_tensors()

        if args.warmup > 0:
            log.info("Warmup: %d iterations...", args.warmup)
            for i in range(min(args.warmup, len(image_paths))):
                bgr = cv2.imread(image_paths[i])
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                flat, *_ = preprocess(
                    rgb, meta.input_h, meta.input_w,
                    meta.input_qn, meta.input_offset, meta.input_signed)
                input_tensor = DVTensor(flat, dv_model.input_param[0])
                dv_model.infer_sync([input_tensor], output_tensors)

        # postprocess = nms_decode + mask_decode (per-detection upsample
        # + threshold). RLE encode is COCO-output formatting and lives
        # in `rle`, excluded from postprocess + model-path so the
        # comparison with edgefirst (HAL) stays apples-to-apples.
        timings: dict[str, list[float]] = {
            "decode": [], "preprocess": [], "inference": [],
            "nms_decode": [], "mask_decode": [],
            "postprocess": [], "rle": [], "total": [],
            "pci_input": [], "npu_compute": [], "pci_output": [],
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

            # 2. Preprocess: letterbox → normalize → quantize → HWC→CHW
            t0 = time.perf_counter()
            flat, scale_f, pad_x, pad_y, orig_w, orig_h = preprocess(
                rgb, meta.input_h, meta.input_w,
                meta.input_qn, meta.input_offset, meta.input_signed)
            timings["preprocess"].append((time.perf_counter() - t0) * 1e3)

            # 3. NPU inference via dvproxy.
            #
            # DVModel.infer_sync() internally:
            #   1. Packs DVTensor → DVBlob (raw pointer + size)
            #   2. Calls dv_infer_async (sends input blobs over Unix socket
            #      to dvproxy, which DMA-writes to Ara240 device DRAM)
            #   3. Calls dv_infer_wait_for_completion (blocks until NPU
            #      finishes and output is DMA-read back to host DRAM)
            #   4. Populates DVInferStatistics with PCIe transfer and
            #      NPU compute timing from firmware counters
            t0 = time.perf_counter()
            input_tensor = DVTensor(flat, dv_model.input_param[0])
            ret, inf_req = dv_model.infer_sync(
                [input_tensor], output_tensors)
            inf_wall = (time.perf_counter() - t0) * 1e3
            timings["inference"].append(inf_wall)

            if ret != dv_status_code.DV_SUCCESS:
                log.warning("Inference failed on %s: %s", filename, ret)
                continue

            # NOTE: Despite the dvapi struct comments saying "milliseconds",
            # the firmware reports transfer times in *microseconds*.
            # Evidence: 2139 "ms" for a 1.2MB PCIe DMA is ~2.1ms, not 2.1s.
            stats = inf_req.stats
            if stats:
                timings["pci_input"].append(stats.input_transfer_time / 1000)
                timings["npu_compute"].append(stats.npu_compute_ms)
                timings["pci_output"].append(stats.output_transfer_time / 1000)

            # 4. Postprocess: dequantize → decode → NMS → mask decode.
            t0 = time.perf_counter()

            raw_by_role: dict[str, np.ndarray] = {}
            for role, tm in meta.outputs.items():
                raw = output_tensors[tm.index].numpy_data.copy()
                if tm.dtype != np.int8:
                    raw = raw.view(tm.dtype)
                raw = raw.reshape(tm.shape)
                raw_by_role[role] = raw

            from .postprocess_numpy import (
                decode_masks, decode_masks_retina, postprocess_numpy,
                unletterbox_boxes as _ulb, unletterbox_masks,
            )
            from .preprocess import LetterboxInfo

            outputs_float: dict[str, np.ndarray] = {}
            for role, tm in meta.outputs.items():
                outputs_float[role] = _dequantize(
                    raw_by_role[role], tm.qn, tm.offset)

            boxes_np, scores_np, classes_np, mask_logits = postprocess_numpy(
                outputs_float, meta.input_h, meta.input_w,
                args.score_threshold, args.iou_threshold,
                args.max_detections, with_masks=args.with_masks,
            )
            timings["nms_decode"].append((time.perf_counter() - t0) * 1e3)

            lb_info = LetterboxInfo(
                scale=scale_f, pad_x=pad_x, pad_y=pad_y,
                orig_w=orig_w, orig_h=orig_h)
            boxes_orig = _ulb(boxes_np, lb_info)

            # Mask decode (per-detection upsample + threshold) is part of
            # postprocess. Time it separately so the comparison with the
            # edgefirst HAL path can isolate the inference-side mask cost
            # from the COCO output formatting (build_coco_results / RLE).
            t_mask = time.perf_counter()
            orig_masks = None
            if mask_logits is not None and len(mask_logits) > 0:
                if not args.fast_masks:
                    orig_masks = decode_masks_retina(
                        mask_logits, boxes_orig, lb_info,
                        meta.input_h, meta.input_w)
                else:
                    binary = decode_masks(
                        mask_logits, boxes_np, meta.input_h, meta.input_w)
                    orig_masks = unletterbox_masks(
                        binary, lb_info, meta.input_h, meta.input_w)
            timings["mask_decode"].append((time.perf_counter() - t_mask) * 1e3)
            timings["postprocess"].append(
                timings["nms_decode"][-1] + timings["mask_decode"][-1])

            # RLE encode (build_coco_results) is COCO-output formatting,
            # not part of the inference pipeline. Time it here, exclude
            # it from postprocess + model-path so the HAL/dvapi
            # comparison stays apples-to-apples.
            t_rle = time.perf_counter()
            results = build_coco_results(
                img_path, boxes_orig, scores_np, classes_np, orig_masks)
            timings["rle"].append((time.perf_counter() - t_rle) * 1e3)
            timings["total"].append((time.perf_counter() - t_total) * 1e3)
            all_results.extend(results)

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(image_paths) - 1:
                avg = np.mean(timings["total"][-10:])
                log.info(
                    "  [%4d/%d] %20s  %3d dets  %6.1f ms/img",
                    idx + 1, len(image_paths), filename, len(results), avg,
                )

        # ── Timing summary ────────────────────────────────────────────────
        print()
        hdr = summary_header()
        print(hdr)
        print("─" * len(hdr))

        print(fmt_row("image decode", trimmed_stats(timings["decode"])))
        print(fmt_row("preprocess", trimmed_stats(timings["preprocess"])))
        print(fmt_row("inference (wall)", trimmed_stats(timings["inference"])))
        print(fmt_row("pci input (host→device)",
                      trimmed_stats(timings["pci_input"]), indent=1))
        print(fmt_row("npu compute",
                      trimmed_stats(timings["npu_compute"]), indent=1))
        print(fmt_row("pci output (device→host)",
                      trimmed_stats(timings["pci_output"]), indent=1))
        print(fmt_row("postprocess (nms+mask)",
                      trimmed_stats(timings["postprocess"])))
        print(fmt_row("nms_decode (numpy)",
                      trimmed_stats(timings["nms_decode"]), indent=1))
        print(fmt_row("mask_decode (numpy)",
                      trimmed_stats(timings["mask_decode"]), indent=1))
        if any(t > 0 for t in timings["rle"]):
            print(fmt_row("output formatting (rle)",
                          trimmed_stats(timings["rle"])))
        print("─" * len(hdr))
        s = trimmed_stats(timings["total"])
        print(fmt_row("TOTAL (incl. output)", s))
        if s["mean"] > 0:
            print(f"  Throughput: {1000 / s['mean']:.1f} FPS")

        model_path = [
            timings["preprocess"][i] + timings["inference"][i]
            + timings["postprocess"][i]
            for i in range(len(timings["total"]))]
        mp = trimmed_stats(model_path)
        print(f"\n  Model-path (pre+inf+post, excl. output): "
              f"{mp['mean']:.2f} ms ({1000/mp['mean']:.1f} FPS)")
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

            if args.with_masks and any(
                    "segmentation" in r for r in all_results):
                segm = evaluate_coco(args.gt, args.output, iou_type="segm")
                if segm:
                    print(f"\n  Mask mAP@0.50:0.95 = {segm['AP']:.4f}")
                    print(f"  Mask mAP@0.50      = {segm['AP50']:.4f}")
                    print(f"  Mask mAR@100       = {segm['AR100']:.4f}")

        # Explicitly unload model before session context exits to avoid
        # double-free in DVModel.__del__ after session disconnect.
        dv_model.unload()


if __name__ == "__main__":
    main()
