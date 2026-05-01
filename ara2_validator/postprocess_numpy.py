"""NumPy/OpenCV postprocessing for Ultralytics YOLO split-decoder output.

Handles the full decode pipeline:
  1. Dequantize each tensor (done upstream in model.run_inference)
  2. Reshape to [N, features] layout
  3. Convert boxes from xcycwh → xyxy pixel coordinates
  4. Apply class-aware NMS
  5. Decode instance segmentation masks via prototype matmul
  6. Crop and threshold masks to bounding boxes

Box coordinate flow:
  Model output → xcycwh in 640×640 letterboxed space → xyxy → NMS
  → for COCO output: un-letterbox to original image coordinates
"""

from __future__ import annotations

import cv2
import numpy as np

from .nms import nms_with_masks, nms_class_aware
from .preprocess import LetterboxInfo


def xcycwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [xc, yc, w, h] → [x1, y1, x2, y2]."""
    xy = boxes[:, :2]
    wh = boxes[:, 2:4]
    return np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)


def postprocess_numpy(
    outputs: dict[str, np.ndarray],
    input_h: int,
    input_w: int,
    score_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    with_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Decode YOLO split-decoder outputs using pure NumPy.

    Parameters
    ----------
    outputs : dict
        Dequantized float32 tensors from model.run_inference():
          "boxes"     : [1, 4, N]  (xcycwh pixel coords)
          "scores"    : [1, nc, N]
          "mask_coefs": [1, 32, N] (optional)
          "protos"    : [1, C, H, W] (optional)
    input_h, input_w : int
        Model input dimensions (e.g. 640×640).
    score_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        IoU threshold for NMS.
    max_detections : int
        Maximum detections to keep.
    with_masks : bool
        Whether to decode instance segmentation masks.

    Returns
    -------
    boxes : (M, 4) xyxy in model input pixel coordinates [0, input_w/h]
    scores : (M,) confidence values
    classes : (M,) class indices (0-indexed)
    mask_logits : (M, ph, pw) float logits at proto resolution, or None.
        Use decode_masks() for non-retina or decode_masks_retina() for
        retina-style decode.
    """
    # --- Unpack and reshape ---
    # Handle either merged "boxes" (4, N) or split "box_xy"/"box_wh" (2, N) each
    if "boxes" in outputs:
        boxes_raw = outputs["boxes"]
    elif "box_xy" in outputs and "box_wh" in outputs:
        xy = outputs["box_xy"]
        wh = outputs["box_wh"]
        if xy.ndim == 3:
            xy = xy[0]
        if wh.ndim == 3:
            wh = wh[0]
        # Ensure [2, N] orientation then stack → [4, N]
        if xy.shape[0] > xy.shape[1]:
            xy = xy.T
        if wh.shape[0] > wh.shape[1]:
            wh = wh.T
        boxes_raw = np.concatenate([xy, wh], axis=0)  # [4, N]
    else:
        raise KeyError("outputs must contain 'boxes' or 'box_xy'+'box_wh'")
    scores_raw = outputs["scores"]

    # Remove batch dim: [1, C, N] → [C, N]
    if boxes_raw.ndim == 3:
        boxes_raw = boxes_raw[0]
    if scores_raw.ndim == 3:
        scores_raw = scores_raw[0]

    # Transpose [C, N] → [N, C] if channels-first
    if boxes_raw.shape[0] < boxes_raw.shape[1]:
        boxes_raw = boxes_raw.T          # [N, 4]
    if scores_raw.shape[0] < scores_raw.shape[1]:
        scores_raw = scores_raw.T        # [N, nc]

    # --- Box decode: xcycwh → xyxy ---
    boxes_xyxy = xcycwh_to_xyxy(boxes_raw)

    # --- Mask coefficients ---
    mask_coeffs = None
    has_masks = (with_masks
                 and "mask_coefs" in outputs
                 and "protos" in outputs)
    if has_masks:
        mc = outputs["mask_coefs"]
        if mc.ndim == 3:
            mc = mc[0]
        if mc.shape[0] < mc.shape[1]:
            mc = mc.T                    # [N, 32]
        mask_coeffs = mc

    # --- NMS ---
    if mask_coeffs is not None:
        boxes_nms, scores_nms, classes_nms, mc_nms = nms_with_masks(
            boxes_xyxy, scores_raw, mask_coeffs,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_detections,
            class_aware=True,
        )
    else:
        boxes_nms, scores_nms, classes_nms, _ = nms_class_aware(
            boxes_xyxy, scores_raw,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        mc_nms = None

    # --- Compute mask logits (if masks requested) ---
    mask_logits = None
    if mc_nms is not None and len(mc_nms) > 0:
        protos = outputs["protos"]
        mask_logits = compute_mask_logits(mc_nms, protos)

    return boxes_nms, scores_nms, classes_nms, mask_logits


def compute_mask_logits(
    mask_coeffs: np.ndarray,
    protos: np.ndarray,
) -> np.ndarray:
    """Compute raw mask logits at prototype resolution.

    Returns float logits (M, ph, pw) — thresholding is deferred to allow
    the caller to choose retina (upsample then threshold) or non-retina
    (threshold at proto resolution then upsample).

    Ultralytics equivalent (ops.py:521):
      masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    """
    # Ensure protos are [C, H, W] (channels-first)
    if protos.ndim == 4:
        protos = protos[0]
    if protos.shape[0] != 32 and protos.shape[-1] == 32:
        protos = np.transpose(protos, (2, 0, 1))
    c, ph, pw = protos.shape

    # (M, 32) @ (32, ph*pw) → (M, ph*pw) → (M, ph, pw)
    logits = mask_coeffs @ protos.reshape(c, -1)
    return logits.reshape(-1, ph, pw)


def decode_masks(
    mask_logits: np.ndarray,
    boxes: np.ndarray,
    input_h: int,
    input_w: int,
) -> np.ndarray:
    """Non-retina mask decode: crop at proto resolution, then threshold.

    Matches ultralytics.utils.ops.process_mask() (non-retina path).
    Masks are returned at proto resolution (e.g. 160×160) and must be
    upsampled to original image size via unletterbox_masks().

    Parameters
    ----------
    mask_logits : (M, ph, pw) float logits from compute_mask_logits()
    boxes : (M, 4) xyxy in model input pixel coordinates
    input_h, input_w : int
        Model input dimensions (e.g. 640, 640).

    Returns
    -------
    masks : (M, ph, pw) binary uint8 masks at proto resolution.
    """
    _, ph, pw = mask_logits.shape

    # Scale boxes from model-input space to proto space
    scale_x = pw / input_w
    scale_y = ph / input_h

    # Reuse one crop mask across detections — fill(0) inside the loop
    # is microseconds; np.zeros((ph, pw)) per-detection adds up on
    # 300-detection NMS outputs.
    logits = mask_logits.copy()
    crop = np.zeros((ph, pw), dtype=np.float32)
    for i in range(len(logits)):
        x1 = int(max(0, boxes[i, 0] * scale_x))
        y1 = int(max(0, boxes[i, 1] * scale_y))
        x2 = int(min(pw, boxes[i, 2] * scale_x + 0.5))
        y2 = int(min(ph, boxes[i, 3] * scale_y + 0.5))
        crop.fill(0)
        crop[y1:y2, x1:x2] = 1.0
        logits[i] *= crop

    return (logits > 0.0).astype(np.uint8)


def decode_masks_retina(
    mask_logits: np.ndarray,
    boxes_orig: np.ndarray,
    lb_info: LetterboxInfo,
    input_h: int,
    input_w: int,
) -> list[np.ndarray]:
    """Retina mask decode: upsample logits to original size, then crop + threshold.

    Matches ultralytics.utils.ops.process_mask_native() (retina path):
      1. scale_masks() — remove letterbox padding, bilinear to (orig_h, orig_w)
      2. crop_mask()   — zero outside bbox at original coordinates
      3. .gt_(0.0)     — threshold (equivalent to sigmoid > 0.5)

    Thresholding at original resolution preserves mask edge detail compared
    to the non-retina path which thresholds at proto resolution (160×160).

    Parameters
    ----------
    mask_logits : (M, ph, pw) float logits from compute_mask_logits()
    boxes_orig : (M, 4) xyxy in original image coordinates
    lb_info : LetterboxInfo
    input_h, input_w : int
        Model input dimensions (e.g. 640, 640).

    Returns
    -------
    list of (orig_h, orig_w) binary uint8 masks in original image space.
    """
    if len(mask_logits) == 0:
        return []

    _, ph, pw = mask_logits.shape
    orig_h, orig_w = lb_info.orig_h, lb_info.orig_w

    # scale_masks: compute letterbox padding at proto resolution
    # Ultralytics ops.py:549-560
    gain = min(ph / orig_h, pw / orig_w)
    pad_w_proto = (pw - orig_w * gain) / 2
    pad_h_proto = (ph - orig_h * gain) / 2
    top = round(pad_h_proto - 0.1)
    left = round(pad_w_proto - 0.1)
    bottom = ph - round(pad_h_proto + 0.1)
    right = pw - round(pad_w_proto + 0.1)

    # Reuse the crop mask across detections — fill(0) per iteration
    # is microseconds; allocating np.zeros((orig_h, orig_w)) per
    # detection on a 1080p image would cost ~1 ms × N detections.
    crop = np.zeros((orig_h, orig_w), dtype=np.float32)
    result = []
    for i, logit in enumerate(mask_logits):
        # Remove letterbox padding at proto resolution, bilinear to original
        valid = logit[top:bottom, left:right]
        upsampled = cv2.resize(valid, (orig_w, orig_h),
                               interpolation=cv2.INTER_LINEAR)

        # crop_mask: zero outside bbox at original coordinates
        x1 = int(max(0, boxes_orig[i, 0]))
        y1 = int(max(0, boxes_orig[i, 1]))
        x2 = int(min(orig_w, boxes_orig[i, 2] + 0.5))
        y2 = int(min(orig_h, boxes_orig[i, 3] + 0.5))
        crop.fill(0)
        crop[y1:y2, x1:x2] = 1.0
        upsampled *= crop

        result.append((upsampled > 0.0).astype(np.uint8))

    return result


def unletterbox_boxes(
    boxes: np.ndarray,
    lb_info: LetterboxInfo,
) -> np.ndarray:
    """Map xyxy boxes from model-input space back to original image coords.

    Parameters
    ----------
    boxes : (M, 4) xyxy in 640×640 letterboxed space
    lb_info : LetterboxInfo

    Returns
    -------
    boxes : (M, 4) xyxy in original image pixel coordinates
    """
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - lb_info.pad_x) / lb_info.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - lb_info.pad_y) / lb_info.scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, lb_info.orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, lb_info.orig_h)
    return out


def unletterbox_masks(
    masks: np.ndarray,
    lb_info: LetterboxInfo,
    input_h: int,
    input_w: int,
) -> list[np.ndarray]:
    """Resize masks from proto resolution to original image size.

    Parameters
    ----------
    masks : (M, ph, pw) binary masks at proto resolution
    lb_info : LetterboxInfo
    input_h, input_w : int

    Returns
    -------
    list of (orig_h, orig_w) binary masks in original image space
    """
    if len(masks) == 0:
        return []

    result = []
    for mask in masks:
        # Upsample from proto resolution to model-input resolution
        full = cv2.resize(mask, (input_w, input_h),
                          interpolation=cv2.INTER_LINEAR)
        # Crop the letterbox region (remove padding)
        ph, pw = full.shape[:2]
        # Compute the valid (non-padded) region in model-input space
        valid_w = int(round(lb_info.orig_w * lb_info.scale))
        valid_h = int(round(lb_info.orig_h * lb_info.scale))
        cropped = full[lb_info.pad_y:lb_info.pad_y + valid_h,
                       lb_info.pad_x:lb_info.pad_x + valid_w]
        # Resize to original image dimensions
        orig_mask = cv2.resize(cropped, (lb_info.orig_w, lb_info.orig_h),
                               interpolation=cv2.INTER_LINEAR)
        # Re-threshold after resize (bilinear can produce non-binary values)
        result.append((orig_mask > 0.5).astype(np.uint8))
    return result
