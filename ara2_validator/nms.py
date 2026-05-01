"""Pure NumPy NMS implementation (no TensorFlow/PyTorch dependency).

Supports both class-agnostic and class-aware (per-class) NMS.
"""

from __future__ import annotations

import numpy as np


def nms_class_agnostic(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.001,
    max_detections: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Class-agnostic NMS on multi-class scores.

    Parameters
    ----------
    boxes : (N, 4) xyxy
    scores : (N, nc) per-class confidences
    iou_threshold : float
    score_threshold : float
    max_detections : int

    Returns
    -------
    boxes : (M, 4) xyxy
    scores : (M,) confidence
    classes : (M,) class indices
    masks : None (placeholder for API compatibility)
    """
    # Take max class per box
    class_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)

    # Score filter
    keep = confs >= score_threshold
    boxes_f = boxes[keep]
    confs_f = confs[keep]
    class_ids_f = class_ids[keep]

    if len(boxes_f) == 0:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.intp),
                None)

    # Greedy NMS
    keep_idx = _greedy_nms(boxes_f, confs_f, iou_threshold)
    keep_idx = keep_idx[:max_detections]

    return (boxes_f[keep_idx], confs_f[keep_idx],
            class_ids_f[keep_idx], None)


def nms_class_aware(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.001,
    max_detections: int = 300,
    max_wh: int = 7680,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Class-aware NMS using the Ultralytics offset trick.

    Instead of running NMS independently per class (slow for 80 classes),
    we offset each box spatially by class_id * max_wh so boxes from
    different classes never overlap. Then a single greedy NMS pass
    handles all classes at once.

    This exactly matches Ultralytics nms.py:143:
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes = x[:, :4] + c
        i = torchvision.ops.nms(boxes, scores, iou_thres)

    Parameters
    ----------
    boxes : (N, 4) xyxy
    scores : (N, nc) per-class confidences
    max_wh : int
        Box offset per class. Must exceed max image dimension (default 7680
        matches Ultralytics). Each class's boxes are shifted by
        class_id * max_wh so they cannot intersect across classes.

    Returns
    -------
    Same as nms_class_agnostic.
    """
    # Take max class per anchor (Ultralytics nms.py:133)
    class_ids = scores.argmax(axis=1)
    confs = scores[np.arange(len(scores)), class_ids]

    # Score filter (Ultralytics nms.py:135)
    keep = confs >= score_threshold
    boxes_f = boxes[keep]
    confs_f = confs[keep]
    class_ids_f = class_ids[keep]

    if len(boxes_f) == 0:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.intp),
                None)

    # Pre-NMS limit: sort by conf, keep top candidates (Ultralytics: max_nms=30000)
    if len(boxes_f) > 30000:
        order = confs_f.argsort()[::-1][:30000]
        boxes_f = boxes_f[order]
        confs_f = confs_f[order]
        class_ids_f = class_ids_f[order]

    # Offset boxes by class_id * max_wh for class-aware NMS
    # (Ultralytics nms.py:143,149)
    offsets = class_ids_f[:, None].astype(np.float32) * max_wh
    boxes_offset = boxes_f + offsets

    # Single greedy NMS pass on offset boxes
    keep_idx = _greedy_nms(boxes_offset, confs_f, iou_threshold)
    keep_idx = keep_idx[:max_detections]

    return (boxes_f[keep_idx], confs_f[keep_idx],
            class_ids_f[keep_idx], None)


def nms_with_masks(
    boxes: np.ndarray,
    scores: np.ndarray,
    mask_coeffs: np.ndarray,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.001,
    max_detections: int = 300,
    class_aware: bool = True,
    max_wh: int = 7680,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NMS that also filters mask coefficients in lockstep with boxes.

    Uses the Ultralytics offset trick for class-aware NMS when
    class_aware=True (default). Falls back to class-agnostic NMS
    otherwise.

    Parameters
    ----------
    boxes : (N, 4) xyxy
    scores : (N, nc)
    mask_coeffs : (N, 32) mask prototype weights

    Returns
    -------
    boxes, scores, classes, mask_coeffs : post-NMS arrays
    """
    # Take max class per box (Ultralytics nms.py:133)
    class_ids = scores.argmax(axis=1)
    confs = scores[np.arange(len(scores)), class_ids]

    keep = confs >= score_threshold
    boxes_f = boxes[keep]
    confs_f = confs[keep]
    class_ids_f = class_ids[keep]
    mc_f = mask_coeffs[keep]

    if len(boxes_f) == 0:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.intp),
                np.empty((0, mask_coeffs.shape[1]), dtype=np.float32))

    if len(boxes_f) > 30000:
        order = confs_f.argsort()[::-1][:30000]
        boxes_f, confs_f = boxes_f[order], confs_f[order]
        class_ids_f, mc_f = class_ids_f[order], mc_f[order]

    if class_aware:
        # Offset trick (Ultralytics nms.py:143,149)
        offsets = class_ids_f[:, None].astype(np.float32) * max_wh
        boxes_nms = boxes_f + offsets
    else:
        boxes_nms = boxes_f

    nms_idx = _greedy_nms(boxes_nms, confs_f, iou_threshold)
    nms_idx = nms_idx[:max_detections]
    return boxes_f[nms_idx], confs_f[nms_idx], class_ids_f[nms_idx], mc_f[nms_idx]


def _greedy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Standard greedy NMS. Returns indices of kept boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=np.intp)
