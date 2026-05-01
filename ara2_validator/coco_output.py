"""COCO JSON annotation serialization.

Converts detection + segmentation results into the COCO results format
suitable for evaluation with pycocotools.COCOeval.

Output format (per detection):
    {
        "image_id": int,
        "category_id": int,      # COCO 91-category ID (not 0-indexed 80)
        "bbox": [x, y, w, h],   # original image pixel coordinates
        "score": float,
        "segmentation": {...}    # RLE-encoded binary mask (optional)
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# COCO uses 91 category IDs (1-90 with gaps) for its 80 detection classes.
# This maps the 0-indexed model output class index → COCO category_id.
COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def image_id_from_filename(filename: str) -> int:
    """Extract COCO image_id from filename.

    COCO image filenames are zero-padded 12-digit numbers:
    000000000009.jpg → 9

    Falls back to hashing the stem if the filename isn't numeric.
    """
    stem = Path(filename).stem
    try:
        return int(stem)
    except ValueError:
        return hash(stem) % (2**31)


def encode_mask_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask to COCO RLE format using pycocotools.

    Parameters
    ----------
    mask : (H, W) uint8 binary mask

    Returns
    -------
    dict with "size" and "counts" keys (COCO RLE format)
    """
    from pycocotools import mask as mask_utils
    # pycocotools expects Fortran-order uint8 array
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # counts is bytes in Python 3, convert to string for JSON
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_coco_results(
    image_path: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    masks: list[np.ndarray] | None = None,
) -> list[dict]:
    """Build COCO results entries for one image.

    Parameters
    ----------
    image_path : str
        Image filename (for image_id extraction).
    boxes : (M, 4) xyxy in original image pixel coordinates
    scores : (M,)
    classes : (M,) 0-indexed class indices
    masks : list of (H, W) binary masks, **or** list of pre-encoded
        RLE dicts (e.g. returned by ``materialize_masks_for_coco`` with
        ``return_rle=True``), or ``None``.

    Returns
    -------
    list of COCO result dicts
    """
    image_id = image_id_from_filename(image_path)
    results = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        w = float(x2 - x1)
        h = float(y2 - y1)
        if w <= 0 or h <= 0:
            continue

        cls_idx = int(classes[i])
        # Map to COCO 91-category ID
        if cls_idx < len(COCO80_TO_COCO91):
            cat_id = COCO80_TO_COCO91[cls_idx]
        else:
            cat_id = cls_idx + 1  # fallback

        entry = {
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [round(float(x1), 2), round(float(y1), 2),
                     round(w, 2), round(h, 2)],
            "score": round(float(scores[i]), 6),
        }

        if masks is not None and i < len(masks):
            m = masks[i]
            # Accept either a pre-encoded RLE dict (fast path — caller
            # has already encoded inline using a reused canvas) or a
            # binary mask array (legacy path).
            entry["segmentation"] = m if isinstance(m, dict) else encode_mask_rle(m)

        results.append(entry)

    return results


def save_coco_results(results: list[dict], output_path: str) -> None:
    """Write COCO results JSON to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} detections to {output_path}")
