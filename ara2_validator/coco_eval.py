"""COCO evaluation using pycocotools.

Runs COCOeval on a results JSON (predictions) against a COCO GT JSON
(ground truth) and prints the standard COCO AP/AR metrics table.

Supports both bbox and segmentation (segm) evaluation types.

Usage as a module:
    from ara2_validator.coco_eval import evaluate_coco
    evaluate_coco("instances_train2017.json", "results.json", iou_type="bbox")

Usage from the command line:
    python -m ara2_validator.coco_eval \\
        --gt annotations/instances_train2017.json \\
        --results results_reference.json \\
        --type bbox
"""

from __future__ import annotations

import argparse
import json
import sys


def evaluate_coco(
    gt_path: str,
    results_path: str,
    iou_type: str = "bbox",
) -> dict:
    """Run COCO evaluation and return metrics.

    Parameters
    ----------
    gt_path : str
        Path to COCO ground truth JSON (instances_*.json format).
    results_path : str
        Path to COCO results JSON (list of detection dicts).
    iou_type : str
        "bbox" for bounding box evaluation, "segm" for instance segmentation.

    Returns
    -------
    dict with keys: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("Error: pycocotools required. Install with: pip install pycocotools",
              file=sys.stderr)
        sys.exit(1)

    # Load GT
    coco_gt = COCO(gt_path)

    # Load results
    with open(results_path) as f:
        results = json.load(f)

    if not results:
        print("Warning: no detections in results file.", file=sys.stderr)
        return {}

    # Ensure all result image_ids exist in GT (COCOeval silently skips mismatches)
    gt_img_ids = set(coco_gt.getImgIds())
    res_img_ids = set(r["image_id"] for r in results)
    matched = gt_img_ids & res_img_ids
    missing = res_img_ids - gt_img_ids
    if missing:
        print(f"Warning: {len(missing)} result image_ids not in GT "
              f"(will be ignored by COCOeval)", file=sys.stderr)
    print(f"Evaluating {len(results)} detections across "
          f"{len(matched)} images (type={iou_type})")

    # Load results into COCO format
    coco_dt = coco_gt.loadRes(results_path)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics from the standard 12-element stats array
    # See pycocotools/cocoeval.py for the order
    stats = coco_eval.stats
    metric_names = [
        "AP", "AP50", "AP75", "APs", "APm", "APl",
        "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    ]
    metrics = {name: float(val) for name, val in zip(metric_names, stats)}
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="COCO evaluation (pycocotools)")
    parser.add_argument("--gt", required=True,
                        help="Path to COCO ground truth JSON")
    parser.add_argument("--results", required=True,
                        help="Path to COCO results JSON")
    parser.add_argument("--type", default="bbox", choices=["bbox", "segm"],
                        help="Evaluation type (default: bbox)")
    args = parser.parse_args()

    metrics = evaluate_coco(args.gt, args.results, iou_type=args.type)

    if metrics:
        print("\n── Summary ──")
        print(f"  mAP@0.50:0.95 = {metrics['AP']:.4f}")
        print(f"  mAP@0.50      = {metrics['AP50']:.4f}")
        print(f"  mAR@100       = {metrics['AR100']:.4f}")


if __name__ == "__main__":
    main()
