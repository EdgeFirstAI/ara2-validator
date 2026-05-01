"""Deployment-style HAL benchmark using ImageProcessor.draw_masks.

The validator entry points (`edgefirst.py`, `__main__.py`) measure
*validation* mask cost — they call `materialize_masks_for_coco_rle`
which produces per-detection numpy mask arrays for pycocotools RLE
encoding. That is the right thing for COCO scoring but it is **not**
what a live deployment does.

A real deployment composites masks directly onto a display canvas via
HAL's fused ``ImageProcessor.draw_masks``: decode → NMS → mask
materialise → blend onto the destination tensor, all inside Rust/GPU,
with no per-detection numpy round-trip. The reference live pipeline is
``ara2-rs/examples/yolov8_live.py``; this script is a stripped-down,
camera-free version that runs the same sequence on a directory of
images so we can capture per-stage timings.

Usage
-----

::

    python tests/bench_draw_masks.py \\
        --model yolov8n-seg-kinara-1.2.1.dvm \\
        --images /root/coco128/images/

Reports min / mean / max for: image decode, GPU letterbox, NPU
inference, ``draw_masks`` (fused decode+render), and end-to-end.

See HAL's ``ImageProcessor.draw_masks`` and ``draw_decoded_masks`` API
docs for the full surface, and the ``EdgeFirst for i.MX`` white-paper
manual for the canonical live-pipeline timings.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import edgefirst_hal as hal

# Reuse the validator's HAL pipeline so the preprocess + decoder setup
# matches edgefirst.py exactly. This keeps preprocess + inference timings
# directly comparable between the two scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ara2_validator._stats import fmt_row, summary_header, trimmed_stats
from ara2_validator.hal_pipeline import HalPipeline
from ara2_validator.model import load_model

log = logging.getLogger("ara2_validator.bench_draw_masks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to .dvm model")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Deployment NMS score threshold (default 0.5)")
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100,
                        help="Deployment max_det (default 100)")
    parser.add_argument("--socket", default="/var/run/ara2.sock")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(message)s", stream=sys.stderr)

    img_dir = Path(args.images)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts
    )
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        log.error("no images in %s", args.images)
        return 1

    model, spec = load_model(args.model, socket_path=args.socket)
    pipeline = HalPipeline(
        model, spec,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )
    log.info("Model:   %s", args.model)
    log.info("Images:  %d from %s", len(image_paths), args.images)
    log.info("Threshold: %.3f (deployment-style)", args.score_threshold)

    # Per-frame canvas the same size as the original image. In a real
    # deployment this is the display compositor's surface (Wayland
    # DMA-BUF, KMS framebuffer, or similar) — there is one canvas, and
    # draw_masks writes it in place each frame. Here we recreate it per
    # frame so we don't need to track per-image dimensions; in a fixed-
    # resolution camera pipeline the canvas is created once.
    processor = pipeline.preprocessor.processor

    if args.warmup > 0:
        log.info("Warmup: %d iterations", args.warmup)
        pipeline.warmup(image_paths, n=args.warmup)

    timings: dict[str, list[float]] = {
        "decode": [], "preprocess": [], "inference": [], "draw_masks": [],
        "end_to_end": [], "total": [],
        "dma_input": [], "npu_compute": [], "dma_output": [],
    }

    for idx, img_path in enumerate(image_paths):
        t_total = time.perf_counter()

        # 1. Image decode (JPEG → DMA-backed HAL tensor)
        t0 = time.perf_counter()
        src, w, h = pipeline.preprocessor.decode_image(img_path)
        timings["decode"].append((time.perf_counter() - t0) * 1e3)

        # 2. GPU letterbox preprocess (writes NPU input DMA-BUF)
        t0 = time.perf_counter()
        lb_info = pipeline.preprocessor.preprocess(src, w, h)
        timings["preprocess"].append((time.perf_counter() - t0) * 1e3)

        # 3. NPU inference
        t0 = time.perf_counter()
        timing = pipeline.model.run()
        timings["inference"].append((time.perf_counter() - t0) * 1e3)
        timings["dma_input"].append(timing.input_time_us / 1000.0)
        timings["npu_compute"].append(timing.run_time_us / 1000.0)
        timings["dma_output"].append(timing.output_time_us / 1000.0)

        # 4. Fused decode + materialize + draw, in one HAL call.
        #
        # `draw_masks` reads the NPU output DMA-BUF directly via the
        # output tensors registered on HalPipeline, runs the decoder,
        # materialises masks on the GPU, and composites them onto
        # `canvas` in place. Returns (boxes, scores, classes) but
        # *not* the per-detection masks — masks live on the GPU and
        # never round-trip into Python. This is the path described
        # in HAL's `ImageProcessor.draw_masks` API and benchmarked in
        # the EdgeFirst for i.MX manual.
        canvas = processor.create_image(lb_info.orig_w, lb_info.orig_h,
                                        hal.PixelFormat.Rgba)
        letterbox_norm = (
            lb_info.pad_x / pipeline.input_w,
            lb_info.pad_y / pipeline.input_h,
            (lb_info.pad_x + int(lb_info.orig_w * lb_info.scale)) / pipeline.input_w,
            (lb_info.pad_y + int(lb_info.orig_h * lb_info.scale)) / pipeline.input_h,
        )
        t0 = time.perf_counter()
        result = processor.draw_masks(
            decoder=pipeline.decoder,
            model_output=pipeline.output_tensors,
            dst=canvas,
            background=src,
            letterbox=letterbox_norm,
        )
        timings["draw_masks"].append((time.perf_counter() - t0) * 1e3)

        # End-to-end = preprocess + inference + draw_masks. Image
        # decode is excluded (camera supplies the frame in deployment).
        timings["end_to_end"].append(
            timings["preprocess"][-1]
            + timings["inference"][-1]
            + timings["draw_masks"][-1]
        )
        timings["total"].append((time.perf_counter() - t_total) * 1e3)

        # `result` is (boxes, scores, classes) — number of dets only.
        n_dets = len(result[1]) if isinstance(result, tuple) else len(result.scores)

        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(image_paths) - 1:
            avg = sum(timings["end_to_end"][-10:]) / min(10, len(timings["end_to_end"]))
            log.info(
                "  [%4d/%d] %20s  %3d dets  %6.1f ms e2e",
                idx + 1, len(image_paths), Path(img_path).name, n_dets, avg,
            )

    # ── Timing summary ───────────────────────────────────────────────
    print()
    hdr = summary_header()
    print(hdr)
    print("─" * len(hdr))
    print(fmt_row("image decode (excluded)", trimmed_stats(timings["decode"])))
    print(fmt_row("preprocess (GPU)", trimmed_stats(timings["preprocess"])))
    print(fmt_row("inference (wall)", trimmed_stats(timings["inference"])))
    print(fmt_row("dma input (host→device)",
                  trimmed_stats(timings["dma_input"]), indent=1))
    print(fmt_row("npu compute",
                  trimmed_stats(timings["npu_compute"]), indent=1))
    print(fmt_row("dma output (device→host)",
                  trimmed_stats(timings["dma_output"]), indent=1))
    print(fmt_row("draw_masks (decode+render)",
                  trimmed_stats(timings["draw_masks"])))
    print("─" * len(hdr))
    e2e = trimmed_stats(timings["end_to_end"])
    print(fmt_row("END-TO-END (pre+inf+draw)", e2e))
    print(
        f"  Outlier filter: 1% trimmed "
        f"({e2e['n']}/{len(timings['end_to_end'])} samples)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
