"""Hailo TAPPAS inference pipeline for benchmarking.

This module exposes :class:`HailoTappasPipeline`, the validator's
``--backend hailo`` baseline. It wraps the vendored Hailo
Application Code Examples postprocess (in
``third_party/hailo_tappas_baseline``) and exists for one purpose:
**fair head-to-head comparison against the EdgeFirst HAL pipeline**.

What is and isn't measured
==========================

The Hailo TAPPAS path is the canonical "OpenCV preprocess + Hailo
NPU + TAPPAS postprocess" reference users get out of the box from
Hailo's published examples. Crucially, the C++ shim owns the *full*
per-frame pipeline — letterbox, NPU inference, decode, NMS, mask
matmul + sigmoid + crop + resize — so the validator can compare:

* ``preprocess_ms`` — OpenCV ``cvtColor`` + aspect-preserving
  resize/pad. This is the apples-to-apples target for HAL's GPU
  letterbox: showing HAL beats OpenCV here is one of the main
  results we want from the comparison.
* ``inference_ms`` — HailoRT sync inference, single-frame, with no
  concurrent pipelining tricks. Full host-visible latency.
* ``postprocess_ms`` — TAPPAS DFL decode + NMS + per-detection
  proto×coef matmul + sigmoid + crop + resize. All host CPU.

We deliberately don't try to break out NPU vs DMA inside
``inference_ms``. HailoRT can report HW latency separately, but the
benchmark target is full-frame host wall-clock, not NPU-only.

Why no LetterboxInfo
====================

Unlike :class:`HalPipeline`, this pipeline does not surface a
:class:`LetterboxInfo`. The shim does the inverse-letterbox of
boxes and masks itself — boxes come back as normalized ``[0, 1]``
xyxy in *original-image* coordinates, masks come back at the input
image's native ``(H, W)``. The validator only needs to multiply
boxes by ``(orig_w, orig_h)`` for COCO pixel output and threshold
masks at 0.5 for COCO RLE encoding.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _import_shim():
    """Import the pybind11 backend, falling back to in-tree dev path."""
    try:
        import hailo_tappas_baseline  # type: ignore[import-not-found]
        return hailo_tappas_baseline, None
    except ImportError:
        pass

    here = Path(__file__).resolve().parent
    pybind_dir = (
        here.parent / "third_party" / "hailo_tappas_baseline" / "pybind11"
    )
    if pybind_dir.exists() and str(pybind_dir) not in sys.path:
        sys.path.insert(0, str(pybind_dir))

    try:
        import hailo_tappas_baseline  # type: ignore[import-not-found]
        return hailo_tappas_baseline, None
    except ImportError as exc:
        return None, str(exc)


_shim, _import_error = _import_shim()
log = logging.getLogger(__name__)


@dataclass
class HailoInferenceResult:
    """One image's worth of Hailo TAPPAS output, plus per-stage timings.

    Attributes
    ----------
    boxes : np.ndarray
        ``(N, 4)`` xyxy normalized to ``[0, 1]`` in *original-image*
        coordinates. The shim handles inverse-letterbox internally.
    scores : np.ndarray
        ``(N,)`` float32.
    classes : np.ndarray
        ``(N,)`` int32, 0-indexed COCO class IDs.
    masks_float : list[np.ndarray]
        Per-detection ``(orig_h, orig_w)`` float32 masks, post-sigmoid,
        bbox-cropped. Caller thresholds at 0.5 for binary uint8.
    timings : dict[str, float]
        Per-stage milliseconds. Keys: ``decode`` (cv2.imread),
        ``preprocess`` (BGR→RGB + letterbox), ``inference`` (HailoRT
        sync), ``postprocess`` (TAPPAS decode + NMS + masks).
    """

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    masks_float: list
    timings: dict


class HailoTappasPipeline:
    """One-stop Hailo TAPPAS pipeline for sync, full-frame benchmarking.

    Owns a single :class:`HailoTappasBackend` (= ``VDevice`` +
    ``InferModel`` + ``ConfiguredInferModel``) for the lifetime of a
    run. The shim does its own configure-once work; the only state
    this Python class adds is per-frame timing aggregation.

    Parameters
    ----------
    hef_path : str
        Path to a compiled HEF (Hailo's reference or one produced by
        hailo-converter v0.3.0+).
    score_threshold, iou_threshold : float
        **Currently informational only** — both are baked into the
        vendored postprocess source at compile time
        (``SCORE_THRESHOLD = 0.6f``, ``IOU_THRESHOLD = 0.7f`` in
        ``instance_seg_postprocess.cpp``). Passing different values
        here logs a warning so we don't silently mislead the user;
        making these runtime-tunable is a follow-up tracked in
        ``third_party/hailo_tappas_baseline/pybind11/README.md``.
    max_detections : int
        Currently unused — the upstream postprocess does not cap
        detection count beyond NMS itself.
    """

    DEFAULT_SCORE_THRESHOLD = 0.6
    DEFAULT_IOU_THRESHOLD = 0.7

    def __init__(
        self,
        hef_path: str,
        *,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_detections: int = 300,
    ):
        if _shim is None:
            raise ImportError(
                "hailo_tappas_baseline native module is not importable: "
                f"{_import_error}\n"
                "Build it first on a host with HailoRT installed:\n"
                "  cd third_party/hailo_tappas_baseline/pybind11\n"
                "  cmake -S . -B build && cmake --build build -j\n"
                "  cmake --install build"
            )

        self.backend = _shim.HailoTappasBackend(hef_path)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.input_w = int(self.backend.model_width)
        self.input_h = int(self.backend.model_height)

        if (score_threshold != self.DEFAULT_SCORE_THRESHOLD
                or iou_threshold != self.DEFAULT_IOU_THRESHOLD):
            log.warning(
                "Hailo TAPPAS thresholds are baked at the C++ level "
                "(SCORE=%.3f, IOU=%.3f). Requested score=%.3f / iou=%.3f "
                "are ignored until upstream is patched to read them at "
                "runtime.",
                self.DEFAULT_SCORE_THRESHOLD, self.DEFAULT_IOU_THRESHOLD,
                score_threshold, iou_threshold,
            )

    def warmup(self, image_paths, n: int = 3) -> None:
        """Run ``n`` warmup iterations to settle thread pools and caches."""
        for image_path in image_paths[: max(0, n)]:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                continue
            self.backend.infer(bgr)

    def infer_one(self, image_path) -> HailoInferenceResult:
        """Run the full Hailo TAPPAS pipeline on one image.

        Pipeline stages (each timed):

        1. **decode** — ``cv2.imread`` JPEG → BGR ndarray.
        2. **preprocess** — BGR→RGB ``cvtColor`` + aspect-preserving
           letterbox (``cv::INTER_AREA`` resize + bottom/right zero-pad).
        3. **inference** — HailoRT sync inference, single bindings,
           ``run_async`` + ``wait()``.
        4. **postprocess** — TAPPAS ``filter()``: per-scale dequant +
           DFL decode + NMS + per-detection ``coef × proto`` matmul +
           sigmoid + crop + resize to original-image size.

        Boxes and masks are returned in original-image coordinates;
        no un-letterbox step is needed at the validator level.
        """
        timings: dict = {}

        t0 = time.perf_counter()
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        timings["decode"] = (time.perf_counter() - t0) * 1e3

        # The shim handles preprocess + inference + postprocess and
        # reports each separately in milliseconds. We do NOT overlay
        # a host wall-clock measurement here: the shim's clock is
        # ``std::chrono::steady_clock`` (same monotonic source) but
        # excludes the GIL acquire/release boundary cost, which a
        # naive Python ``time.perf_counter`` around ``backend.infer()``
        # would lump into ``inference_ms``.
        result = self.backend.infer(bgr)
        timings["preprocess"] = float(result["timings"]["preprocess_ms"])
        timings["inference"] = float(result["timings"]["inference_ms"])
        timings["postprocess"] = float(result["timings"]["postprocess_ms"])

        return HailoInferenceResult(
            boxes=np.asarray(result["boxes"], dtype=np.float32),
            scores=np.asarray(result["scores"], dtype=np.float32),
            classes=np.asarray(result["classes"], dtype=np.int32),
            masks_float=list(result["masks"]),
            timings=timings,
        )


def boxes_normalized_to_pixels(
    boxes: np.ndarray, orig_w: int, orig_h: int,
) -> np.ndarray:
    """Convert ``(N, 4)`` xyxy normalized ``[0, 1]`` to pixel coords.

    The Hailo backend emits boxes already un-letterboxed into the
    original-image coordinate system but normalized; multiplying by
    ``(orig_w, orig_h)`` gets COCO-ready pixel xyxy.
    """
    if len(boxes) == 0:
        return boxes.astype(np.float32)
    bx = np.asarray(boxes, dtype=np.float32).copy()
    bx[:, [0, 2]] *= orig_w
    bx[:, [1, 3]] *= orig_h
    return bx


def threshold_masks_to_uint8(
    masks_float, threshold: float = 0.5,
) -> list[np.ndarray]:
    """Threshold post-sigmoid float32 masks at 0.5 → uint8 binary.

    Used to feed :func:`coco_output.encode_mask_rle`. The masks
    produced by the shim are already bbox-cropped (pixels outside
    the detection's bounding box are zero) and post-sigmoid, so
    a single threshold is the correct binarization step.
    """
    return [(m > threshold).astype(np.uint8) for m in masks_float]
