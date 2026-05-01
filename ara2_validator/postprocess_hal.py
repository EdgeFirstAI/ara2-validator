"""HAL-accelerated postprocessing for YOLO segmentation outputs.

Public entry point: :func:`materialize_masks_for_coco_rle` (and its
``return_arrays`` sibling) — the recommended pattern for turning a
HAL :class:`Decoder.decode_proto` result into binary masks at original
image resolution, ready for COCO RLE evaluation.

The helpers below are written so a customer adopting EdgeFirst HAL
can copy them verbatim into their own pipeline. Specifically, they:

* Always pass a normalised ``letterbox`` tuple to
  :func:`hal.ImageProcessor.materialize_masks` (otherwise HAL would
  try to threshold the gray padding bars, producing spurious mask
  pixels — the most common HAL integration mistake).
* Reuse a single ``(orig_h, orig_w)`` canvas across detections in the
  paste loop, encoding-and-discarding the binary mask inline as
  pycocotools RLE. This eliminates a per-detection ~1 MB allocation
  on high-resolution images.
* Split the per-stage timings into ``materialize_hal_ms`` (HAL call),
  ``paste_ms`` (canvas paste), and ``rle_ms`` (RLE encode) so the
  caller can attribute regressions between HAL and caller-side code.

Requires ``edgefirst-hal>=0.18.1`` for the batched-GEMM
``materialize_masks`` path with rayon parallelism, the
``squeeze_padding_dims`` schema-v2 fix, and the per-detection logit-
precompute path. See HAL ``README § Optimization Guide`` Rule 8.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .preprocess import LetterboxInfo

try:
    import edgefirst_hal as hal
except ImportError:
    hal = None  # type: ignore


class MaskMode(Enum):
    """Mask materialisation strategy.

    Two modes ship today; an enum (rather than a boolean flag) keeps
    the door open for ``MaskResolution::ScaledNonRetina`` or other
    HAL-side modes without a breaking API change.

    Attributes
    ----------
    RETINA
        :class:`hal.MaskResolution.Scaled` — bilinear-upsamples the
        f32 logit plane to original-image resolution then thresholds
        at ``logit > 0`` (equivalent to ``sigmoid > 0.5``). Highest
        accuracy. Matches Ultralytics ``ops.process_mask_native()``.
        This is the default and the recommended choice for COCO
        evaluation.
    FAST
        :class:`hal.MaskResolution.Proto` — HAL returns continuous-
        sigmoid u8 tiles at proto resolution (160×160) and the caller
        upsamples each tile via ``cv2.resize(INTER_LINEAR)``.
        About 1.5× faster end-to-end on imx8mp-frdm at the cost of
        roughly 0.8 pp mask mAP. Use when latency matters more than
        the last fraction of accuracy.
    """

    RETINA = "retina"
    FAST = "fast"


@dataclass
class MaskTimings:
    """Per-stage timings (ms) for one image's mask materialisation.

    All three keys are always populated; ``paste`` and ``rle`` are
    reported separately so a caller can isolate HAL-side work
    (``materialize_hal``) from output-formatting work (``paste`` +
    ``rle``).
    """

    materialize_hal_ms: float = 0.0
    paste_ms: float = 0.0
    rle_ms: float = 0.0

    @property
    def paste_rle_ms(self) -> float:
        """Convenience: ``paste_ms + rle_ms``."""
        return self.paste_ms + self.rle_ms


@dataclass
class MaskResult:
    """Result of a mask-materialisation call.

    Holds either RLE dicts (for COCO output) or binary numpy arrays
    (for visualisation / debugging) — never both. The two production
    paths (:func:`materialize_masks_for_coco_rle` and
    :func:`materialize_masks_for_coco_arrays`) populate the matching
    field and leave the other ``None``.
    """

    rles: list[dict] | None = None
    """Per-detection pycocotools RLE dicts, aligned with the input
    ``boxes``. ``None`` from the arrays-returning entry point."""

    arrays: list[np.ndarray] | None = None
    """Per-detection ``(orig_h, orig_w)`` uint8 binary masks.
    ``None`` from the RLE-returning entry point."""

    timings: MaskTimings = field(default_factory=MaskTimings)
    """Per-stage timings for this image's mask path."""

    def __len__(self) -> int:
        return len(self.rles or self.arrays or [])


def _materialize_tiles(
    processor,
    boxes,
    scores,
    classes,
    proto_data,
    lb_info: LetterboxInfo,
    input_w: int,
    input_h: int,
    mode: MaskMode,
) -> tuple[list, float, np.ndarray]:
    """Run the HAL ``materialize_masks`` call and return ``(tiles, hal_ms, bx)``.

    Centralises the two-mode dispatch (Scaled vs Proto), the
    ``letterbox_norm`` computation, and the contract smoke-check.
    Used by both public entry points.
    """
    if hal is None:
        raise ImportError(
            "edgefirst_hal is required for materialize_masks_for_coco. "
            "Install with: pip install 'edgefirst-hal>=0.18.1'"
        )

    # Letterbox region in normalised model-input coords.
    #
    # The proto plane covers the full 640×640 model-input space
    # *including* the gray padding bars. HAL needs to know which
    # portion is actual image content so it doesn't materialise masks
    # over the padding. Without this, the bilinear upsample produces
    # spurious mask pixels at image edges that bleed into bbox
    # boundaries — this is the single most common mistake when first
    # using `materialize_masks`.
    letterbox_norm = (
        lb_info.pad_x / input_w,
        lb_info.pad_y / input_h,
        (lb_info.pad_x + int(lb_info.orig_w * lb_info.scale)) / input_w,
        (lb_info.pad_y + int(lb_info.orig_h * lb_info.scale)) / input_h,
    )

    if mode is MaskMode.FAST:
        # FAST: HAL returns per-detection bbox-cropped continuous-
        # sigmoid u8 tiles at proto resolution; caller upsamples each
        # tile via cv2.resize. Letterbox is intentionally None — the
        # caller-side bbox un-letterbox positions tiles in original-
        # image pixel coords directly.
        resolution = hal.MaskResolution.Proto()
        resolved_letterbox = None
    else:
        # RETINA (default): HAL upsamples the f32 logit plane to
        # original-image resolution then thresholds. Highest accuracy
        # path — matches Ultralytics process_mask_native.
        resolution = hal.MaskResolution.Scaled(lb_info.orig_w, lb_info.orig_h)
        resolved_letterbox = letterbox_norm

    t0 = _time.perf_counter_ns()
    tiles = processor.materialize_masks(
        boxes, scores, classes, proto_data,
        letterbox=resolved_letterbox,
        resolution=resolution,
    )
    hal_ms = (_time.perf_counter_ns() - t0) / 1e6

    # Smoke-check the HAL contract: every tile is (h, w, 1) uint8
    # with values already binary {0, 255}. Catches schema-v2 silent
    # shape truncations and dtype regressions immediately.
    if tiles:
        first = tiles[0]
        if first.ndim != 3 or first.shape[2] != 1 or first.dtype != np.uint8:
            raise RuntimeError(
                f"materialize_masks contract violation: first tile "
                f"shape={first.shape} dtype={first.dtype}; "
                f"expected (h, w, 1) uint8"
            )

    # Un-letterbox boxes to original-image pixel coords for paste.
    # The copy is necessary because the in-place clamp below would
    # otherwise mutate the caller's `boxes` array (HAL returns a view
    # over its internal buffer, not a fresh allocation).
    bx = np.asarray(boxes, dtype=np.float32).copy()
    if len(bx) > 0:
        bx[:, [0, 2]] = (bx[:, [0, 2]] * input_w - lb_info.pad_x) / lb_info.scale
        bx[:, [1, 3]] = (bx[:, [1, 3]] * input_h - lb_info.pad_y) / lb_info.scale
        bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], 0, lb_info.orig_w)
        bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], 0, lb_info.orig_h)

    return tiles, hal_ms, bx


def _paste_one(
    canvas: np.ndarray,
    tile: np.ndarray,
    box: np.ndarray,
    lb_info: LetterboxInfo,
    mode: MaskMode,
) -> None:
    """Zero ``canvas`` and paste one detection's mask tile into it.

    Mode-aware: in :attr:`MaskMode.RETINA` the tile is already at
    original-image resolution and binary; in :attr:`MaskMode.FAST` the
    tile is at proto resolution with continuous-sigmoid u8 values and
    needs cv2 upsampling.
    """
    canvas.fill(0)
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1 = max(x1, 0); y1 = max(y1, 0)
    x2 = min(x2, lb_info.orig_w); y2 = min(y2, lb_info.orig_h)
    if x2 <= x1 or y2 <= y1:
        return
    t = tile[:, :, 0]
    if mode is MaskMode.FAST:
        # Cast to f32 to preserve precision through the bilinear
        # upsample (cv2 with u8 input does integer math; with f32 it
        # does float math, which preserves the continuous sigmoid
        # intermediate). Threshold at 127.5 = sigmoid 0.5.
        target_w = x2 - x1
        target_h = y2 - y1
        if t.size > 0 and target_w > 0 and target_h > 0:
            upsampled = cv2.resize(
                t.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
            canvas[y1:y2, x1:x2] = (upsampled > 127.5).astype(np.uint8)
    else:
        # RETINA: tile is already binary {0, 255} at orig resolution
        # sized to the bbox region. Direct paste + > 127 threshold.
        th, tw = t.shape[:2]
        sh = min(th, y2 - y1); sw = min(tw, x2 - x1)
        canvas[y1:y1+sh, x1:x1+sw] = (t[:sh, :sw] > 127).astype(np.uint8)


def materialize_masks_for_coco_rle(
    processor,
    boxes,
    scores,
    classes,
    proto_data,
    lb_info: LetterboxInfo,
    input_w: int,
    input_h: int,
    *,
    mode: MaskMode = MaskMode.RETINA,
) -> MaskResult | None:
    """Materialise per-detection masks and RLE-encode them inline.

    The COCO-eval path. Encodes each binary mask as a pycocotools RLE
    dict immediately after pasting and reuses a single
    ``(orig_h, orig_w)`` canvas across detections — this eliminates a
    per-detection ~1 MB allocation on high-resolution images and
    matches HAL Optimization Guide Rule 1 (reuse buffers in hot paths).

    Returns
    -------
    MaskResult | None
        :attr:`MaskResult.rles` populated; :attr:`MaskResult.arrays`
        is ``None``. Returns ``None`` when there are no detections.

    Notes
    -----
    The returned RLE dicts can be passed directly to
    :func:`build_coco_results` (in ``coco_output.py``), which detects
    the dict form and skips the redundant encode step.
    """
    if proto_data is None or len(scores) == 0:
        return None

    # Local import keeps the module load lean; called once per image.
    from .coco_output import encode_mask_rle

    tiles, hal_ms, bx = _materialize_tiles(
        processor, boxes, scores, classes, proto_data,
        lb_info, input_w, input_h, mode,
    )
    timings = MaskTimings(materialize_hal_ms=hal_ms)
    if not tiles:
        return MaskResult(rles=[], timings=timings)

    # Allocate the reuse-canvas ONCE — this is the cornerstone of the
    # B1 (canvas reuse) optimisation. fill(0) per detection costs
    # microseconds; np.zeros((1080,1920)) per detection on a 1080p
    # image would cost ~1 ms × N detections.
    canvas = np.zeros((lb_info.orig_h, lb_info.orig_w), dtype=np.uint8)
    rles: list[dict] = []
    paste_ns = 0
    rle_ns = 0
    for tile, box in zip(tiles, bx):
        t_paste = _time.perf_counter_ns()
        _paste_one(canvas, tile, box, lb_info, mode)
        t_rle = _time.perf_counter_ns()
        paste_ns += t_rle - t_paste
        rles.append(encode_mask_rle(canvas))
        rle_ns += _time.perf_counter_ns() - t_rle
    timings.paste_ms = paste_ns / 1e6
    timings.rle_ms = rle_ns / 1e6
    return MaskResult(rles=rles, timings=timings)


def materialize_masks_for_coco_arrays(
    processor,
    boxes,
    scores,
    classes,
    proto_data,
    lb_info: LetterboxInfo,
    input_w: int,
    input_h: int,
    *,
    mode: MaskMode = MaskMode.RETINA,
) -> MaskResult | None:
    """Materialise per-detection masks and return them as binary arrays.

    Use this for visualisation, debugging, or when downstream code
    needs the binary mask in numpy form. For COCO-eval workflows
    prefer :func:`materialize_masks_for_coco_rle` (encodes inline,
    reuses the canvas).

    Returns
    -------
    MaskResult | None
        :attr:`MaskResult.arrays` populated with one fresh
        ``(orig_h, orig_w)`` uint8 binary mask per detection;
        :attr:`MaskResult.rles` is ``None``.
    """
    if proto_data is None or len(scores) == 0:
        return None

    tiles, hal_ms, bx = _materialize_tiles(
        processor, boxes, scores, classes, proto_data,
        lb_info, input_w, input_h, mode,
    )
    timings = MaskTimings(materialize_hal_ms=hal_ms)
    if not tiles:
        return MaskResult(arrays=[], timings=timings)

    arrays: list[np.ndarray] = []
    paste_ns = 0
    for tile, box in zip(tiles, bx):
        t_paste = _time.perf_counter_ns()
        canvas = np.zeros((lb_info.orig_h, lb_info.orig_w), dtype=np.uint8)
        _paste_one(canvas, tile, box, lb_info, mode)
        paste_ns += _time.perf_counter_ns() - t_paste
        arrays.append(canvas)
    timings.paste_ms = paste_ns / 1e6
    # No RLE in this path — leave rle_ms = 0.0.
    return MaskResult(arrays=arrays, timings=timings)


# ────────────────────────────────────────────────────────────────────
# Back-compat shim
# ────────────────────────────────────────────────────────────────────
#
# The earlier API (`materialize_masks_for_coco(..., fast_masks=...,
# return_rle=..., timings=dict_or_none)`) has been retired in favour
# of the typed pair above. This shim is kept for one release cycle to
# avoid breaking callers that have not yet migrated.

def materialize_masks_for_coco(  # pragma: no cover — back-compat shim
    processor,
    boxes,
    scores,
    classes,
    proto_data,
    lb_info: LetterboxInfo,
    input_w: int,
    input_h: int,
    *,
    fast_masks: bool = False,
    return_rle: bool = True,
    timings: dict | None = None,
):
    """DEPRECATED — use :func:`materialize_masks_for_coco_rle` /
    :func:`materialize_masks_for_coco_arrays` instead.

    This shim maps ``fast_masks=True`` → :attr:`MaskMode.FAST`,
    ``return_rle=True`` → ``materialize_masks_for_coco_rle``, and
    populates the legacy ``timings`` dict if one is supplied. Will
    be removed in a future release.
    """
    import warnings
    warnings.warn(
        "materialize_masks_for_coco() is deprecated; use "
        "materialize_masks_for_coco_rle() or "
        "materialize_masks_for_coco_arrays() with mode=MaskMode.RETINA / FAST.",
        DeprecationWarning,
        stacklevel=2,
    )
    mode = MaskMode.FAST if fast_masks else MaskMode.RETINA
    fn = (
        materialize_masks_for_coco_rle
        if return_rle
        else materialize_masks_for_coco_arrays
    )
    result = fn(
        processor, boxes, scores, classes, proto_data,
        lb_info, input_w, input_h, mode=mode,
    )
    if timings is not None and result is not None:
        timings["materialize_hal_ms"] = result.timings.materialize_hal_ms
        timings["paste_ms"] = result.timings.paste_ms
        timings["rle_ms"] = result.timings.rle_ms
        timings["paste_rle_ms"] = result.timings.paste_rle_ms
    if result is None:
        return None
    return result.rles if return_rle else result.arrays
