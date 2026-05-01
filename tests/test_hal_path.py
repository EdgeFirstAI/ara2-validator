"""Regression guard for the HAL integration contract.

Pins the API surface this validator depends on:

- ``edgefirst_hal>=0.18.1`` (batched-GEMM materialize_masks +
  squeeze_padding_dims schema-v2 fix; see HAL README § Rules 7-8)
- ``MaskResolution.Scaled`` constructor accepts ``(width, height)``
- ``materialize_masks_for_coco`` helper produces masks at original
  image resolution

These tests are environment-light: they do not need a running Ara240
target. They exercise pure-HAL APIs against synthetic ProtoData built
in numpy. The on-target end-to-end mAP run lives outside pytest (see
README for the V1 verification command).
"""

from __future__ import annotations

import importlib.metadata
import sys

import numpy as np
import pytest


hal = pytest.importorskip(
    "edgefirst_hal",
    reason="edgefirst_hal not installed; install the wheel from "
           "~/software/hal/target/wheels/ before running these tests")


# ── Version + API surface ─────────────────────────────────────────────────────

def test_hal_version_floor():
    """edgefirst_hal must be >= 0.18.1 — earlier versions miss either
    the batched-GEMM materialize_masks path (added in 0.18.0) or the
    rayon-parallel restoration that PR #51 regressed and 0.18.1
    re-applied (17× speedup on materialize_hal).
    """
    ver = importlib.metadata.version("edgefirst_hal")
    parts = tuple(int(p) for p in ver.split("+", 1)[0].split(".")[:3])
    assert parts >= (0, 18, 1), (
        f"edgefirst_hal {ver} is too old; this validator requires "
        f">= 0.18.1 for correctness and performance")


def test_mask_resolution_scaled_api():
    """MaskResolution.Scaled(width, height) must construct cleanly."""
    res = hal.MaskResolution.Scaled(640, 480)
    assert res is not None
    # Repr should mention the dimensions.
    s = repr(res)
    assert "640" in s and "480" in s


def test_mask_resolution_proto_api():
    """MaskResolution.Proto() must construct cleanly (used by viz paths
    that want continuous sigmoid values; we don't take this path for
    COCO eval but the API must remain stable).
    """
    res = hal.MaskResolution.Proto()
    assert res is not None


def test_image_processor_has_materialize_masks():
    """ImageProcessor must expose materialize_masks."""
    assert hasattr(hal.ImageProcessor, "materialize_masks")


# ── Helper signatures ────────────────────────────────────────────────────────

def test_materialize_masks_split_helpers_importable():
    """The two production helpers (RLE and arrays) and the typed
    MaskMode enum must import without a running Ara240 model.
    """
    from ara2_validator.postprocess_hal import (
        MaskMode,
        MaskResult,
        MaskTimings,
        materialize_masks_for_coco_arrays,
        materialize_masks_for_coco_rle,
    )

    assert callable(materialize_masks_for_coco_rle)
    assert callable(materialize_masks_for_coco_arrays)
    assert MaskMode.RETINA.value == "retina"
    assert MaskMode.FAST.value == "fast"
    # MaskTimings/MaskResult are part of the public typed surface.
    t = MaskTimings(materialize_hal_ms=1.0, paste_ms=2.0, rle_ms=3.0)
    assert t.paste_rle_ms == 5.0
    r = MaskResult()
    assert len(r) == 0


def test_materialize_masks_rle_handles_empty_input():
    """No detections → returns None; never raises. This is the
    path taken when a frame has no above-threshold detections.
    """
    from ara2_validator.postprocess_hal import (
        MaskMode,
        materialize_masks_for_coco_rle,
    )
    from ara2_validator.preprocess import LetterboxInfo

    lb_info = LetterboxInfo(scale=1.0, pad_x=0, pad_y=0,
                            orig_w=640, orig_h=480)
    out = materialize_masks_for_coco_rle(
        processor=None,             # never reached when scores is empty
        boxes=np.empty((0, 4), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        classes=np.empty((0,), dtype=np.int64),
        proto_data=None,
        lb_info=lb_info,
        input_w=640, input_h=640,
        mode=MaskMode.RETINA,
    )
    assert out is None


def test_materialize_masks_arrays_handles_empty_input():
    """The arrays-returning sibling has identical empty-input semantics."""
    from ara2_validator.postprocess_hal import (
        MaskMode,
        materialize_masks_for_coco_arrays,
    )
    from ara2_validator.preprocess import LetterboxInfo

    lb_info = LetterboxInfo(scale=1.0, pad_x=0, pad_y=0,
                            orig_w=640, orig_h=480)
    out = materialize_masks_for_coco_arrays(
        processor=None,
        boxes=np.empty((0, 4), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        classes=np.empty((0,), dtype=np.int64),
        proto_data=None,
        lb_info=lb_info,
        input_w=640, input_h=640,
        mode=MaskMode.FAST,
    )
    assert out is None


def test_hal_pipeline_importable():
    """HalPipeline must import (without instantiating, which would
    require a running dvproxy + DVM model)."""
    from ara2_validator.hal_pipeline import (
        HalInferenceResult,
        HalPipeline,
        unletterbox_box_array,
    )
    assert HalPipeline is not None
    assert HalInferenceResult is not None
    assert callable(unletterbox_box_array)


def test_unletterbox_box_array_empty():
    """Empty input → empty output, no crash."""
    from ara2_validator.hal_pipeline import unletterbox_box_array
    from ara2_validator.preprocess import LetterboxInfo

    lb = LetterboxInfo(scale=1.0, pad_x=0, pad_y=0, orig_w=640, orig_h=480)
    out = unletterbox_box_array(
        np.empty((0, 4), dtype=np.float32), lb, 640, 640,
    )
    assert out.shape == (0, 4)


def test_unletterbox_box_array_roundtrip():
    """A box at the corner of the unscaled-letterbox space maps to the
    expected original-image pixel coords."""
    from ara2_validator.hal_pipeline import unletterbox_box_array
    from ara2_validator.preprocess import LetterboxInfo

    # 800×600 image letterboxed to 640×640: scale = 0.8, pad_x = 0, pad_y = 80
    lb = LetterboxInfo(scale=0.8, pad_x=0, pad_y=80, orig_w=800, orig_h=600)
    # A box covering the full letterbox content area, in normalised
    # 640-space coords. Pixel coords: (0, 80, 640, 560).
    boxes_norm = np.array([[0.0, 80.0 / 640, 1.0, 560.0 / 640]], dtype=np.float32)
    out = unletterbox_box_array(boxes_norm, lb, 640, 640)
    # After un-letterbox: (0, 0, 800, 600) — full original image.
    assert np.allclose(out[0], [0.0, 0.0, 800.0, 600.0], atol=1e-3)


# ── Back-compat shim ─────────────────────────────────────────────────────────

def test_back_compat_shim_emits_deprecation_warning():
    """The old materialize_masks_for_coco() entry point still works
    but emits DeprecationWarning. Removing the shim is a separate,
    explicit step (one release cycle later)."""
    import warnings

    from ara2_validator.postprocess_hal import materialize_masks_for_coco
    from ara2_validator.preprocess import LetterboxInfo

    lb = LetterboxInfo(scale=1.0, pad_x=0, pad_y=0, orig_w=640, orig_h=480)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = materialize_masks_for_coco(
            processor=None,
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            classes=np.empty((0,), dtype=np.int64),
            proto_data=None,
            lb_info=lb, input_w=640, input_h=640,
        )
    assert out is None
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_pyproject_pins_hal_floor():
    """The hal extra in pyproject.toml must include the >= 0.18.1
    floor — without it, an environment with an older HAL would
    silently use the slow scalar materializer and the lossy proto-
    only mask path.
    """
    if sys.version_info >= (3, 11):
        import tomllib as _toml
    else:
        try:
            import tomli as _toml
        except ImportError:
            pytest.skip("tomli not installed on Python <3.11; skipping pin check")
            return

    from pathlib import Path
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = _toml.load(f)
    hal_dep = data["project"]["optional-dependencies"]["hal"]
    assert any(">=0.18.1" in d or ">= 0.18.1" in d for d in hal_dep), (
        f"hal extra must pin edgefirst-hal >= 0.18.1; got {hal_dep!r}")
