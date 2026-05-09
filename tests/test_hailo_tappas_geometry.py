"""Geometry-equivalence test for the Hailo TAPPAS shim's letterbox path.

Asserts the shim's pre-inference letterbox canvas matches
ara2_validator.preprocess.letterbox() byte-for-byte (or within 1 LSB) on
five image sizes that together exercise width-bound, height-bound, even
pad, odd pad, downscale, and upscale.

Skipped on hosts where the shim isn't built (e.g. host CI without HailoRT).

Three legitimate sources of <=1 LSB drift between the shim's C++ resize
and the Python-side cv2.resize:

1. Rounding mode at the half-integer boundary. C++ std::round is
   half-away-from-zero; Python round is banker's. For
   src_w * scale = 240.5 exactly, C++ -> 241, Python -> 240.
2. Float vs double precision in `scale`. The shim uses static_cast<float>
   (single precision) to compute the scale; Python uses double throughout.
3. OpenCV link-time skew. The shim links the system C++ OpenCV; Python
   uses the bundled opencv-python wheel. INTER_LINEAR rounding may
   differ across versions.

Anything beyond <=1 LSB indicates a real bug.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

try:
    import hailo_tappas_baseline as _shim
    _shim_available = True
except ImportError:
    _shim = None
    _shim_available = False

from ara2_validator.preprocess import letterbox as py_letterbox


# (H, W) sizes covering width-bound, height-bound, even pad, odd pad,
# downscale, and upscale. See spec Section 5.1 for the rationale.
SIZES = [
    (640, 481),    # H=640, W=481 -> r=1.0 (no resize), W-pad=159 odd
    (481, 640),    # H=481, W=640 -> r=1.0 (no resize), H-pad=159 odd
    (1280, 720),   # downscale, r=0.5, even pad on both axes
    (240, 320),    # scaleup required, both dims < 640
    (321, 241),    # scaleup required, odd pad
]


def _make_image(h: int, w: int) -> np.ndarray:
    """Solid BGR colour (50, 100, 200) so RGB conversion can be verified."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[..., 0] = 50   # B
    img[..., 1] = 100  # G
    img[..., 2] = 200  # R
    return img


@pytest.mark.skipif(
    not _shim_available,
    reason="hailo_tappas_baseline shim not built (RPi5 + HailoRT required)",
)
@pytest.mark.parametrize("h,w", SIZES)
def test_letterbox_matches_python_preprocess(h: int, w: int) -> None:
    bgr = _make_image(h, w)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    shim_canvas = _shim.letterbox_for_test(bgr, 640, 640)
    py_canvas, _, _, _ = py_letterbox(rgb, 640, 640, color=(114, 114, 114))

    assert shim_canvas.shape == py_canvas.shape == (640, 640, 3)
    assert shim_canvas.dtype == py_canvas.dtype == np.uint8

    if np.array_equal(shim_canvas, py_canvas):
        return  # exact match — best case

    # Fall back to <=1 LSB bound; log the actual max diff for future drift detection.
    diff = np.abs(shim_canvas.astype(np.int16) - py_canvas.astype(np.int16))
    max_diff = int(diff.max())
    print(f"\nletterbox max abs diff at ({h}x{w}): {max_diff}")
    assert max_diff <= 1, (
        f"letterbox geometry drift > 1 LSB at ({h}x{w}): max abs diff = {max_diff}\n"
        "Expected <=1 LSB from float-vs-double, std::round-vs-banker's, or "
        "OpenCV link-time skew. >1 LSB indicates a real geometry bug."
    )


@pytest.mark.skipif(
    not _shim_available,
    reason="hailo_tappas_baseline shim not built (RPi5 + HailoRT required)",
)
def test_bgr_to_rgb_conversion() -> None:
    """A non-padding pixel should be RGB (200, 100, 50), not BGR (50, 100, 200)."""
    bgr = _make_image(1280, 720)
    canvas = _shim.letterbox_for_test(bgr, 640, 640)
    # Centre pixel of the canvas falls inside the resized image (not padding)
    # for any of our test sizes, so it carries the input colour after
    # BGR -> RGB conversion.
    centre = canvas[320, 320]
    assert tuple(int(c) for c in centre) == (200, 100, 50), (
        f"BGR->RGB conversion failed: centre pixel = {tuple(centre)}"
    )
