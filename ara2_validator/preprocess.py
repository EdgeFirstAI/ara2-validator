"""Image preprocessing for Ara240 YOLO inference.

Supports two paths:
  1. **cv2/numpy** — CPU letterbox resize, normalize, quantize, transpose.
  2. **HAL GPU** — DMA-BUF tensor import, GPU letterbox-resize via
     ImageProcessor.convert() with zero-copy into the NPU input tensor.

The HAL GPU path eliminates all CPU copies between camera/disk and NPU:
  JPEG → hal.Tensor.load_from_bytes → GPU letterbox-resize → DMA NPU input
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import edgefirst_hal as hal
except ImportError:
    hal = None


@dataclass
class LetterboxInfo:
    """Metadata needed to map detections back to original image coords."""
    scale: float        # Resize scale factor
    pad_x: int          # Left padding pixels
    pad_y: int          # Top padding pixels
    orig_w: int         # Original image width
    orig_h: int         # Original image height


def compute_letterbox_rect(
    src_w: int, src_h: int, dst_w: int, dst_h: int
) -> tuple[float, int, int, int, int]:
    """Compute letterbox fit parameters.

    Returns (scale, pad_x, pad_y, new_w, new_h).
    """
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    return scale, pad_x, pad_y, new_w, new_h


def letterbox(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    """CPU letterbox resize matching Ultralytics LetterBox(center=True).

    Uses cv2.copyMakeBorder for padding (same as Ultralytics) rather than
    np.full + array assignment. The rounding convention for asymmetric
    padding (odd remainder) follows Ultralytics: round(x - 0.1) for
    top/left, round(x + 0.1) for bottom/right.
    """
    h, w = img.shape[:2]
    scale, pad_x, pad_y, new_w, new_h = compute_letterbox_rect(
        w, h, target_w, target_h)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Ultralytics: round(dw - 0.1) for left, round(dw + 0.1) for right
    # Our pad_x = (target_w - new_w) // 2 is equivalent to round(dw - 0.1)
    right = target_w - new_w - pad_x
    bottom = target_h - new_h - pad_y
    padded = cv2.copyMakeBorder(
        resized, pad_y, bottom, pad_x, right,
        cv2.BORDER_CONSTANT, value=color)
    return padded, scale, pad_x, pad_y


def preprocess(
    image_path: str,
    input_h: int,
    input_w: int,
    input_qn: float,
    input_offset: int,
    input_signed: bool,
) -> tuple[np.ndarray, LetterboxInfo]:
    """CPU preprocessing: load → letterbox → normalize → quantize → CHW."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]

    padded, scale, pad_x, pad_y = letterbox(rgb, input_h, input_w)

    normalized = padded.astype(np.float32) / 255.0
    quantized = np.round(normalized / input_qn + input_offset)
    dtype = np.int8 if input_signed else np.uint8
    info = np.iinfo(dtype)
    quantized = np.clip(quantized, info.min, info.max).astype(dtype)

    tensor = np.transpose(quantized, (2, 0, 1))[np.newaxis]  # (1, C, H, W)

    lb_info = LetterboxInfo(
        scale=scale, pad_x=pad_x, pad_y=pad_y,
        orig_w=orig_w, orig_h=orig_h,
    )
    return tensor, lb_info


# ── HAL GPU preprocessing ────────────────────────────────────────────────────


class HalPreprocessor:
    """GPU-accelerated preprocessor with DMA-BUF tensor caching.

    Mirrors the pattern from ara2-rs/examples/yolov8.py:
      - Imports the NPU input tensor as a DMA-BUF HAL image (once)
      - Each frame: loads JPEG → GPU letterbox-resize → zero-copy to NPU
      - No CPU copies in the hot path
    """

    def __init__(
        self,
        model,
        input_h: int,
        input_w: int,
        input_signed: bool,
    ):
        if hal is None:
            raise ImportError("edgefirst_hal required for HAL preprocessing")

        self.input_h = input_h
        self.input_w = input_w
        self.processor = hal.ImageProcessor()

        # Import the model's input DMA-BUF as a HAL image for zero-copy writes
        input_fd = model.input_tensor_fd(0)
        try:
            dtype_str = "int8" if input_signed else "uint8"
            self.dst = self.processor.import_image(
                input_fd, input_w, input_h, hal.PixelFormat.PlanarRgb,
                dtype=dtype_str,
            )
        finally:
            os.close(input_fd)

    def decode_image(self, image_path: str):
        """Load/decode image into a DMA-backed HAL tensor.

        This represents the image acquisition step (from disk, camera, etc.)
        and is timed separately from the preprocess (GPU convert) step.

        Returns the source tensor and original dimensions.
        """
        with open(image_path, "rb") as f:
            jpeg_bytes = f.read()
        src = hal.Tensor.load_from_bytes(
            jpeg_bytes, format=hal.PixelFormat.Rgba,
            mem=hal.TensorMemory.DMA,
        )
        return src, src.width, src.height

    def preprocess(self, src, img_w: int, img_h: int) -> LetterboxInfo:
        """GPU letterbox-resize DMA tensor into the NPU input tensor.

        This is the pure DMA-to-DMA preprocess step, independent of
        image acquisition method. The model's input tensor is written
        in-place via DMA — no CPU copy needed.
        """
        scale, pad_x, pad_y, new_w, new_h = compute_letterbox_rect(
            img_w, img_h, self.input_w, self.input_h)

        # GPU letterbox-resize with YOLO gray padding
        dst_crop = hal.Rect(pad_x, pad_y, new_w, new_h)
        self.processor.convert(src, self.dst, dst_crop=dst_crop,
                               dst_color=(114, 114, 114, 255))

        return LetterboxInfo(
            scale=scale, pad_x=pad_x, pad_y=pad_y,
            orig_w=img_w, orig_h=img_h,
        )
