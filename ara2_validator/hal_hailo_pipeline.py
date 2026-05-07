"""HAL inference pipeline for Hailo-8/8L NPUs.

This module exposes :class:`HalHailoPipeline`, the validator's
``--backend hal-hailo`` baseline. It pairs HailoRT (NPU compute) with
EdgeFirst HAL (preprocess + decode + masks) on the same RPi5 + Hailo-8L
hardware that :class:`HailoTappasPipeline` benchmarks against the
upstream TAPPAS reference. The intent is the exact head-to-head:

  HAL preprocess + Hailo NPU + HAL decode  vs.  OpenCV preprocess + Hailo NPU + TAPPAS decode

Why a separate pipeline class
=============================

The Ara240 :class:`HalPipeline` is structured around the EdgeFirst NPU's
shared DMA-BUF pool with the V3D GPU — ``Tensor.from_fd`` wraps NPU
output buffers into HAL Tensors zero-copy, and the GPU letterbox writes
straight into the NPU's input DMA-BUF. None of that is available on
Hailo: the Hailo NPU's host buffers live behind HailoRT's own
PCIe-bound runtime, with no shared mapping into RPi5's GPU.

The pragmatic compromise is therefore:

* HAL ImageProcessor still does the GPU letterbox into a HAL-allocated
  destination, but we then memcpy the result into a HailoRT-managed
  numpy input buffer (``np.copyto`` over the mapped GPU buffer).
* Each per-frame HailoRT output (numpy uint8 host buffer) is wrapped as
  a fresh HAL Tensor via ``hal.Tensor(...) + .from_numpy(buf)`` and
  tagged with the embedded-schema's child name so the per-scale Decoder
  binds it correctly.

This is one extra memcpy in each direction vs. Ara240, but the heavy
work — per-scale DFL decode, NMS, mask matmul + sigmoid + crop +
resize — runs in HAL exactly the same as on Ara240, so the
postprocess time is the apples-to-apples comparable against TAPPAS.

Schema name binding
===================

The embedded ``edgefirst.json`` (v0.3.0+) carries semantic per-FPN
names — ``boxes_0``, ``scores_1``, ``mask_coefs_2``, ``protos`` —
and HAL's per-scale Decoder binds runtime tensors by these names.
HailoRT, in contrast, enumerates outputs by the DFC compiler's
internal layer names (``model/conv44`` and friends), which the schema
does not currently embed. We bridge the gap at construction time by
matching schema children to HailoRT outputs by *shape* — every yolov8-seg
FPN level has a unique 4-D shape, so the pairing is unambiguous.

Older (pre-v0.3.0) HEFs encode the protos direct-output as NCHW
``[1, 32, 160, 160]``; v0.3.0+ correctly uses NHWC
``[1, 160, 160, 32]``. We canonicalize on the way in: read the
embedded schema, normalise protos to NHWC if needed, then both the
HailoRT format request and the HAL Decoder schema agree.
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
import time as _time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import edgefirst_hal as hal
except ImportError:  # pragma: no cover — caught at construction
    hal = None  # type: ignore

try:
    import hailo_platform as hpf  # HailoRT Python bindings
except ImportError:  # pragma: no cover
    hpf = None  # type: ignore

from .preprocess import LetterboxInfo, compute_letterbox_rect


log = logging.getLogger(__name__)


@dataclass
class HalHailoInferenceResult:
    """One image's worth of HAL+Hailo output, plus per-stage timings.

    Mirrors :class:`HalInferenceResult` so the validator dispatch can
    treat both backends uniformly.
    """

    boxes: object
    scores: object
    classes: object
    proto_data: object
    lb_info: LetterboxInfo
    timings: dict


# ── HEF helpers ──────────────────────────────────────────────────────────────


def _split_hef_zip_trailer(
    hef_path: Path,
) -> tuple[bytes, Optional[dict], Optional[list]]:
    """Split a hailo-converter HEF into (raw_hef, edgefirst_json, labels).

    HailoRT 4.23.0's HEF parser does a strict length check that rejects
    files with a ZIP trailer. The trailer pattern is the documented
    schema-v2 wire format (see hailo-converter ARCHITECTURE.md): the
    raw HEF protobuf followed by a ZIP archive containing
    ``edgefirst.json`` and ``labels.txt``.

    We forward-scan for the first ``PK\\x03\\x04`` magic and validate by
    opening as a ZIP. The first match whose archive contains
    ``edgefirst.json`` is the split point.
    """
    data = hef_path.read_bytes()
    for i in range(len(data) - 4):
        if data[i:i + 4] != b"PK\x03\x04":
            continue
        try:
            zf = zipfile.ZipFile(io.BytesIO(data[i:]))
        except zipfile.BadZipFile:
            continue
        names = zf.namelist()
        if "edgefirst.json" not in names:
            continue
        meta = json.loads(zf.read("edgefirst.json"))
        labels: Optional[list] = None
        if "labels.txt" in names:
            text = zf.read("labels.txt").decode("utf-8").strip()
            labels = [ln for ln in text.splitlines() if ln.strip()]
        return data[:i], meta, labels
    return data, None, None


def _normalize_protos_to_nhwc(meta: dict) -> dict:
    """Patch protos top-level shape/dshape to NHWC if it's NCHW.

    Pre-v0.3.0 hailo-converter emitted protos as NCHW
    ``[1, 32, 160, 160]``. v0.3.0+ correctly emits NHWC
    ``[1, 160, 160, 32]``. HAL's per-scale Decoder reads the schema's
    ``dshape`` to bind tensor axes by name, and our HailoRT output is
    NHWC (``set_format_order(NHWC)``) — so we canonicalize on the way
    in to keep the runtime path agnostic.
    """
    out_meta = json.loads(json.dumps(meta))  # deep copy
    for top in out_meta.get("outputs", []) or []:
        if top.get("type") != "protos":
            continue
        shape = top.get("shape")
        if not (isinstance(shape, list) and len(shape) == 4):
            continue
        n, c, h, w = shape
        # NCHW heuristic: channels (32) is much smaller than spatial dims
        if c < h and c < w:
            top["shape"] = [n, h, w, c]
            top["dshape"] = [
                {"batch": n}, {"height": h}, {"width": w}, {"num_protos": c},
            ]
    return out_meta


# ── Pipeline ────────────────────────────────────────────────────────────────


class HalHailoPipeline:
    """One-stop HAL+Hailo inference pipeline for benchmarking.

    Holds long-lived HailoRT objects (``VDevice``, ``InferModel``,
    ``ConfiguredInferModel``, bindings, output buffers) and HAL objects
    (``ImageProcessor``, pre-allocated destination tensor, ``Decoder``)
    for the lifetime of the run. All per-frame work in :meth:`infer_one`
    is reused state — the only allocations on the hot path are the
    short-lived HAL Tensor wrappers around the HailoRT output buffers.

    Parameters
    ----------
    hef_path : str
        Path to a hailo-converter v0.3.0+ HEF (with embedded schema-v2
        ``edgefirst.json`` ZIP trailer). Pre-v0.3.0 HEFs work too — the
        protos NCHW→NHWC normalization handles them transparently.
    score_threshold, iou_threshold : float
        NMS thresholds forwarded to the HAL Decoder. Defaults are
        Ultralytics ``val`` (0.001 / 0.7) for full PR-curve integration;
        deployed pipelines typically use 0.5 / 0.7.
    max_detections : int
        Per-frame detection cap. Matches Ultralytics ``max_det=300``.
    """

    def __init__(
        self,
        hef_path: str,
        *,
        score_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
    ):
        if hal is None:
            raise ImportError(
                "edgefirst_hal is required for HalHailoPipeline. "
                "Install with: pip install 'edgefirst-hal>=0.20.0'"
            )
        if hpf is None:
            raise ImportError(
                "hailo_platform is required for HalHailoPipeline. "
                "Install HailoRT debs from Hailo Developer Zone."
            )

        self.hef_path = hef_path
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = max_detections

        # 1. Strip ZIP trailer for HailoRT, keep the JSON for HAL Decoder.
        raw_hef, edgefirst_json, labels = _split_hef_zip_trailer(Path(hef_path))
        if edgefirst_json is None:
            raise RuntimeError(
                f"{hef_path}: no edgefirst.json ZIP trailer found. "
                "HAL+Hailo requires hailo-converter-emitted HEFs with "
                "embedded schema-v2 metadata."
            )
        self.embedded_json = _normalize_protos_to_nhwc(edgefirst_json)
        self.labels = labels or []

        # HailoRT 4.23.0 only opens HEFs from a file path (the Python
        # binding doesn't expose the C++ ``HEF::create_from_buffer``
        # overload). Write the stripped bytes to a temp file alive for
        # the lifetime of the InferModel.
        self._hef_tmp = tempfile.NamedTemporaryFile(
            suffix=".hef", delete=False,
        )
        self._hef_tmp.write(raw_hef)
        self._hef_tmp.flush()

        # 2. HailoRT setup: VDevice → InferModel → configure → activate.
        self.vdevice = hpf.VDevice()
        self.infer_model = self.vdevice.create_infer_model(self._hef_tmp.name)

        self.input_name = list(self.infer_model.input_names)[0]
        self.output_names = list(self.infer_model.output_names)

        # Force NHWC on every output. HailoRT's default is FCR
        # (padding-aware physical layout); HAL's per-scale Decoder reads
        # tensors in canonical NHWC of the schema.
        for name in self.output_names:
            self.infer_model.output(name).set_format_order(
                hpf.FormatOrder.NHWC,
            )

        in_vstream = self.infer_model.input(self.input_name)
        ih, iw, ic = in_vstream.shape
        self.input_h = int(ih)
        self.input_w = int(iw)
        self.input_channels = int(ic)
        self.input_dim = float(max(self.input_w, self.input_h))

        self.cfg = self.infer_model.configure()
        self.cfg.activate()
        self.bindings = self.cfg.create_bindings()

        # 3. Per-output host buffers (numpy uint8) and quantization
        # cache. HailoRT writes outputs in-place via bindings; we reuse
        # the buffers across frames.
        self._output_buffers: dict = {}
        self._output_quants: dict = {}
        self._output_shapes: dict = {}
        for name in self.output_names:
            ovs = self.infer_model.output(name)
            shape = tuple(int(x) for x in ovs.shape)  # HWC, no batch
            buf = np.zeros((1, *shape), dtype=np.uint8)
            self._output_buffers[name] = buf
            self._output_shapes[name] = (1, *shape)
            # quant_infos is a list (per-channel possible); for the standard
            # YOLOv8-seg presets HailoRT reports a single per-tensor entry.
            qi = ovs.quant_infos[0]
            self._output_quants[name] = (
                float(qi.qp_scale), float(qi.qp_zp),
            )
            self.bindings.output(name).set_buffer(buf)

        # Input buffer (HailoRT input is uint8 NHWC for a YOLOv8-seg
        # model; we update its contents per-frame via np.copyto).
        self._input_buffer = np.zeros(
            (self.input_h, self.input_w, self.input_channels),
            dtype=np.uint8,
        )
        self.bindings.input(self.input_name).set_buffer(self._input_buffer)

        # 4. Pair HailoRT output names → schema child names by shape.
        # HAL's per-scale Decoder will look up tensors by the schema's
        # ``name`` field; HailoRT-internal names won't resolve. This map
        # is computed once at construction.
        self._schema_name_for: dict = self._pair_schema_to_runtime()

        # 5. HAL preprocess: ImageProcessor + interleaved-Rgb destination.
        # PixelFormat.Rgb is HWC — avoids a CHW→HWC transpose on map
        # (~3 ms saved on a 640×640 frame vs PlanarRgb).
        self.processor = hal.ImageProcessor()
        self.dst = self.processor.create_image(
            self.input_w, self.input_h, hal.PixelFormat.Rgb, dtype="uint8",
        )

        # 6. HAL Decoder from the (normalized) embedded schema. ClassAware
        # NMS matches Ultralytics + the numpy reference. ``input_dims``
        # must be passed explicitly: without it, HAL infers from the
        # first 3 dims of an arbitrary schema shape and lands on
        # ``(H, C)`` instead of ``(W, H)``, which produces nonsense
        # boxes (mixed normalized x / pixel y).
        self.decoder = hal.Decoder.new_from_json_str(
            json.dumps(self.embedded_json),
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            nms=hal.Nms.ClassAware,
            input_dims=(self.input_w, self.input_h),
        )

        log.info(
            "HalHailoPipeline: %d×%d×%d in, %d outputs",
            self.input_h, self.input_w, self.input_channels,
            len(self.output_names),
        )

    def close(self) -> None:
        """Explicit teardown for use in long-lived processes.

        Calling ``cfg.deactivate()`` from ``__del__`` segfaults during
        Python interpreter shutdown — the HailoRT VDevice destructor
        and the ConfiguredInferModel destructor race over PCIe
        resources. We rely on HailoRT's own RAII for the implicit case
        and expose ``close()`` for callers who care about ordering.
        """
        try:
            self.cfg.deactivate()
        except Exception:
            pass
        try:
            Path(self._hef_tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _pair_schema_to_runtime(self) -> dict:
        """Build {hailort_name → schema_name} by shape-matching.

        The embedded schema names children semantically (``boxes_0``,
        ``scores_2``, ``protos``) but HailoRT enumerates outputs by
        their DFC layer names (``model/conv44``, etc.). HAL's per-scale
        Decoder binds tensors by the schema's ``name``, so we re-tag
        each runtime tensor with its matching schema name.

        For YOLOv8-seg the pairing is unambiguous: each FPN child has a
        unique 4-D NHWC shape (e.g. ``[1, 80, 80, 64]`` vs
        ``[1, 40, 40, 80]``), and the protos direct-output is the only
        ``[1, 160, 160, 32]`` tensor.
        """
        # Collect (schema_name, shape) for both children of split
        # outputs and bare top-level (non-split) outputs.
        schema_entries: list = []
        for top in self.embedded_json.get("outputs", []) or []:
            children = top.get("outputs") or []
            if children:
                for ch in children:
                    if ch.get("name") and ch.get("shape"):
                        schema_entries.append(
                            (ch["name"], tuple(ch["shape"]))
                        )
            elif top.get("name") and top.get("shape"):
                schema_entries.append((top["name"], tuple(top["shape"])))

        # Greedy shape-match against runtime outputs.
        used = set()
        pairing: dict = {}
        for hailo_name, shape in self._output_shapes.items():
            for idx, (schema_name, schema_shape) in enumerate(schema_entries):
                if idx in used:
                    continue
                if tuple(schema_shape) == tuple(shape):
                    pairing[hailo_name] = schema_name
                    used.add(idx)
                    break
            else:
                # No match — surface the mismatch loudly. Common cause:
                # schema and HEF tensor count diverge.
                raise RuntimeError(
                    f"HailoRT output '{hailo_name}' shape {shape} has no "
                    f"matching schema entry. Schema entries: "
                    f"{schema_entries}"
                )
        return pairing

    # ------------------------------------------------------------------
    # Per-frame interface
    # ------------------------------------------------------------------

    def warmup(self, image_paths, n: int = 3) -> None:
        """Warm caches by running the full pipeline on ``n`` images."""
        for image_path in image_paths[: max(0, n)]:
            try:
                self.infer_one(image_path)
            except FileNotFoundError:
                continue

    def infer_one(self, image_path) -> HalHailoInferenceResult:
        """Run the full HAL+Hailo pipeline on one image.

        Pipeline stages (each timed):

        1. **decode** — JPEG → DMA-backed source tensor via
           :func:`hal.Tensor.load_from_bytes`.
        2. **preprocess** — HAL GPU letterbox into pre-allocated ``Rgb``
           destination, then ``np.copyto`` the mapped buffer into
           HailoRT's input (one host-side memcpy at the GPU/PCIe
           boundary; unavoidable on this hardware).
        3. **inference** — HailoRT sync ``cfg.run([bindings], 5000)``.
        4. **nms_decode** — wrap each runtime output buffer as a HAL
           Tensor (with schema name + per-tensor quantization) and call
           ``decoder.decode_proto``.

        Mask materialisation runs *outside* this method — the validator
        chooses retina vs fast mode via ``materialize_masks_for_coco_rle``
        in :mod:`postprocess_hal`.
        """
        timings: dict = {}

        # Stage 1: decode JPEG.
        t0 = _time.perf_counter()
        with open(image_path, "rb") as f:
            jpeg_bytes = f.read()
        src = hal.Tensor.load_from_bytes(
            jpeg_bytes, format=hal.PixelFormat.Rgba,
            mem=hal.TensorMemory.DMA,
        )
        img_w, img_h = src.width, src.height
        timings["decode"] = (_time.perf_counter() - t0) * 1e3

        # Stage 2: HAL letterbox + copy to HailoRT input.
        t0 = _time.perf_counter()
        scale, pad_x, pad_y, new_w, new_h = compute_letterbox_rect(
            img_w, img_h, self.input_w, self.input_h,
        )
        self.processor.convert(
            src, self.dst,
            dst_crop=hal.Rect(pad_x, pad_y, new_w, new_h),
            dst_color=(114, 114, 114, 255),
        )
        with self.dst.map() as mv:
            np.copyto(
                self._input_buffer,
                np.asarray(mv.view()).reshape(
                    self.input_h, self.input_w, self.input_channels,
                ),
            )
        timings["preprocess"] = (_time.perf_counter() - t0) * 1e3

        lb_info = LetterboxInfo(
            scale=scale, pad_x=pad_x, pad_y=pad_y,
            orig_w=img_w, orig_h=img_h,
        )

        # Stage 3: HailoRT sync inference.
        t0 = _time.perf_counter()
        self.cfg.run([self.bindings], 5000)
        timings["inference"] = (_time.perf_counter() - t0) * 1e3

        # Stage 4: wrap outputs as HAL Tensors and decode.
        t0 = _time.perf_counter()
        tensors = []
        for hailo_name in self.output_names:
            buf = self._output_buffers[hailo_name]
            schema_name = self._schema_name_for[hailo_name]
            t = hal.Tensor(list(buf.shape), "uint8", name=schema_name)
            t.from_numpy(buf)
            scale_q, zp_q = self._output_quants[hailo_name]
            t.set_quantization_per_tensor(scale_q, int(round(zp_q)))
            tensors.append(t)
        boxes, scores, classes, proto_data = self.decoder.decode_proto(
            tensors, max_boxes=self.max_detections,
        )
        timings["nms_decode"] = (_time.perf_counter() - t0) * 1e3

        return HalHailoInferenceResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            proto_data=proto_data,
            lb_info=lb_info,
            timings=timings,
        )
