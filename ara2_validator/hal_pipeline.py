"""HAL inference pipeline for Ara240 segmentation models.

This module exposes :class:`HalPipeline`, the recommended one-stop
binding between a NXP Ara240 DVM model and the EdgeFirst HAL
library. It encodes the integration patterns the HAL Optimization
Guide expects (https://github.com/EdgeFirstAI/hal#optimization-guide)
so that downstream applications can copy the class verbatim instead
of re-deriving the patterns from scratch.

The validator's :mod:`ara2_validator.edgefirst` and the ``--backend
hal`` branch of :mod:`ara2_validator.__main__` both delegate to this
class. Customers building their own Ara240 + HAL applications should
be able to lift this file into their project as-is.

Why this module exists
======================

The naïve approach to using HAL — construct an ``ImageProcessor``,
``Decoder``, and output ``Tensor`` per frame — silently breaks the
zero-copy DMA-BUF pipeline that makes HAL fast in the first place.
Each per-frame construction triggers GPU driver re-imports, dvproxy
re-allocation, and lost rayon thread-pool warmth. The :class:`HalPipeline`
class exists to make the right pattern (construct once, reuse
everywhere) the obvious one.

Concretely, this class enforces:

* **HAL Rule 1** — output DMA-BUF tensors are wrapped via
  :func:`hal.Tensor.from_fd` exactly once at startup.
* **HAL Rule 4** — the :class:`hal.Decoder` is constructed once and
  reused on every frame.
* **HAL Rule 5** — a single :class:`hal.ImageProcessor` (owned by
  :class:`HalPreprocessor`) is constructed once and reused; the
  destination image descriptor binding the NPU's input DMA-BUF is
  registered once via :func:`processor.import_image`.

Each of these is also commented inline below.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import edgefirst_hal as hal
except ImportError:  # pragma: no cover — caught at construction
    hal = None  # type: ignore

from .preprocess import HalPreprocessor, LetterboxInfo


@dataclass
class HalInferenceResult:
    """One image's worth of HAL inference output, plus per-stage timings.

    Attributes
    ----------
    boxes, scores, classes : array-like
        ``Decoder.decode_proto()`` outputs. Boxes are in normalised
        ``xyxy`` coordinates inside the letterboxed model-input space
        (because of the box-quant ``qn / input_dim`` trick — see
        :class:`HalPipeline` docstring). Caller is responsible for
        un-letterboxing if pixel-space coords are needed.
    proto_data : edgefirst_hal.ProtoData | None
        Prototype tensor + per-detection mask coefficients, ready to be
        passed to :func:`materialize_masks_for_coco`. ``None`` when the
        model has no segmentation head.
    lb_info : LetterboxInfo
        Letterbox descriptor for this image (pad_x, pad_y, scale,
        orig_w, orig_h). Needed by the caller to un-letterbox boxes
        and to compute the letterbox-norm tuple for mask materialisation.
    timings : dict[str, float]
        Per-stage timings in milliseconds. Keys: ``decode``,
        ``preprocess``, ``inference``, ``dma_input``, ``npu_compute``,
        ``dma_output``, ``nms_decode``.
    """

    boxes: object
    scores: object
    classes: object
    proto_data: object
    lb_info: LetterboxInfo
    timings: dict[str, float]


class HalPipeline:
    """One-stop HAL inference pipeline for Ara240 segmentation models.

    Constructs the ``HalPreprocessor``, ``Decoder``, and DMA-BUF output
    tensor handles **once** and reuses them across all frames. This is
    the construct-once / reuse-many pattern the HAL Optimization Guide
    requires for zero-copy throughput.

    Customers porting from this validator can copy this class verbatim
    into their own application.

    Parameters
    ----------
    model : edgefirst_ara2.Model
        Loaded Ara240 model (returned by
        :func:`ara2_validator.model.load_model`). The pipeline calls
        ``model.allocate_tensors("dma")`` during construction, which is
        required for :func:`hal.Tensor.from_fd` to obtain valid
        DMA-BUF file descriptors from ``model.output_tensor_fd(i)``.
    spec : ModelSpec
        Companion spec returned by ``load_model`` (input shape, dtype,
        quantisation parameters).
    score_threshold : float, default 0.001
        NMS score threshold. ``0.001`` matches Ultralytics ``val``
        defaults and produces a complete precision-recall curve. A
        deployed pipeline would use a much higher threshold (typically
        ``0.50``) which reduces detection counts and postprocess time.
    iou_threshold : float, default 0.7
        NMS IoU threshold. ``0.7`` matches Ultralytics ``val`` default.
    max_detections : int, default 300
        Maximum detections per image. ``300`` matches Ultralytics
        ``max_det``.
    """

    def __init__(
        self,
        model,
        spec,
        *,
        score_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
    ):
        if hal is None:
            raise ImportError(
                "edgefirst_hal is required for HalPipeline. "
                "Install with: pip install 'edgefirst-hal>=0.18.2'"
            )

        self.model = model
        self.spec = spec
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        c, input_h, input_w = spec.input_shape
        self.input_h = input_h
        self.input_w = input_w
        self.input_dim = float(max(input_w, input_h))

        # Re-allocate model tensors as DMA-BUF.
        #
        # `load_model()` calls `allocate_tensors()` in the default mode,
        # which does NOT create DMA-BUF file descriptors. Without DMA
        # mode, `model.output_tensor_fd(i)` returns -1 (or raises) and
        # the `Tensor.from_fd()` wrapping below would fail. Calling
        # `allocate_tensors("dma")` here is the prerequisite for the
        # HAL zero-copy decode path.
        model.allocate_tensors("dma")

        # HAL Rule 5 — One ImageProcessor per pipeline thread.
        #
        # `HalPreprocessor.__init__` calls `processor.import_image()`,
        # which registers the NPU's input DMA-BUF fd with the GPU
        # driver and maps it into the GPU's address space. This is a
        # one-shot setup: every frame, `processor.convert(src, dst,
        # ...)` writes letterboxed pixels directly into NPU input
        # memory via the cached mapping. Re-importing the fd per frame
        # would pay a kernel round-trip and prevent the GPU from
        # pipelining the DMA-to-DMA transfer.
        self.preprocessor = HalPreprocessor(
            model, input_h, input_w, spec.input_signed
        )

        # HAL Rule 4 — Decoder built once, decode many.
        #
        # `Decoder.new_from_outputs()` parses the typed Output
        # descriptors (one per model output: boxes, scores,
        # mask_coefficients, protos), validates the schema, and
        # allocates rayon-shared scratch buffers for the NMS + mask
        # paths. Recreating the decoder per frame would rebuild that
        # schema and re-allocate scratch on every call.
        self._shapes, self.decoder = self._build_decoder()

        # HAL Rule 1 — Reuse output tensors across frames.
        #
        # Each `Tensor.from_fd()` call registers the NPU's output
        # DMA-BUF fd with the GPU driver. After construction here, the
        # HAL Decoder reads NPU output memory directly through the GPU
        # mapping every frame — no memcpy from device DRAM to host
        # numpy. Wrapping per frame would re-register the fd each
        # iteration and break the zero-copy path.
        #
        # Ownership note: from_fd() takes the fd — do NOT call
        # os.close(fd) after this loop.
        self.output_tensors = self._wrap_output_tensors()

    # ------------------------------------------------------------------
    # Private setup helpers
    # ------------------------------------------------------------------

    def _build_decoder(self):
        """Construct the typed Output descriptors and the HAL Decoder.

        The DVM model exposes per-output ``shape``, ``qn``, and
        ``offset`` metadata. We classify each output by its shape into
        one of four roles (protos, boxes, mask_coefficients, scores)
        and emit a typed :class:`hal.Output` descriptor for it.

        The "box quant trick" — dividing the boxes' raw quant scale by
        ``input_dim`` — is the only non-obvious step here. Models in
        this family emit boxes in *pixel* coordinates (xyxy ∈ [0, 640]
        for a 640×640 model). Pre-scaling the dequant slope by
        ``1 / input_dim`` makes HAL produce ``[0, 1]`` normalised
        coordinates directly, saving a per-frame division. The caller
        un-letterboxes afterwards via ``(p * input_dim - pad) / scale``
        (see :func:`unletterbox_box_array`).
        """
        shapes, quants_list = [], []
        for i in range(self.model.n_outputs):
            shape = list(self.model.output_shape(i))
            # Strip trailing-1 dimensions and prepend a batch axis. HAL
            # expects every output as ``[batch, ...]``; a 4-D protos
            # tensor coming back as ``[32, 160, 160, 1]`` needs the
            # trailing 1 stripped before we add a leading 1.
            while len(shape) > 1 and shape[-1] == 1:
                shape.pop()
            shape.insert(0, 1)
            shapes.append(shape)
            quants_list.append(self.model.output_quants(i))

        proto_shape = next((s for s in shapes if len(s) == 4), None)
        n_proto_ch = proto_shape[1] if proto_shape else None

        outputs_desc = []
        for i, shape in enumerate(shapes):
            qn = quants_list[i].qn
            offset = quants_list[i].offset

            if len(shape) == 4:
                # Prototype masks: [1, 32, 160, 160] — NCHW from the
                # DVM. HAL's canonical protos layout is NHWC, so
                # passing the raw shape would make HAL read
                # num_protos=160 (trailing dim) instead of 32.
                # Provide explicit dshape so HAL interprets NCHW.
                b, ch, h, w = shape
                dshape = [
                    (hal.DimName.Batch, b),
                    (hal.DimName.NumProtos, ch),
                    (hal.DimName.Height, h),
                    (hal.DimName.Width, w),
                ]
                out = hal.Output.protos(
                    dshape=dshape, decoder=hal.DecoderType.Ultralytics
                )
                out = out.with_quantization(qn, offset)
            elif len(shape) == 3 and shape[1] == 4:
                # Boxes: [1, 4, 8400] in pixel space. Pre-scale the
                # dequant slope by 1/input_dim so HAL emits normalised
                # [0,1] coords directly. See _build_decoder docstring
                # for the trade-off.
                scale = qn / self.input_dim
                out = hal.Output.boxes(
                    shape=shape, decoder=hal.DecoderType.Ultralytics
                )
                out = out.with_quantization(scale, offset).with_normalized(True)
            elif n_proto_ch and len(shape) == 3 and shape[1] == n_proto_ch:
                # Mask coefficients: [1, 32, 8400] — one row per
                # detection that linearly combines with the proto
                # plane to produce the per-detection mask logits.
                out = hal.Output.mask_coefficients(
                    shape=shape, decoder=hal.DecoderType.Ultralytics
                )
                out = out.with_quantization(qn, offset)
            else:
                # Scores: [1, num_classes, 8400]. Also catches any
                # unrecognised shape — HAL's NMS path will fail
                # explicitly if the shape isn't valid for scores.
                out = hal.Output.scores(
                    shape=shape, decoder=hal.DecoderType.Ultralytics
                )
                out = out.with_quantization(qn, offset)
            outputs_desc.append(out)

        # ClassAware NMS matches both Ultralytics' default and the
        # numpy reference's :func:`nms_class_aware`. Class-agnostic
        # NMS would suppress across classes (a high-scoring "dog"
        # could suppress an overlapping "cat"), changing per-class
        # mAP results.
        decoder = hal.Decoder.new_from_outputs(
            outputs_desc,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            nms=hal.Nms.ClassAware,
            decoder_version=hal.DecoderVersion.Yolov8,
        )
        return shapes, decoder

    def _wrap_output_tensors(self):
        """Wrap each output DMA-BUF fd as a HAL Tensor (one-shot)."""
        output_tensors = []
        for i in range(self.model.n_outputs):
            fd = self.model.output_tensor_fd(i)
            oq = self.model.output_quants(i)
            oi = self.model.output_info(i)
            bpp = oi.bpp
            # Dtype is determined by bytes-per-pixel + signedness,
            # which the model metadata reports per-output. Validators
            # must respect this — passing the wrong dtype to
            # `from_fd()` produces silently corrupt output.
            dtype_str = (
                ("int8" if oq.is_signed else "uint8")
                if bpp == 1
                else ("int16" if oq.is_signed else "uint16")
            )
            # `from_fd` takes ownership of the fd; do not call
            # os.close(fd) after this. The HAL Tensor's drop releases
            # the GPU mapping and the fd together.
            t = hal.Tensor.from_fd(fd, self._shapes[i], dtype_str)
            output_tensors.append(t)
        return output_tensors

    # ------------------------------------------------------------------
    # Per-frame interface
    # ------------------------------------------------------------------

    def warmup(self, image_paths, n: int = 3) -> None:
        """Run ``n`` warmup iterations on the first ``n`` images.

        Drives the full pipeline (image decode → preprocess → infer
        → decode_proto) without recording timings. This warms the
        rayon thread pool, GPU shader compile cache, and HAL scratch
        buffers so the first measured frame produces representative
        numbers.
        """
        for image_path in image_paths[: max(0, n)]:
            src, w, h = self.preprocessor.decode_image(str(image_path))
            self.preprocessor.preprocess(src, w, h)
            self.model.run()
            self.decoder.decode_proto(
                self.output_tensors, max_boxes=self.max_detections
            )

    def infer_one(self, image_path) -> HalInferenceResult:
        """Run the full HAL pipeline on one image.

        Pipeline stages (each timed):

        1. **decode** — JPEG → DMA-BUF source tensor.
        2. **preprocess** — GPU letterbox resize, DMA-to-DMA.
        3. **inference** — NPU compute (with PCIe DMA in/out).
        4. **nms_decode** — HAL Decoder NMS + dequant directly on
           the DMA-BUF output tensors.

        Mask materialisation is intentionally *not* part of this
        method — the caller decides between retina and fast modes via
        :func:`ara2_validator.postprocess_hal.materialize_masks_for_coco`.
        """
        import time as _time

        timings: dict[str, float] = {}

        t0 = _time.perf_counter()
        src, img_w, img_h = self.preprocessor.decode_image(str(image_path))
        timings["decode"] = (_time.perf_counter() - t0) * 1e3

        t0 = _time.perf_counter()
        lb_info = self.preprocessor.preprocess(src, img_w, img_h)
        timings["preprocess"] = (_time.perf_counter() - t0) * 1e3

        t0 = _time.perf_counter()
        timing = self.model.run()
        timings["inference"] = (_time.perf_counter() - t0) * 1e3
        # Ara240 firmware reports PCIe transfer + NPU compute time in
        # microseconds. Convert to ms for consistency with the
        # Python-side wall-clock numbers above.
        timings["dma_input"] = timing.input_time_us / 1000.0
        timings["npu_compute"] = timing.run_time_us / 1000.0
        timings["dma_output"] = timing.output_time_us / 1000.0

        t0 = _time.perf_counter()
        boxes, scores, classes, proto_data = self.decoder.decode_proto(
            self.output_tensors, max_boxes=self.max_detections
        )
        timings["nms_decode"] = (_time.perf_counter() - t0) * 1e3

        return HalInferenceResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            proto_data=proto_data,
            lb_info=lb_info,
            timings=timings,
        )


def unletterbox_box_array(
    boxes,
    lb_info: LetterboxInfo,
    input_w: int,
    input_h: int,
):
    """Map normalised letterboxed-space xyxy boxes to original-image pixel coords.

    HAL's :class:`Decoder.decode_proto` returns boxes in normalised
    ``[0, 1]`` xyxy coordinates inside the **letterboxed** model-input
    space (consequence of the box-quant trick described in
    :meth:`HalPipeline._build_decoder`). To produce coordinates in the
    original-image pixel space (what COCO RLE encoding expects), we
    invert the letterbox transform: ``(p * input_dim - pad) / scale``,
    then clip to image bounds.

    Parameters
    ----------
    boxes : array-like
        ``(N, 4)`` xyxy in normalised letterboxed space (HAL output).
    lb_info : LetterboxInfo
        Letterbox descriptor produced by the preprocess step.
    input_w, input_h : int
        Model input dimensions (typically 640, 640).

    Returns
    -------
    numpy.ndarray
        ``(N, 4)`` xyxy in original-image pixel coordinates.
    """
    import numpy as np

    bx = np.asarray(boxes, dtype=np.float32).copy()
    if len(bx) == 0:
        return bx
    bx[:, [0, 2]] = (bx[:, [0, 2]] * input_w - lb_info.pad_x) / lb_info.scale
    bx[:, [1, 3]] = (bx[:, [1, 3]] * input_h - lb_info.pad_y) / lb_info.scale
    bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], 0, lb_info.orig_w)
    bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], 0, lb_info.orig_h)
    return bx
