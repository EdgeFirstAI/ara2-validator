"""Ara240 DVM model loader, metadata parser, and inference driver.

This module handles:
  - Loading a DVM model via the edgefirst_ara2 Unix-socket API
  - Parsing embedded edgefirst.json metadata (schema_version:2 ZIP trailer)
  - Running single-image inference and returning raw quantized output tensors
  - Merging split boxes_xy + boxes_wh into a single [1, 4, N] tensor
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import edgefirst_ara2 as ara2
except ImportError:
    ara2 = None


# ── Metadata types ────────────────────────────────────────────────────────────

@dataclass
class TensorSpec:
    """Specification for one physical output tensor from the DVM."""
    name: str
    role: str               # "boxes_xy", "boxes_wh", "scores", "mask_coefs", "protos"
    shape: list[int]
    dtype: np.dtype
    qn: float               # quantization scale
    offset: int             # quantization zero-point
    index: int              # physical output index in the compiled model


@dataclass
class ModelSpec:
    """Parsed model specification from DVM metadata."""
    input_shape: tuple[int, int, int]  # (C, H, W)
    input_qn: float
    input_offset: int
    input_signed: bool
    labels: list[str]
    decoder: str             # "ultralytics" or "modelpack"
    task: str                # "instance_segmentation", "detection"
    outputs: dict[str, TensorSpec] = field(default_factory=dict)
    # For split-decoder: merged boxes quantization
    merged_box_qn: Optional[float] = None
    merged_box_offset: Optional[int] = None


# ── Metadata extraction ──────────────────────────────────────────────────────

def _read_embedded_json(dvm_path: str) -> dict | None:
    """Read edgefirst.json from the ZIP trailer appended to a DVM file.

    Returns the parsed dict, or None if no embedded ZIP/metadata found.
    """
    data = Path(dvm_path).read_bytes()
    # ZIP trailer appended after the DVM binary — scan for PK header
    for i in range(len(data) - 4):
        if data[i:i + 4] == b"PK\x03\x04":
            try:
                zf = zipfile.ZipFile(io.BytesIO(data[i:]))
                if "edgefirst.json" in zf.namelist():
                    return json.loads(zf.read("edgefirst.json"))
            except zipfile.BadZipFile:
                continue
    # Also try opening the file directly as a ZIP (standard path)
    if zipfile.is_zipfile(dvm_path):
        with zipfile.ZipFile(dvm_path) as zf:
            if "edgefirst.json" in zf.namelist():
                return json.loads(zf.read("edgefirst.json"))
    return None


def _read_labels_from_zip(dvm_path: str) -> list[str]:
    """Try to read labels.txt from the DVM's embedded ZIP."""
    data = Path(dvm_path).read_bytes()
    for i in range(len(data) - 4):
        if data[i:i + 4] == b"PK\x03\x04":
            try:
                zf = zipfile.ZipFile(io.BytesIO(data[i:]))
                if "labels.txt" in zf.namelist():
                    text = zf.read("labels.txt").decode("utf-8").strip()
                    return [ln for ln in text.splitlines() if ln.strip()]
            except zipfile.BadZipFile:
                continue
    return []


# ── Requantization ───────────────────────────────────────────────────────────

def requantize(
    arr: np.ndarray,
    src_qn: float, src_off: int,
    dst_qn: float, dst_off: int,
) -> np.ndarray:
    """Re-express a quantized tensor under a new (qn, offset) pair.

    Computes: dst_raw = round((src_raw - src_off) * src_qn / dst_qn + dst_off)
    and clamps to the original dtype range.
    """
    if src_qn == dst_qn and src_off == dst_off:
        return arr
    dtype = arr.dtype
    info = np.iinfo(dtype)
    fp = (arr.astype(np.float32) - src_off) * np.float32(src_qn / dst_qn) + dst_off
    return np.clip(np.round(fp), info.min, info.max).astype(dtype)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(
    dvm_path: str,
    labels_path: str | None = None,
    socket_path: str = "/var/run/ara2.sock",
) -> tuple["ara2.Model", ModelSpec]:
    """Load a DVM model and parse its metadata.

    Parameters
    ----------
    dvm_path : str
        Path to the .dvm model file.
    labels_path : str, optional
        Path to labels.txt (fallback if not embedded in DVM).
    socket_path : str
        Unix socket path for the Ara240 proxy.

    Returns
    -------
    model : ara2.Model
        The loaded Ara240 model ready for inference.
    spec : ModelSpec
        Parsed model specification with output tensor details.
    """
    if ara2 is None:
        raise ImportError(
            "edgefirst_ara2 is required. Install with: pip install edgefirst-ara2")

    # Connect and load
    session = ara2.Session.create_via_unix_socket(socket_path)
    endpoints = session.list_endpoints()
    if not endpoints:
        raise RuntimeError("No Ara240 endpoints. Is the proxy running?")
    model = endpoints[0].load_model(dvm_path)
    model.allocate_tensors()

    # Input metadata
    iq = model.input_quants(0)
    c, h, w = model.input_shape(0)

    # Labels: embedded ZIP → fallback file → COCO default
    labels = _read_labels_from_zip(dvm_path)
    if not labels and labels_path and Path(labels_path).exists():
        labels = [
            ln for ln in Path(labels_path).read_text().splitlines()
            if ln.strip()
        ]

    # Parse embedded v2 metadata for output classification
    meta = _read_embedded_json(dvm_path)
    decoder = "ultralytics"
    task = "instance_segmentation"
    if meta:
        decoder = "ultralytics"
        for out in meta.get("outputs", []):
            if out.get("decoder"):
                decoder = out["decoder"]
                break
        task = meta.get("model", {}).get("task", task)
        # Labels from metadata if ZIP didn't have them
        if not labels:
            labels = meta.get("dataset", {}).get("classes", [])

    # Build output tensor specs by matching layer names to metadata
    # Build a name→role map from v2 metadata
    name_to_role: dict[str, str] = {}
    if meta and meta.get("schema_version", 1) >= 2:
        for logical in meta.get("outputs", []):
            children = logical.get("outputs", [])
            if children:
                for child in children:
                    name_to_role[child.get("name", "")] = child.get("type", "")
            else:
                name_to_role[logical.get("name", "")] = logical.get("type", "")

    outputs: dict[str, TensorSpec] = {}
    for i in range(model.n_outputs):
        shape = list(model.output_shape(i))
        oq = model.output_quants(i)
        info = model.output_info(i)
        bpp = info.bpp
        dtype = np.int8 if oq.is_signed else np.uint8
        if bpp == 2:
            dtype = np.int16 if oq.is_signed else np.uint16

        # Identify role from metadata layer name
        role = name_to_role.get(info.layer_name, "")
        if not role:
            # Fallback: use shape heuristic
            role = _infer_role_from_shape(shape)

        spec_entry = TensorSpec(
            name=info.layer_name,
            role=role,
            shape=shape,
            dtype=dtype,
            qn=float(oq.qn),
            offset=int(oq.offset),
            index=i,
        )
        outputs[role] = outputs.get(role, spec_entry)
        # If multiple tensors have the same role (shouldn't happen), keep first
        # Store all by index too
        outputs[f"_idx_{i}"] = spec_entry

    # Clean up the index entries
    outputs = {k: v for k, v in outputs.items() if not k.startswith("_idx_")}

    # Determine merged box quantization for split-decoder
    xy_spec = outputs.get("boxes_xy")
    wh_spec = outputs.get("boxes_wh")
    merged_qn = None
    merged_off = None
    if xy_spec and wh_spec:
        merged_qn = xy_spec.qn
        merged_off = xy_spec.offset

    model_spec = ModelSpec(
        input_shape=(c, h, w),
        input_qn=float(iq.qn),
        input_offset=int(iq.offset),
        input_signed=iq.is_signed,
        labels=labels,
        decoder=decoder,
        task=task,
        outputs=outputs,
        merged_box_qn=merged_qn,
        merged_box_offset=merged_off,
    )

    return model, model_spec


def _strip_trailing_ones(shape: list[int]) -> list[int]:
    """Strip trailing dimensions of size 1 (padding), keeping ≥2 dims."""
    while len(shape) > 2 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


def _infer_role_from_shape(shape: list[int]) -> str:
    """Guess output tensor role from its shape (fallback heuristic).

    Handles both clean shapes (e.g. [4, 8400]) and Ara240 padded
    shapes (e.g. [4, 8400, 1]) by stripping trailing ones first.
    """
    s = _strip_trailing_ones(shape)
    if len(s) == 0:
        return "unknown"
    # 3D spatial with 32 channels and large spatial dims → protos
    # e.g. [32, 160, 160] or [160, 160, 32]
    if len(s) == 3 and 32 in s and max(s) > 100:
        return "protos"
    # 2D after stripping: identify by the non-N dimension
    if len(s) == 2:
        # Determine which dim is the feature dim (smaller one, or known size)
        d0, d1 = s[0], s[1]
        feat = d0 if d0 < d1 else d1
        if feat == 2:
            return "boxes_xy"  # ambiguous xy vs wh; caller resolves
        if feat == 4:
            return "boxes"
        if feat == 80:
            return "scores"
        if feat == 32:
            return "mask_coefs"
    return "unknown"


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(
    model,
    spec: ModelSpec,
    input_tensor: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run inference and return dequantized float32 output tensors by role.

    For split-decoder models (boxes_xy + boxes_wh), the two halves are
    dequantized separately and concatenated into a single "boxes" tensor
    with shape [1, 4, N].

    Parameters
    ----------
    model : ara2.Model
        The loaded model.
    spec : ModelSpec
        Model specification with output tensor details.
    input_tensor : np.ndarray
        Preprocessed input tensor, quantized, shape (1, C, H, W).

    Returns
    -------
    dict mapping role name to dequantized float32 tensor:
        "boxes"     : [1, 4, N]  (xcycwh in pixel coords)
        "scores"    : [1, nc, N]
        "mask_coefs": [1, 32, N] (if segmentation model)
        "protos"    : [1, 32, H, W] (if segmentation model)
    """
    model.set_input_tensor(0, input_tensor.flatten())
    model.run()

    # Fetch all raw tensors, stripping trailing padding dimensions
    raw: dict[str, np.ndarray] = {}
    for role, ts in spec.outputs.items():
        t = model.get_output_tensor(ts.index).reshape(ts.shape)
        # Strip trailing size-1 dims (Ara240 padding)
        stripped = _strip_trailing_ones(list(t.shape))
        if stripped != list(t.shape):
            t = t.reshape(stripped)
        raw[role] = t

    # Dequantize each tensor: float_val = (quantized - offset) * qn
    def dequant(arr: np.ndarray, qn: float, offset: int) -> np.ndarray:
        return (arr.astype(np.float32) - offset) * qn

    result: dict[str, np.ndarray] = {}

    # Handle split boxes_xy + boxes_wh → merged boxes [1, 4, N]
    xy_spec = spec.outputs.get("boxes_xy")
    wh_spec = spec.outputs.get("boxes_wh")
    if xy_spec and wh_spec:
        xy_f = dequant(raw["boxes_xy"], xy_spec.qn, xy_spec.offset)
        wh_f = dequant(raw["boxes_wh"], wh_spec.qn, wh_spec.offset)
        # Ensure [1, 2, N] shape
        if xy_f.ndim == 2:
            xy_f = xy_f[np.newaxis]
        if wh_f.ndim == 2:
            wh_f = wh_f[np.newaxis]
        result["boxes"] = np.concatenate([xy_f, wh_f], axis=1)  # [1, 4, N]
    elif "boxes" in raw:
        bs = spec.outputs["boxes"]
        result["boxes"] = dequant(raw["boxes"], bs.qn, bs.offset)

    # Scores
    if "scores" in spec.outputs:
        ss = spec.outputs["scores"]
        result["scores"] = dequant(raw["scores"], ss.qn, ss.offset)

    # Mask coefficients
    if "mask_coefs" in spec.outputs:
        ms = spec.outputs["mask_coefs"]
        result["mask_coefs"] = dequant(raw["mask_coefs"], ms.qn, ms.offset)

    # Protos
    if "protos" in spec.outputs:
        ps = spec.outputs["protos"]
        result["protos"] = dequant(raw["protos"], ps.qn, ps.offset)

    return result


def run_inference_raw(
    model,
    spec: ModelSpec,
    input_tensor: np.ndarray,
) -> list[np.ndarray]:
    """Run inference and return raw quantized tensors as a flat list.

    Used by the HAL postprocessing path which handles dequantization
    internally. For split-decoder models, boxes_xy and boxes_wh are
    requantized to a shared scale and concatenated.

    Returns tensors in the order expected by HAL: one tensor per logical
    output (boxes, scores, mask_coefs, protos).
    """
    model.set_input_tensor(0, input_tensor.flatten())
    model.run()

    raw_by_role: dict[str, np.ndarray] = {}
    for role, ts in spec.outputs.items():
        t = model.get_output_tensor(ts.index).reshape(ts.shape)
        stripped = _strip_trailing_ones(list(t.shape))
        if stripped != list(t.shape):
            t = t.reshape(stripped)
        raw_by_role[role] = t

    tensors = []
    role_order = ["boxes", "scores", "mask_coefs", "protos"]

    # Handle split boxes
    xy_spec = spec.outputs.get("boxes_xy")
    wh_spec = spec.outputs.get("boxes_wh")
    if xy_spec and wh_spec:
        xy = raw_by_role["boxes_xy"]
        wh = raw_by_role["boxes_wh"]
        wh = requantize(wh, wh_spec.qn, wh_spec.offset,
                        spec.merged_box_qn, spec.merged_box_offset)
        xy = requantize(xy, xy_spec.qn, xy_spec.offset,
                        spec.merged_box_qn, spec.merged_box_offset)
        if xy.ndim == 2:
            xy = xy[np.newaxis]
        if wh.ndim == 2:
            wh = wh[np.newaxis]
        tensors.append(np.concatenate([xy, wh], axis=1))
    elif "boxes" in raw_by_role:
        tensors.append(raw_by_role["boxes"])

    for role in ["scores", "mask_coefs", "protos"]:
        if role in raw_by_role:
            tensors.append(raw_by_role[role])

    return tensors
