"""Shared timing statistics + summary table formatter.

Hoisted out of the three validator entry points to keep them in sync —
trimmed-mean computation, column widths, and the per-sub-stage indent
convention should be identical across ``onnx_reference``,
``reference``, and ``edgefirst``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def trimmed_stats(data: Sequence[float], trim_pct: float = 1.0) -> dict:
    """Return min/mean/max after symmetrically trimming ``trim_pct`` % outliers.

    The on-target benchmarks see occasional spikes from cgroup CPU
    pressure, dvproxy stalls, and GPU driver warmup — values that are
    not representative of steady-state pipeline cost. Trimming the top
    and bottom 1 % (one sample on each side of a 128-image run)
    suppresses those without losing the mean.

    Returns
    -------
    dict
        Keys: ``min``, ``mean``, ``max``, ``n``. ``n`` is the *trimmed*
        sample count, so the caller can report e.g. "126/128 samples".
    """
    if not data:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    a = np.asarray(data, dtype=np.float64)
    n = len(a)
    trim_n = max(1, int(n * trim_pct / 100))
    if n > 2 * trim_n:
        a = np.sort(a)[trim_n:-trim_n]
    return {
        "min": float(a.min()),
        "mean": float(a.mean()),
        "max": float(a.max()),
        "n": len(a),
    }


def fmt_row(label: str, stats: dict, indent: int = 0, label_width: int = 32) -> str:
    """Format one row of the per-stage timing table.

    Parameters
    ----------
    label
        Stage label (e.g. ``"image decode"``).
    stats
        Output of :func:`trimmed_stats`.
    indent
        Number of two-space indent levels (used for sub-stages such
        as ``dma input`` / ``npu compute`` / ``dma output`` nested
        under ``inference``).
    label_width
        Width of the label column. Default 32 — wide enough for the
        longest stage label including a single indent level.
    """
    prefix = "  " * indent
    return (
        f"{prefix}{label:<{label_width}s} "
        f"{stats['mean']:7.2f}  {stats['min']:7.2f}  {stats['max']:7.2f}"
    )


def summary_header(label_width: int = 32) -> str:
    """Return the timing-summary table header line, sized to ``label_width``."""
    return (
        f"{'Stage':<{label_width}s} {'Mean':>7s}  {'Min':>7s}  {'Max':>7s}  ms"
    )
