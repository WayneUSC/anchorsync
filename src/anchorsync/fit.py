"""Match anchor events between two clocks and fit an offset + drift model.

Clock model: ``t_ref = scale * t_local + offset``. ``scale = 1 + drift``; a drift of
50 ppm (a cheap crystal) accumulates 180 ms per hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


class SyncError(RuntimeError):
    """Raised when two streams cannot be matched."""


@dataclass
class ClockFit:
    stream: str
    reference: str
    scale: float
    offset: float
    drift_estimated: bool
    pairs: np.ndarray = field(repr=False)  # (M, 2) columns: local time, reference time
    n_events: int = 0
    n_ref_events: int = 0
    warnings: List[str] = field(default_factory=list)
    # Parameter covariance of (scale, offset) in s² units; offset-only fits have var only on offset.
    cov: np.ndarray = field(default_factory=lambda: np.full((2, 2), np.nan), repr=False)
    sigma_s: float = float("nan")  # per-anchor timing noise used for the covariance

    @property
    def drift_se_ppm(self) -> float:
        return float(np.sqrt(self.cov[0, 0]) * 1e6) if self.drift_estimated and np.isfinite(self.cov[0, 0]) else float("nan")

    def uncertainty_ms(self, t_local) -> np.ndarray:
        """1-σ uncertainty of the mapped reference time at local time(s) ``t_local``.

        Covers detection noise only. For offset-only fits it cannot include the
        unmodelled drift, which is usually the larger error.
        """
        t = np.asarray(t_local, dtype=float)
        if self.drift_estimated:
            var = self.cov[0, 0] * t ** 2 + 2 * self.cov[0, 1] * t + self.cov[1, 1]
        else:
            var = np.full_like(t, self.cov[1, 1])
        return np.sqrt(np.maximum(var, 0)) * 1e3

    @property
    def drift_ppm(self) -> float:
        return (self.scale - 1.0) * 1e6

    @property
    def n_matched(self) -> int:
        return int(len(self.pairs))

    @property
    def residuals_ms(self) -> np.ndarray:
        if len(self.pairs) == 0:
            return np.empty(0)
        return (self.pairs[:, 1] - self.to_ref(self.pairs[:, 0])) * 1e3

    @property
    def rms_ms(self) -> float:
        r = self.residuals_ms
        return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")

    @property
    def max_ms(self) -> float:
        r = self.residuals_ms
        return float(np.max(np.abs(r))) if len(r) else float("nan")

    @property
    def anchor_span_s(self) -> float:
        return float(np.ptp(self.pairs[:, 0])) if len(self.pairs) > 1 else 0.0

    def to_ref(self, t_local):
        return self.scale * np.asarray(t_local, dtype=float) + self.offset

    def to_local(self, t_ref):
        return (np.asarray(t_ref, dtype=float) - self.offset) / self.scale

    def to_dict(self) -> dict:
        return {
            "stream": self.stream,
            "reference": self.reference,
            "offset_s": self.offset,
            "scale": self.scale,
            "drift_ppm": self.drift_ppm,
            "drift_estimated": self.drift_estimated,
            "n_matched": self.n_matched,
            "n_events": self.n_events,
            "n_ref_events": self.n_ref_events,
            "anchor_span_s": self.anchor_span_s,
            "residual_rms_ms": self.rms_ms,
            "residual_max_ms": self.max_ms,
            "drift_se_ppm": self.drift_se_ppm,
            "sigma_ms": self.sigma_s * 1e3,
            "cov": self.cov.tolist(),
            "pairs": self.pairs.tolist(),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClockFit":
        return cls(stream=d["stream"], reference=d["reference"], scale=float(d["scale"]),
                   offset=float(d["offset_s"]), drift_estimated=bool(d["drift_estimated"]),
                   pairs=np.asarray(d.get("pairs", []), dtype=float).reshape(-1, 2),
                   n_events=int(d.get("n_events", 0)), n_ref_events=int(d.get("n_ref_events", 0)),
                   warnings=list(d.get("warnings", [])),
                   cov=np.asarray(d["cov"], dtype=float) if d.get("cov") is not None else np.full((2, 2), np.nan),
                   sigma_s=float(d["sigma_ms"]) / 1e3 if d.get("sigma_ms") is not None else float("nan"))


def identity_fit(name: str, n_events: int) -> ClockFit:
    return ClockFit(stream=name, reference=name, scale=1.0, offset=0.0, drift_estimated=True,
                    pairs=np.empty((0, 2)), n_events=n_events, n_ref_events=n_events)


def _match(ref: np.ndarray, src_mapped: np.ndarray, tol: float) -> np.ndarray:
    """One-to-one mutual-nearest matches within ``tol``. Returns index pairs (i_src, i_ref)."""
    if len(ref) == 0 or len(src_mapped) == 0:
        return np.empty((0, 2), dtype=int)
    pos = np.searchsorted(ref, src_mapped)
    lo = np.clip(pos - 1, 0, len(ref) - 1)
    hi = np.clip(pos, 0, len(ref) - 1)
    pick = np.where(np.abs(ref[lo] - src_mapped) <= np.abs(ref[hi] - src_mapped), lo, hi)
    dist = np.abs(ref[pick] - src_mapped)
    best_for_ref = {}
    for i, (j, d) in enumerate(zip(pick, dist)):
        if d <= tol and (j not in best_for_ref or d < best_for_ref[j][1]):
            best_for_ref[j] = (i, d)
    pairs = sorted((i, j) for j, (i, _) in best_for_ref.items())
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def _fit_line(x: np.ndarray, y: np.ndarray, estimate_scale: bool) -> Tuple[float, float]:
    if estimate_scale and len(x) >= 2 and np.ptp(x) > 0:
        xm = x.mean()
        a = float(np.sum((x - xm) * (y - y.mean())) / np.sum((x - xm) ** 2))
        b = float(y.mean() - a * xm)
        return a, b
    return 1.0, float(np.median(y - x))


def fit_clock(
    ref_times: np.ndarray,
    times: np.ndarray,
    name: str = "stream",
    reference: str = "reference",
    tol: float = 0.02,
    max_drift_ppm: float = 1000.0,
    min_anchor_span_s: float = 10.0,
    fixed_drift_ppm: Optional[float] = None,
    n_candidates: int = 25,
    noise_floor_s: float = 0.0,
) -> ClockFit:
    """Find the clock model mapping ``times`` (local) onto ``ref_times`` (reference).

    1. **Vote** over every pairing of a stream event with a reference event (candidate
       offset ``r_i - s_j``); count matches within a tolerance widened for the largest
       drift allowed (``max_drift_ppm``). Missing anchors and spurious events (footsteps,
       bumps) just lose the vote.
    2. **Refine** the best candidates ICP-style: match one-to-one, fit offset + drift by
       least squares, re-match with the tight tolerance ``tol``, repeat.
    3. Drift is only estimated when matched anchors span at least ``min_anchor_span_s``;
       otherwise it is fixed to zero (or ``fixed_drift_ppm``) with a warning — a single
       clap gives you an offset, not a clock rate.
    """
    ref = np.sort(np.asarray(ref_times, dtype=float))
    src = np.sort(np.asarray(times, dtype=float))
    if len(ref) == 0 or len(src) == 0:
        raise SyncError(f"{name}: no anchor events detected in {'reference' if len(ref) == 0 else 'stream'}")
    span_src = float(np.ptp(src)) if len(src) > 1 else 0.0
    tol_coarse = tol + max_drift_ppm * 1e-6 * max(span_src, 1.0)
    fixed_scale = None if fixed_drift_ppm is None else 1.0 + fixed_drift_ppm * 1e-6

    cands = (ref[:, None] - src[None, :]).ravel()
    cands = np.unique(np.round(cands / (tol / 2)) * (tol / 2))
    counts = np.empty(len(cands), dtype=int)
    for k in range(0, len(cands), 512):
        chunk = cands[k:k + 512]
        mapped = src[None, :] + chunk[:, None]  # (K, S)
        pos = np.searchsorted(ref, mapped)
        lo = np.clip(pos - 1, 0, len(ref) - 1)
        hi = np.clip(pos, 0, len(ref) - 1)
        d = np.minimum(np.abs(ref[lo] - mapped), np.abs(ref[hi] - mapped))
        counts[k:k + 512] = np.sum(d <= tol_coarse, axis=1)
    top = cands[np.argsort(-counts, kind="stable")[:n_candidates]]

    best = None
    for b0 in top:
        a, b = (fixed_scale or 1.0), b0
        pairs = np.empty((0, 2), dtype=int)
        for it in range(12):
            tol_it = tol_coarse if it == 0 else tol
            new_pairs = _match(ref, a * src + b, tol_it)
            if len(new_pairs) == 0:
                break
            x, y = src[new_pairs[:, 0]], ref[new_pairs[:, 1]]
            estimate = fixed_scale is None and np.ptp(x) >= min_anchor_span_s and len(x) >= 2
            a_new, b_new = _fit_line(x, y, estimate)
            if not estimate and fixed_scale is not None:
                a_new = fixed_scale
                b_new = float(np.median(y - a_new * x))
            if abs((a_new - 1) * 1e6) > max_drift_ppm:
                a_new, b_new = 1.0, float(np.median(y - x))
            converged = np.array_equal(new_pairs, pairs) and np.isclose(a_new, a) and np.isclose(b_new, b)
            a, b, pairs = a_new, b_new, new_pairs
            if converged:
                break
        if len(pairs) == 0:
            continue
        res = ref[pairs[:, 1]] - (a * src[pairs[:, 0]] + b)
        score = (len(pairs), -float(np.sqrt(np.mean(res ** 2))))
        if best is None or score > best[0]:
            best = (score, a, b, pairs)
    if best is None:
        raise SyncError(f"{name}: no anchor events could be matched to {reference} (tolerance {tol * 1e3:.0f} ms)")

    _, a, b, pairs = best
    pair_times = np.column_stack([src[pairs[:, 0]], ref[pairs[:, 1]]])
    span = float(np.ptp(pair_times[:, 0])) if len(pairs) > 1 else 0.0
    drift_estimated = fixed_scale is None and span >= min_anchor_span_s and len(pairs) >= 2
    fit = ClockFit(stream=name, reference=reference, scale=a, offset=b, drift_estimated=drift_estimated,
                   pairs=pair_times, n_events=len(src), n_ref_events=len(ref))
    # Parameter uncertainty. Per-anchor noise from the residuals, but never below the
    # quantization floor of the slower stream (a few anchors can agree by luck).
    x, y = pair_times[:, 0], pair_times[:, 1]
    n_par = 2 if drift_estimated else 1
    resid = y - (a * x + b)
    s2 = float(np.sum(resid ** 2) / (len(x) - n_par)) if len(x) > n_par else float("nan")
    s2 = max(s2, noise_floor_s ** 2) if np.isfinite(s2) else max(noise_floor_s ** 2, (tol / 2) ** 2)
    fit.sigma_s = float(np.sqrt(s2))
    cov = np.zeros((2, 2))
    if drift_estimated:
        X = np.column_stack([x, np.ones_like(x)])
        cov = s2 * np.linalg.inv(X.T @ X)
    else:
        cov[1, 1] = s2 / len(x)
    fit.cov = cov
    if not drift_estimated and fixed_scale is None:
        fit.warnings.append(
            f"matched anchors span only {span:.1f} s, so clock drift was not estimated (fixed at 0 ppm). "
            f"Add a second burst of taps at the end of the recording (dual-anchor)."
        )
    if len(pairs) < 3:
        fit.warnings.append(f"only {len(pairs)} anchor(s) matched; the fit has no redundancy to check itself")
    if fit.rms_ms > tol * 1e3 / 2:
        fit.warnings.append(f"residual RMS {fit.rms_ms:.1f} ms is large for the {tol * 1e3:.0f} ms tolerance")
    return fit
