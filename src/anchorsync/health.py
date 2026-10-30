"""Timing health of a single stream: monotonicity, gaps, jitter, rate error."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from .stream import Stream


@dataclass
class TimingHealth:
    name: str
    n_samples: int
    duration_s: float
    median_rate_hz: float
    nominal_rate_hz: Optional[float]
    rate_error_ppm: Optional[float]
    non_monotonic: int
    duplicates: int
    gaps: int
    gap_total_s: float
    largest_gap_s: float
    jitter_ms: float
    jitter_robust_ms: float
    issues: List[str] = field(default_factory=list)
    verdict: str = "ok"

    def to_dict(self) -> dict:
        return asdict(self)


def check_timing(stream: Stream, gap_factor: float = 1.5, jitter_warn: float = 0.10,
                 rate_warn_ppm: float = 1000.0) -> TimingHealth:
    """Inspect the timestamps of ``stream``.

    * non-monotonic timestamps (time going backwards) → fail
    * duplicate timestamps → warn
    * gaps longer than ``gap_factor`` × the median period → warn (dropped samples)
    * jitter (std of the period, outliers excluded) above ``jitter_warn`` × period → warn
    * effective rate off the nominal rate by more than ``rate_warn_ppm`` → warn: times
      reconstructed as ``i / nominal_rate`` will drift
    """
    t = stream.t
    dt = np.diff(t)
    pos = dt[dt > 0]
    med = float(np.median(pos)) if len(pos) else float("nan")
    non_mono = int(np.sum(dt < 0))
    dups = int(np.sum(dt == 0))
    # Robust jitter first, so that heavy timestamp jitter is not mistaken for dropouts.
    mad_dt = float(1.4826 * np.median(np.abs(pos - med))) if len(pos) > 1 else 0.0
    gap_thr = med + max((gap_factor - 1.0) * med, 5.0 * mad_dt)
    gap_mask = dt > gap_thr
    gaps = int(gap_mask.sum())
    gap_total = float(np.sum(dt[gap_mask] - med)) if gaps else 0.0
    largest = float(dt.max() - med) if gaps else 0.0
    regular = pos[pos <= gap_thr]
    jitter = float(np.std(regular)) if len(regular) > 1 else 0.0
    mad = float(1.4826 * np.median(np.abs(regular - np.median(regular)))) if len(regular) > 1 else 0.0
    rate = 1.0 / med if med > 0 else float("nan")
    # Effective rate over the whole stream, counting dropped samples as missing.
    rate_err = None
    if stream.nominal_rate:
        effective = (len(t) - 1 + (gap_total / med if med > 0 else 0)) / max(t[-1] - t[0], 1e-12)
        rate_err = float((effective / stream.nominal_rate - 1.0) * 1e6)

    issues: List[str] = []
    verdict = "ok"
    if non_mono:
        issues.append(f"{non_mono} timestamps go backwards")
        verdict = "fail"
    if dups:
        issues.append(f"{dups} duplicate timestamps")
        verdict = verdict if verdict == "fail" else "warn"
    if gaps:
        issues.append(f"{gaps} gaps (> {gap_thr * 1e3:.1f} ms), ~{gap_total * 1e3:.0f} ms missing, "
                      f"largest {largest * 1e3:.1f} ms")
        verdict = verdict if verdict == "fail" else "warn"
    if med > 0 and jitter > jitter_warn * med:
        issues.append(f"jitter {jitter * 1e3:.2f} ms is {jitter / med:.0%} of the {med * 1e3:.2f} ms period")
        verdict = verdict if verdict == "fail" else "warn"
    if rate_err is not None and abs(rate_err) > rate_warn_ppm:
        issues.append(f"effective rate is {rate_err / 1e4:+.2f}% off the nominal {stream.nominal_rate:g} Hz; "
                      f"index/rate timestamps would drift {abs(rate_err) * 3.6:.0f} ms per hour")
        verdict = verdict if verdict == "fail" else "warn"
    return TimingHealth(
        name=stream.name,
        n_samples=len(t),
        duration_s=float(t[-1] - t[0]),
        median_rate_hz=float(rate),
        nominal_rate_hz=stream.nominal_rate,
        rate_error_ppm=rate_err,
        non_monotonic=non_mono,
        duplicates=dups,
        gaps=gaps,
        gap_total_s=gap_total,
        largest_gap_s=largest,
        jitter_ms=jitter * 1e3,
        jitter_robust_ms=mad * 1e3,
        issues=issues,
        verdict=verdict,
    )
