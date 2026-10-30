"""Fit several streams onto one reference clock and resample them onto a shared timeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .events import Events, detect_events
from .fit import ClockFit, SyncError, fit_clock, identity_fit
from .health import TimingHealth, check_timing
from .stream import Stream, StreamError


@dataclass
class SyncResult:
    reference: str
    streams: Dict[str, Stream]
    events: Dict[str, Events]
    fits: Dict[str, ClockFit]
    health: Dict[str, TimingHealth]
    errors: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ mapping
    def to_ref(self, name: str, t_local) -> np.ndarray:
        """Map local times of stream ``name`` onto the reference clock."""
        if name not in self.fits:
            raise KeyError(f"{name} has no clock fit ({self.errors.get(name, 'unknown stream')})")
        return self.fits[name].to_ref(t_local)

    def uncertainty_ms(self, name: str) -> float:
        """Worst 1-σ mapping uncertainty over the stream's own time span (detection noise only)."""
        s = self.streams[name]
        return float(np.max(self.fits[name].uncertainty_ms(np.array([s.t.min(), s.t.max()]))))

    def aligned_times(self, name: str) -> np.ndarray:
        return self.to_ref(name, self.streams[name].t)

    def overlap(self, names: Optional[Sequence[str]] = None) -> tuple:
        names = list(names or self.fits)
        starts = [self.aligned_times(n).min() for n in names]
        ends = [self.aligned_times(n).max() for n in names]
        return float(max(starts)), float(min(ends))

    # ------------------------------------------------------------------ status
    def verdict(self, name: str) -> str:
        if name in self.errors:
            return "fail"
        fit = self.fits[name]
        if name == self.reference:
            return "reference"
        if fit.warnings or self.health[name].verdict != "ok":
            return "warn"
        return "ok"

    @property
    def ok(self) -> bool:
        return not self.errors and all(self.verdict(n) in ("ok", "reference") for n in self.streams)

    def summary(self) -> str:
        lines = [f"reference clock: {self.reference}"]
        hdr = (f"{'stream':<14}{'events':>7}{'matched':>9}{'offset (s)':>13}{'drift (ppm)':>17}{'rms (ms)':>10}"
               f"{'±1σ (ms)':>10}  status")
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for name in self.streams:
            n_ev = len(self.events[name]) if name in self.events else 0
            if name in self.errors:
                lines.append(f"{name:<14}{n_ev:>7}{'-':>9}{'-':>13}{'-':>17}{'-':>10}{'-':>10}  FAIL: {self.errors[name]}")
                continue
            f = self.fits[name]
            if name == self.reference:
                lines.append(f"{name:<14}{n_ev:>7}{'-':>9}{0.0:>13.4f}{'0':>17}{'-':>10}{'-':>10}  reference")
                continue
            drift = f"{f.drift_ppm:+.1f} ± {f.drift_se_ppm:.1f}" if f.drift_estimated else "not estimated"
            sig = f"{self.uncertainty_ms(name):.2f}" if f.drift_estimated else "—"
            lines.append(f"{name:<14}{n_ev:>7}{f.n_matched:>9}{f.offset:>+13.4f}{drift:>17}{f.rms_ms:>10.2f}"
                         f"{sig:>10}  {self.verdict(name)}")
        notes = []
        for name in self.streams:
            for w in (self.fits[name].warnings if name in self.fits else []):
                notes.append(f"  {name}: {w}")
            for issue in self.health[name].issues:
                notes.append(f"  {name}: timing — {issue}")
        if notes:
            lines.append("")
            lines.append("notes:")
            lines.extend(notes)
        return "\n".join(lines)

    # ------------------------------------------------------------------ resampling
    def resample(
        self,
        rate: float,
        method: str = "linear",
        start: Optional[float] = None,
        end: Optional[float] = None,
        names: Optional[Sequence[str]] = None,
        max_gap_s: Optional[float] = None,
    ):
        """All streams on one uniform reference timeline, as a pandas DataFrame.

        ``method``: ``linear`` (continuous signals), ``nearest`` or ``previous``
        (zero-order hold, for states and labels). The default range is the interval
        covered by every stream. Samples farther than ``max_gap_s`` (default: 2 periods
        of that stream) from a real sample become NaN instead of being interpolated
        across a dropout.
        """
        import pandas as pd

        if method not in ("linear", "nearest", "previous"):
            raise ValueError("method must be linear, nearest or previous")
        names = [n for n in (names or self.streams) if n in self.fits]
        lo, hi = self.overlap(names)
        start = lo if start is None else start
        end = hi if end is None else end
        if end <= start:
            raise ValueError("streams do not overlap on the reference clock")
        grid = start + np.arange(int(np.floor((end - start) * rate)) + 1) / rate
        out = {"t": grid}
        for n in names:
            s = self.streams[n]
            order = np.argsort(s.t, kind="stable")
            tr = self.fits[n].to_ref(s.t[order])
            vals = s.values[order]
            gap = max_gap_s if max_gap_s is not None else 2.0 / s.rate
            pos = np.searchsorted(tr, grid)
            lo_i = np.clip(pos - 1, 0, len(tr) - 1)
            hi_i = np.clip(pos, 0, len(tr) - 1)
            near = np.minimum(np.abs(tr[lo_i] - grid), np.abs(tr[hi_i] - grid))
            invalid = (near > gap) | (grid < tr[0]) | (grid > tr[-1])
            for c, cname in enumerate(s.channel_names):
                if method == "linear":
                    col = np.interp(grid, tr, vals[:, c])
                elif method == "nearest":
                    idx = np.where(np.abs(tr[lo_i] - grid) <= np.abs(tr[hi_i] - grid), lo_i, hi_i)
                    col = vals[idx, c]
                else:
                    idx = np.clip(np.searchsorted(tr, grid, side="right") - 1, 0, len(tr) - 1)
                    col = vals[idx, c]
                col = col.astype(float)
                col[invalid] = np.nan
                key = cname if cname.startswith(n) else f"{n}.{cname}"
                out[key] = col
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        from . import __version__

        return {
            "tool": "anchorsync",
            "version": __version__,
            "reference": self.reference,
            "settings": self.settings,
            "fits": {n: f.to_dict() for n, f in self.fits.items()},
            "errors": dict(self.errors),
            "health": {n: h.to_dict() for n, h in self.health.items()},
            "events": {n: e.times.tolist() for n, e in self.events.items()},
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_clean(self.to_dict()), fh, indent=2)

    def to_html(self, path: Optional[str] = None) -> str:
        from .report import render_html

        html = render_html(self)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def load_fits(path: str) -> Dict[str, ClockFit]:
    """Re-load clock fits saved with :meth:`SyncResult.save` (apply them to new data)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {n: ClockFit.from_dict(d) for n, d in data["fits"].items()}


def _noise_floor(stream: Stream) -> float:
    period = 1.0 / stream.rate
    return period / np.sqrt(12.0) if stream.kind == "video" else 0.1 * period


class SyncSession:
    """Collect streams, detect anchors, fit every clock onto the reference.

    >>> sess = SyncSession(reference="mic")
    >>> sess.add(Stream.from_wav("mic.wav", name="mic"))
    >>> sess.add(Stream.from_csv("imu.csv", name="imu", kind="accel"))
    >>> result = sess.fit()
    >>> print(result.summary())
    >>> df = result.resample(rate=100)
    """

    def __init__(self, reference: Optional[str] = None, tol: float = 0.02, max_drift_ppm: float = 1000.0,
                 min_anchor_span_s: float = 10.0):
        self.reference = reference
        self.tol = tol
        self.max_drift_ppm = max_drift_ppm
        self.min_anchor_span_s = min_anchor_span_s
        self.streams: Dict[str, Stream] = {}
        self.detect_kwargs: Dict[str, dict] = {}
        self.fixed_drift: Dict[str, float] = {}

    def add(self, stream: Stream, fixed_drift_ppm: Optional[float] = None, **detect_kwargs) -> "SyncSession":
        """Add a stream. ``detect_kwargs`` go to :func:`detect_events` (``min_strength``,
        ``refractory_s``, ``onset_fraction``, ...)."""
        if stream.name in self.streams:
            raise StreamError(f"duplicate stream name {stream.name!r}")
        self.streams[stream.name] = stream
        self.detect_kwargs[stream.name] = detect_kwargs
        if fixed_drift_ppm is not None:
            self.fixed_drift[stream.name] = fixed_drift_ppm
        return self

    def fit(self) -> SyncResult:
        if not self.streams:
            raise StreamError("no streams added")
        ref = self.reference or next(iter(self.streams))
        if ref not in self.streams:
            raise StreamError(f"reference {ref!r} is not one of the streams {list(self.streams)}")
        events = {n: detect_events(s, **self.detect_kwargs[n]) for n, s in self.streams.items()}
        health = {n: check_timing(s) for n, s in self.streams.items()}
        fits: Dict[str, ClockFit] = {ref: identity_fit(ref, len(events[ref]))}
        errors: Dict[str, str] = {}
        if len(events[ref]) == 0:
            errors[ref] = "no anchor events detected in the reference stream"
        for n in self.streams:
            if n == ref:
                continue
            # Event times cannot be more precise than the slower stream's sample period.
            tol = max(self.tol, 1.0 / min(self.streams[ref].rate, self.streams[n].rate))
            try:
                # Detection noise floor: one-frame spikes (video) cannot be interpolated below
                # the frame period; impulse responses spread over samples can.
                floor = max(_noise_floor(self.streams[ref]), _noise_floor(self.streams[n]))
                fits[n] = fit_clock(events[ref].times, events[n].times, name=n, reference=ref, tol=tol,
                                    max_drift_ppm=self.max_drift_ppm, min_anchor_span_s=self.min_anchor_span_s,
                                    fixed_drift_ppm=self.fixed_drift.get(n), noise_floor_s=floor)
            except SyncError as exc:
                errors[n] = str(exc)
        return SyncResult(reference=ref, streams=dict(self.streams), events=events, fits=fits, health=health,
                          errors=errors, settings={"tol_s": self.tol, "max_drift_ppm": self.max_drift_ppm,
                                                   "min_anchor_span_s": self.min_anchor_span_s})
