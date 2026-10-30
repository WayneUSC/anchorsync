"""Turn a stream into an activity envelope and detect impulsive anchor events in it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from .stream import Stream


@dataclass
class Events:
    """Detected anchor events on a stream's local clock."""

    times: np.ndarray       # onset times (s, local clock)
    strengths: np.ndarray   # peak height in robust noise units (MADs above the median)
    threshold: float        # detection threshold in envelope units
    env_t: np.ndarray       # envelope time base
    env: np.ndarray         # activity envelope

    def __len__(self) -> int:
        return len(self.times)


def _robust_scale(x: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    if mad <= 0:
        # Noise-free signal (synthetic, quantized or idle sensor): fall back to a tiny
        # fraction of the range so that real spikes still stand out.
        rng = float(np.max(x) - med)
        mad = 1e-3 * rng if rng > 0 else 1.0
    return med, mad


def activity(stream: Stream, smooth_s: Optional[float] = None, max_rate: float = 4000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Non-negative activity envelope that rises sharply at a tap, clap or knock.

    * audio: |first difference| of the channel mean, max-pooled to ≤ ``max_rate`` Hz,
      then a short moving average
    * accel: magnitude of the jerk vector (first difference across all axes)
    * video: the per-frame signal itself, as deviation from its median
    * generic: sum over channels of |first difference|, each channel scaled by its own
      robust noise level so a 16-taxel array does not drown a single pressure channel
    """
    order = np.argsort(stream.t, kind="stable")
    t = stream.t[order]
    v = stream.values[order]
    if stream.kind == "video":
        x = np.abs(v.mean(axis=1) - np.median(v.mean(axis=1)))
        return t, x
    if stream.kind == "audio":
        x = np.abs(np.diff(v.mean(axis=1), prepend=v[0].mean()))
    elif stream.kind == "accel":
        d = np.diff(v, axis=0, prepend=v[:1])
        x = np.sqrt(np.sum(d ** 2, axis=1))
    else:
        d = np.abs(np.diff(v, axis=0, prepend=v[:1]))
        scale = np.array([_robust_scale(d[:, c])[1] for c in range(d.shape[1])])
        scale[scale <= 0] = 1.0
        x = (d / scale).sum(axis=1)
    rate = stream.rate
    if rate > max_rate:
        k = int(np.ceil(rate / max_rate))
        n = (len(x) // k) * k
        x = x[:n].reshape(-1, k).max(axis=1)
        t = t[:n].reshape(-1, k)[:, 0]
        rate = rate / k
    if smooth_s is None:
        smooth_s = 0.002 if stream.kind == "audio" else 0.0
    w = int(round(smooth_s * rate))
    if w > 1:
        x = uniform_filter1d(x, w, mode="nearest")
    return t, x


def detect_events(
    stream: Stream,
    min_strength: float = 12.0,
    refractory_s: float = 0.15,
    onset_fraction: Optional[float] = None,
    peak_window_s: float = 0.05,
    smooth_s: Optional[float] = None,
) -> Events:
    """Detect impulsive events (taps, claps, knocks) in ``stream``.

    An event starts where the envelope crosses ``median + min_strength · MAD``. Its onset
    time is then refined with a *constant-fraction* rule: the moment the envelope first
    rises above ``onset_fraction`` of the event's own peak height, linearly
    interpolated between samples. Unlike a fixed threshold, this does not make loud
    taps look earlier than soft ones. Events closer than ``refractory_s`` are merged.

    ``onset_fraction`` defaults to 0.25, and to 0.5 for ``video`` streams: a frame-
    difference spike appears in the first frame after the event, so the midpoint of the
    preceding frame interval is the unbiased estimate.
    """
    if onset_fraction is None:
        onset_fraction = 0.5 if stream.kind == "video" else 0.25
    t, e = activity(stream, smooth_s=smooth_s)
    med, mad = _robust_scale(e)
    thr = med + min_strength * mad
    above = e > thr
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    if len(starts) == 0:
        return Events(np.empty(0), np.empty(0), thr, t, e)
    rate = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 1.0
    pw = max(1, int(round(peak_window_s * rate)))
    times, strengths = [], []
    last = -np.inf
    for s in starts:
        if t[s] - last < refractory_s:
            continue
        end = min(len(e), s + pw + 1)
        p = s + int(np.argmax(e[s:end]))
        peak = e[p]
        level = med + onset_fraction * (peak - med)
        i = p
        while i > 0 and e[i - 1] > level:
            i -= 1
        if i == 0:
            onset = t[0]
        else:
            # interpolate between i-1 (below) and i (above)
            e0, e1 = e[i - 1], e[i]
            frac = 0.0 if e1 == e0 else (level - e0) / (e1 - e0)
            onset = t[i - 1] + frac * (t[i] - t[i - 1])
        times.append(onset)
        strengths.append((peak - med) / mad)
        last = t[s]
    return Events(np.asarray(times), np.asarray(strengths), thr, t, e)
