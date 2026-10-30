"""A synthetic recording session with known clock offsets and drifts.

Four sensors record the same scene on their own clocks:

``mic``       audio, 16 kHz — the reference. Also hears footsteps (distractor events).
``imu``       3-axis accelerometer, 100 Hz, host timestamps with 1 ms jitter, +80 ppm drift.
``tactile``   16-taxel pressure array, 500 Hz, 0.5 % dropped samples, −35 ppm drift.
``camera``    per-frame motion energy, 30 fps, 2 ms frame jitter, +15 ppm drift.

Someone taps the rig three times at the start and three times at the end (irregular
spacing, so the pattern cannot alias). The IMU also feels a couple of handling bumps
nobody else sees.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .stream import Stream

TRUTH = {
    # name: (scale, offset) with t_ref = scale * t_local + offset
    "imu": (1 + 80e-6, 1.2373),
    "tactile": (1 - 35e-6, -0.4218),
    "camera": (1 + 15e-6, 2.0310),
}


def _bursts(t_local: np.ndarray, onsets: np.ndarray, amp: float, decay_s: float, rng: np.random.Generator,
            noise_carrier: bool = True) -> np.ndarray:
    x = np.zeros_like(t_local)
    for on in onsets:
        dt = t_local - on
        m = (dt >= 0) & (dt < 8 * decay_s)
        env = amp * np.exp(-dt[m] / decay_s)
        x[m] += env * (rng.normal(size=m.sum()) if noise_carrier else 1.0)
    return x


def make_session(duration: float = 120.0, seed: int = 0, dual_anchor: bool = True) -> Tuple[List[Stream], Dict[str, Tuple[float, float]]]:
    """Return (streams, truth). ``truth[name] = (scale, offset)`` maps local → reference."""
    rng = np.random.default_rng(seed)
    taps = np.array([3.0, 3.45, 4.2])
    if dual_anchor:
        taps = np.r_[taps, duration - np.array([6.0, 5.4, 4.9])]
    footsteps = np.sort(rng.uniform(10, duration - 10, 12))
    bumps = np.sort(rng.uniform(20, duration - 20, 2))

    streams: List[Stream] = []

    # mic — reference clock
    fs = 16000.0
    t = np.arange(int(duration * fs)) / fs
    audio = 0.002 * rng.normal(size=len(t))
    audio += _bursts(t, taps, 0.6, 0.008, rng)
    audio += _bursts(t, footsteps, 0.08, 0.03, rng)
    streams.append(Stream.from_rate("mic", audio, fs, kind="audio"))

    def local_times(name: str, rate: float, jitter_s: float = 0.0, drop: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        scale, offset = TRUTH[name]
        # local clock covers the reference interval [max(0, offset), duration]
        t0 = max(0.0, -offset / scale)
        t1 = (duration - offset) / scale
        tl = np.arange(t0, t1, 1.0 / rate)
        keep = rng.random(len(tl)) >= drop
        tl = tl[keep]
        stamp = tl + (rng.normal(0, jitter_s, len(tl)) if jitter_s else 0.0)
        return tl, stamp

    def to_local(name: str, tau: np.ndarray) -> np.ndarray:
        scale, offset = TRUTH[name]
        return (tau - offset) / scale

    # imu
    tl, stamp = local_times("imu", 100.0, jitter_s=0.001)
    acc = 0.02 * rng.normal(size=(len(tl), 3))
    acc[:, 2] += 9.81
    for ax, amp in ((0, 2.5), (1, 1.5), (2, 4.0)):
        acc[:, ax] += _bursts(tl, to_local("imu", taps), amp, 0.02, rng, noise_carrier=False)
        acc[:, ax] += _bursts(tl, to_local("imu", bumps), amp * 0.8, 0.03, rng, noise_carrier=False)
    streams.append(Stream("imu", stamp, acc, kind="accel", nominal_rate=100.0,
                          channel_names=["imu.ax", "imu.ay", "imu.az"]))

    # tactile
    tl, stamp = local_times("tactile", 500.0, drop=0.005)
    tac = 0.01 * rng.normal(size=(len(tl), 16)) + rng.uniform(0.1, 0.2, 16)
    gains = np.zeros(16)
    gains[[5, 6, 9, 10]] = rng.uniform(0.5, 1.0, 4)
    pulse = _bursts(tl, to_local("tactile", taps), 1.0, 0.015, rng, noise_carrier=False)
    tac += pulse[:, None] * gains[None, :]
    streams.append(Stream("tactile", stamp, tac, kind="generic", nominal_rate=500.0,
                          channel_names=[f"tactile.taxel{i:02d}" for i in range(16)]))

    # camera motion energy
    tl, stamp = local_times("camera", 30.0, jitter_s=0.002)
    energy = 0.5 + 0.05 * rng.normal(size=len(tl))
    for on in to_local("camera", taps):
        k = np.searchsorted(tl, on)
        if k < len(tl):
            energy[k:k + 2] += np.array([6.0, 2.0])[: len(energy[k:k + 2])]
    streams.append(Stream("camera", stamp, energy, kind="video", nominal_rate=30.0,
                          channel_names=["camera.motion"]))
    return streams, dict(TRUTH)


def alignment_error_ms(fit, truth: Tuple[float, float], t_local: np.ndarray) -> float:
    """Worst-case |fitted − true| reference time over the stream's local span, in ms."""
    scale, offset = truth
    ts = np.linspace(np.min(t_local), np.max(t_local), 200)
    return float(np.max(np.abs(fit.to_ref(ts) - (scale * ts + offset))) * 1e3)
