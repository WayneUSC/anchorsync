"""A timestamped sensor stream and loaders for common formats."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

KINDS = ("audio", "accel", "video", "generic")


class StreamError(ValueError):
    """Raised when a stream cannot be loaded or is malformed."""


@dataclass
class Stream:
    """Samples of one sensor on its own clock.

    Parameters
    ----------
    name:
        Identifier used in reports and in the aligned output.
    t:
        Sample times in seconds on the sensor's *local* clock, shape (N,).
    values:
        Samples, shape (N,) or (N, C).
    kind:
        How anchor events show up: ``"audio"`` (sound pressure), ``"accel"`` (an
        accelerometer/IMU — taps show up as jerk), ``"video"`` (a per-frame activity
        signal such as motion energy or brightness — used as is) or ``"generic"``
        (anything else: force, pressure, tactile arrays — changes are detected).
    nominal_rate:
        Declared sampling rate in Hz, if known. Used by the timing health check.
    """

    name: str
    t: np.ndarray
    values: np.ndarray
    kind: str = "generic"
    nominal_rate: Optional[float] = None
    channel_names: List[str] = field(default_factory=list)
    source: Optional[str] = None

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).ravel()
        v = np.asarray(self.values, dtype=float)
        if v.ndim == 1:
            v = v[:, None]
        if v.ndim != 2 or v.shape[0] != self.t.shape[0]:
            raise StreamError(f"{self.name}: values must be (N,) or (N, C) with N={len(self.t)}, got {v.shape}")
        if len(self.t) < 4:
            raise StreamError(f"{self.name}: need at least 4 samples")
        if not np.all(np.isfinite(self.t)):
            raise StreamError(f"{self.name}: timestamps contain NaN/inf")
        if self.kind not in KINDS:
            raise StreamError(f"{self.name}: kind must be one of {KINDS}, got {self.kind!r}")
        self.values = v
        if not self.channel_names:
            self.channel_names = [f"{self.name}_{i}" if v.shape[1] > 1 else self.name for i in range(v.shape[1])]

    # ------------------------------------------------------------------ properties
    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def rate(self) -> float:
        """Median sampling rate estimated from the timestamps."""
        dt = np.diff(self.t)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else float("nan")

    def __len__(self) -> int:
        return len(self.t)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_rate(cls, name: str, values, rate: float, t0: float = 0.0, kind: str = "generic", **kw) -> "Stream":
        """Uniformly sampled values starting at ``t0`` on the local clock."""
        values = np.asarray(values)
        t = t0 + np.arange(values.shape[0]) / float(rate)
        return cls(name=name, t=t, values=values, kind=kind, nominal_rate=rate, **kw)

    @classmethod
    def from_wav(cls, path: str, name: Optional[str] = None, channels: Optional[Sequence[int]] = None) -> "Stream":
        """A WAV file. Time starts at 0 on the recorder's clock."""
        from scipy.io import wavfile

        rate, data = wavfile.read(path)
        data = np.asarray(data)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(float) / float(np.iinfo(data.dtype).max)
        if data.ndim == 1:
            data = data[:, None]
        if channels is not None:
            data = data[:, list(channels)]
        return cls.from_rate(name or _stem(path), data, rate, kind="audio", source=path)

    @classmethod
    def from_csv(
        cls,
        path: str,
        time: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        kind: str = "generic",
        time_unit: str = "s",
        nominal_rate: Optional[float] = None,
    ) -> "Stream":
        """A CSV/TSV with a time column. Phyphox exports work out of the box.

        ``time`` defaults to the first column whose name starts with "time" or "t",
        else the first column. ``time_unit`` is one of s, ms, us, ns.
        """
        import pandas as pd

        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] < 2:
            raise StreamError(f"{path}: need a time column and at least one value column")
        if time is None:
            lowered = [str(c).strip().lower() for c in df.columns]
            cands = [c for c, low in zip(df.columns, lowered) if low.startswith("time") or low in ("t", "timestamp", "ts")]
            time = cands[0] if cands else df.columns[0]
        if time not in df.columns:
            raise StreamError(f"{path}: time column {time!r} not found (columns: {list(df.columns)})")
        if columns is None:
            columns = [c for c in df.columns if c != time and pd.api.types.is_numeric_dtype(df[c])]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise StreamError(f"{path}: columns not found: {missing}")
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[time_unit]
        df = df.dropna(subset=[time])
        return cls(
            name=name or _stem(path),
            t=df[time].to_numpy(dtype=float) * scale,
            values=df[list(columns)].to_numpy(dtype=float),
            kind=kind,
            nominal_rate=nominal_rate,
            channel_names=[str(c) for c in columns],
            source=path,
        )

    @classmethod
    def from_npz(cls, path: str, name: Optional[str] = None, kind: str = "generic",
                 time_key: str = "t", values_key: str = "values") -> "Stream":
        data = np.load(path, allow_pickle=False)
        if time_key not in data or values_key not in data:
            raise StreamError(f"{path}: needs arrays {time_key!r} and {values_key!r} (have {data.files})")
        rate = float(data["rate"]) if "rate" in data else None
        return cls(name=name or _stem(path), t=data[time_key], values=data[values_key], kind=kind,
                   nominal_rate=rate, source=path)

    @classmethod
    def load(cls, path: str, name: Optional[str] = None, kind: Optional[str] = None, **kw) -> "Stream":
        """Load by extension: .wav, .csv/.tsv/.txt, .npz, video (.mp4/.avi/.mov/.mkv)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            return cls.from_wav(path, name=name, **kw)
        if ext in (".csv", ".tsv", ".txt"):
            return cls.from_csv(path, name=name, kind=kind or "generic", **kw)
        if ext == ".npz":
            return cls.from_npz(path, name=name, kind=kind or "generic", **kw)
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            from .video import motion_energy

            return motion_energy(path, name=name, **kw)
        raise StreamError(f"{path}: unsupported extension {ext!r}")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
