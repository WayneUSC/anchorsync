"""Turn a video file into a per-frame activity stream (needs ``opencv-python``)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .stream import Stream, StreamError


def motion_energy(
    path: str,
    name: Optional[str] = None,
    signal: str = "motion",
    roi: Optional[Tuple[int, int, int, int]] = None,
    downsample: int = 4,
    use_container_timestamps: bool = True,
) -> Stream:
    """Per-frame activity from a video.

    ``signal="motion"``: mean absolute difference between consecutive grayscale frames
    (a hand clap or a tap in view). ``signal="brightness"``: mean brightness (a flash or
    an LED blink as anchor). ``roi=(x, y, w, h)`` restricts it to where the anchor
    happens, which makes detection far more robust in a busy scene.

    Frame times come from the container (``CAP_PROP_POS_MSEC``) when they increase,
    otherwise from frame index / fps.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("video input needs OpenCV: pip install 'anchorsync[video]'") from exc
    if signal not in ("motion", "brightness"):
        raise ValueError("signal must be 'motion' or 'brightness'")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise StreamError(f"{path}: cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    times, values = [], []
    prev = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y + h, x:x + w]
        gray = gray[::downsample, ::downsample].astype(np.float32)
        if signal == "brightness":
            values.append(float(gray.mean()))
        else:
            values.append(0.0 if prev is None else float(np.mean(np.abs(gray - prev))))
            prev = gray
        times.append(ts if use_container_timestamps else np.nan)
        idx += 1
    cap.release()
    if idx < 4:
        raise StreamError(f"{path}: fewer than 4 frames decoded")
    t = np.asarray(times, dtype=float)
    if not use_container_timestamps or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        if fps <= 0:
            raise StreamError(f"{path}: no usable frame timestamps and no fps in the container")
        t = np.arange(idx) / fps
    if signal == "motion":
        values[0] = values[1]
    return Stream(name=name or os.path.splitext(os.path.basename(path))[0], t=t, values=np.asarray(values),
                  kind="video", nominal_rate=fps or None, channel_names=[f"{signal}"], source=path)
