"""anchorsync: put every sensor stream on one clock using shared anchor events.

Tap the rig a few times at the start and a few times at the end of a recording.
anchorsync finds those taps in every stream (audio, IMU, force/tactile, camera),
matches them despite missed and spurious detections, fits each clock's offset *and*
drift onto a reference clock, checks timing health, and resamples everything onto one
timeline.

    from anchorsync import Stream, SyncSession

    sess = SyncSession(reference="mic")
    sess.add(Stream.from_wav("mic.wav", name="mic"))
    sess.add(Stream.from_csv("imu.csv", name="imu", kind="accel"))
    result = sess.fit()
    print(result.summary())
    frames = result.resample(rate=30)
"""

from .events import Events, activity, detect_events
from .fit import ClockFit, SyncError, fit_clock
from .health import TimingHealth, check_timing
from .session import SyncResult, SyncSession, load_fits
from .stream import Stream, StreamError

__version__ = "0.1.0"

__all__ = [
    "ClockFit", "Events", "Stream", "StreamError", "SyncError", "SyncResult", "SyncSession", "TimingHealth",
    "activity", "check_timing", "detect_events", "fit_clock", "load_fits", "__version__",
]
