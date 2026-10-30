import numpy as np
import pytest

from anchorsync import Stream, StreamError, SyncError, SyncSession, check_timing, detect_events, fit_clock, load_fits
from anchorsync.synthetic import TRUTH, alignment_error_ms, make_session


# ---------------------------------------------------------------- fit_clock


def test_fit_recovers_offset_and_drift_with_distractors():
    rng = np.random.default_rng(0)
    ref = np.array([5.0, 5.4, 6.3, 300.0, 300.5, 301.4])
    a_true, b_true = 1 + 120e-6, -3.217
    local = (ref - b_true) / a_true + rng.normal(0, 0.0005, len(ref))
    local = np.delete(local, 4)                      # one anchor missed in the stream
    local = np.r_[local, rng.uniform(0, 300, 5)]      # spurious bumps in the stream
    ref_all = np.r_[ref, rng.uniform(10, 290, 8)]     # footsteps in the reference
    fit = fit_clock(ref_all, local, tol=0.01)
    assert fit.n_matched == 5
    assert abs(fit.drift_ppm - 120) < 10
    assert abs(fit.offset - b_true) < 0.002
    assert fit.drift_estimated and np.isfinite(fit.drift_se_ppm)


def test_single_burst_gives_offset_only_with_warning():
    ref = np.array([2.0, 2.5, 3.4])
    fit = fit_clock(ref, ref - 1.5, tol=0.01)
    assert not fit.drift_estimated
    assert fit.scale == 1.0 and fit.offset == pytest.approx(1.5, abs=1e-9)
    assert any("dual-anchor" in w for w in fit.warnings)


def test_fixed_drift_is_respected():
    ref = np.array([2.0, 2.5, 3.4])
    fit = fit_clock(ref, (ref - 1.0) / (1 + 50e-6), tol=0.01, fixed_drift_ppm=50)
    assert fit.drift_ppm == pytest.approx(50)
    assert fit.offset == pytest.approx(1.0, abs=1e-6)


def test_fit_errors_on_empty():
    with pytest.raises(SyncError):
        fit_clock(np.array([]), np.array([1.0]))


def test_large_offset_beyond_tolerance_is_found():
    ref = np.array([10.0, 10.7, 11.1, 200.0, 200.9])
    fit = fit_clock(ref, ref + 3600.25, tol=0.005)  # stream clock is an hour ahead
    assert fit.n_matched == 5 and fit.offset == pytest.approx(-3600.25, abs=1e-6)


# ---------------------------------------------------------------- events


@pytest.mark.parametrize("kind", ["audio", "accel", "generic"])
def test_detection_onset_is_amplitude_invariant(kind):
    rate = 8000.0 if kind == "audio" else 500.0
    t = np.arange(int(4 * rate)) / rate
    rng = np.random.default_rng(1)
    x = 0.001 * rng.normal(size=(len(t), 2))
    onsets = [1.0, 2.5]
    for on, amp in zip(onsets, [1.0, 0.1]):  # loud tap, soft tap
        m = t >= on
        x[m, :] += (amp * np.exp(-(t[m] - on) / 0.01))[:, None] * (rng.normal(size=(m.sum(), 2)) if kind == "audio" else 1)
    ev = detect_events(Stream("s", t, x, kind=kind))
    assert len(ev) == 2
    assert np.all(np.abs(ev.times - onsets) < 3.0 / rate + 0.001)


def test_video_kind_uses_level():
    t = np.arange(300) / 30.0
    x = np.full(300, 0.5)
    x[[60, 200]] += 5.0
    ev = detect_events(Stream("cam", t, x, kind="video"))
    assert len(ev) == 2
    # midpoint of the frame interval before the spike frame
    assert ev.times[0] == pytest.approx((59 + 0.5) / 30.0, abs=1e-6)


# ---------------------------------------------------------------- health


def test_timing_health_flags_problems():
    t = np.arange(1000) / 100.0
    t = np.delete(t, np.arange(500, 520))  # 200 ms dropout
    h = check_timing(Stream("s", t, np.zeros(len(t)), nominal_rate=100.0))
    assert h.gaps == 1 and h.verdict == "warn" and h.gap_total_s == pytest.approx(0.2, abs=0.011)
    t2 = np.arange(100) / 100.0
    t2[50] = t2[48]
    h2 = check_timing(Stream("s", t2, np.zeros(100)))
    assert h2.non_monotonic >= 1 and h2.verdict == "fail"
    t3 = np.arange(5000) / 99.0  # declared 100 Hz, really 99 Hz
    h3 = check_timing(Stream("s", t3, np.zeros(5000), nominal_rate=100.0))
    assert h3.rate_error_ppm == pytest.approx(-10000, rel=0.01) and h3.verdict == "warn"


def test_stream_validation():
    with pytest.raises(StreamError):
        Stream("s", np.arange(3), np.zeros(3))
    with pytest.raises(StreamError):
        Stream("s", np.arange(10), np.zeros((9, 2)))
    with pytest.raises(StreamError):
        Stream("s", np.arange(10), np.zeros(10), kind="lidar")


# ---------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def dual():
    streams, truth = make_session(seed=0)
    sess = SyncSession(reference="mic")
    for s in streams:
        sess.add(s)
    return sess.fit(), truth


def test_session_dual_anchor_accuracy(dual):
    result, truth = dual
    assert not result.errors
    for name in ("imu", "tactile"):
        f = result.fits[name]
        assert f.drift_estimated
        assert abs(f.drift_ppm - (truth[name][0] - 1) * 1e6) < 3 * f.drift_se_ppm + 5
        assert alignment_error_ms(f, truth[name], result.streams[name].t) < 2.0
    # camera precision is bounded by the 33 ms frame period
    cam = result.fits["camera"]
    assert alignment_error_ms(cam, truth["camera"], result.streams["camera"].t) < 4 * result.uncertainty_ms("camera")


def test_single_anchor_cannot_correct_drift():
    streams, truth = make_session(seed=0, dual_anchor=False)
    sess = SyncSession(reference="mic")
    for s in streams:
        sess.add(s)
    result = sess.fit()
    imu = result.fits["imu"]
    assert not imu.drift_estimated
    # 80 ppm over ~118 s ≈ 9.5 ms at the far end of the recording
    assert alignment_error_ms(imu, truth["imu"], result.streams["imu"].t) > 8.0


def test_resample_and_roundtrip(dual, tmp_path):
    result, _ = dual
    df = result.resample(100.0)
    lo, hi = result.overlap()
    assert df["t"].iloc[0] == pytest.approx(lo) and df["t"].iloc[-1] <= hi
    assert {"imu.ax", "tactile.taxel05", "camera.motion", "mic"} <= set(df.columns)
    assert df["imu.az"].mean() == pytest.approx(9.81, abs=0.05)
    path = tmp_path / "fits.json"
    result.save(str(path))
    fits = load_fits(str(path))
    assert fits["imu"].offset == pytest.approx(result.fits["imu"].offset)
    assert np.allclose(fits["imu"].cov, result.fits["imu"].cov)
    html = result.to_html(str(tmp_path / "r.html"))
    assert "<svg" in html and "Clock fits" in html


def test_resample_masks_dropouts():
    t = np.r_[np.arange(0, 5, 0.01), np.arange(6, 10, 0.01)]
    ref = Stream.from_rate("ref", np.zeros(1000), 100.0)
    s = Stream("s", t, np.ones(len(t)))
    from anchorsync.fit import identity_fit
    from anchorsync.session import SyncResult

    res = SyncResult(reference="ref", streams={"ref": ref, "s": s}, events={},
                     fits={"ref": identity_fit("ref", 0), "s": identity_fit("s", 0)},
                     health={"ref": check_timing(ref), "s": check_timing(s)})
    df = res.resample(100.0)
    inside_gap = (df["t"] > 5.05) & (df["t"] < 5.95)
    assert df.loc[inside_gap, "s"].isna().all()
    assert df.loc[df["t"] < 4.9, "s"].notna().all()


def test_truth_table_is_consistent():
    assert set(TRUTH) == {"imu", "tactile", "camera"}
