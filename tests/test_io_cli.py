import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from anchorsync import Stream
from anchorsync.cli import main
from anchorsync.synthetic import make_session


@pytest.fixture()
def recording(tmp_path):
    """Write the synthetic session to disk in the formats people actually have."""
    streams, truth = make_session(seed=2)
    s = {x.name: x for x in streams}
    wavfile.write(tmp_path / "mic.wav", 16000, (s["mic"].values[:, 0] * 30000).astype(np.int16))
    # Phyphox-style export
    pd.DataFrame({"Time (s)": s["imu"].t, "Acceleration x (m/s^2)": s["imu"].values[:, 0],
                  "Acceleration y (m/s^2)": s["imu"].values[:, 1],
                  "Acceleration z (m/s^2)": s["imu"].values[:, 2]}).to_csv(tmp_path / "imu.csv", index=False)
    # tactile with millisecond timestamps
    tac = pd.DataFrame(s["tactile"].values, columns=[f"p{i}" for i in range(16)])
    tac.insert(0, "t_ms", s["tactile"].t * 1e3)
    tac.to_csv(tmp_path / "tactile.tsv", sep="\t", index=False)
    np.savez(tmp_path / "camera.npz", t=s["camera"].t, values=s["camera"].values, rate=30.0)
    cfg = {
        "reference": "mic",
        "tolerance_ms": 20,
        "streams": [
            {"name": "mic", "path": "mic.wav"},
            {"name": "imu", "path": "imu.csv", "kind": "accel", "nominal_rate": 100},
            {"name": "tactile", "path": "tactile.tsv", "time": "t_ms", "time_unit": "ms"},
            {"name": "camera", "path": "camera.npz", "kind": "video"},
        ],
    }
    (tmp_path / "sync.json").write_text(json.dumps(cfg))
    return tmp_path, truth


def test_loaders(recording):
    path, _ = recording
    imu = Stream.load(str(path / "imu.csv"), kind="accel")
    assert imu.values.shape[1] == 3 and imu.channel_names[0].startswith("Acceleration")
    tac = Stream.from_csv(str(path / "tactile.tsv"), time="t_ms", time_unit="ms")
    assert tac.rate == pytest.approx(500, rel=0.01)
    mic = Stream.load(str(path / "mic.wav"))
    assert mic.kind == "audio" and mic.rate == pytest.approx(16000)


def test_cli_fit_end_to_end(recording, capsys):
    path, truth = recording
    out = path / "aligned.csv"
    code = main(["fit", str(path / "sync.json"), "--html", str(path / "r.html"), "--json", str(path / "fits.json"),
                 "--resample", "50", "--out", str(out)])
    text = capsys.readouterr().out
    assert code in (0, 1)  # 1 = some warnings (jitter, dropped samples) — expected here
    assert "imu" in text and "camera" in text
    fits = json.loads((path / "fits.json").read_text())["fits"]
    assert abs(fits["imu"]["drift_ppm"] - (truth["imu"][0] - 1) * 1e6) < 25
    assert abs(fits["imu"]["offset_s"] - truth["imu"][1]) < 0.003
    df = pd.read_csv(out)
    assert "imu.Acceleration z (m/s^2)" in df.columns and len(df) > 1000


def test_cli_check_and_detect(recording, capsys):
    path, _ = recording
    assert main(["check", str(path / "imu.csv"), "--rate", "100"]) in (0, 1)
    assert "samples" in capsys.readouterr().out
    assert main(["detect", str(path / "imu.csv"), "--kind", "accel"]) == 0
    assert "events" in capsys.readouterr().out
    assert main(["fit", str(path / "missing.json")]) == 2


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    assert "worst alignment error" in capsys.readouterr().out


def test_toml_config(recording):
    path, _ = recording
    try:
        import tomllib  # noqa: F401
    except ImportError:
        pytest.importorskip("tomli")
    (path / "sync.toml").write_text(
        'reference = "mic"\n[[streams]]\nname = "mic"\npath = "mic.wav"\n'
        '[[streams]]\nname = "imu"\npath = "imu.csv"\nkind = "accel"\n'
    )
    assert main(["fit", str(path / "sync.toml")]) in (0, 1)


def test_video_motion_energy(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from anchorsync.video import motion_energy

    path = str(tmp_path / "clap.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(90):
        frame = np.full((48, 64, 3), 40, np.uint8)
        if i in (30, 60):  # a flash / hand in view for one frame
            frame[:] = 220
        w.write(frame)
    w.release()
    s = motion_energy(path)
    assert s.kind == "video" and len(s) == 90
    from anchorsync import detect_events

    ev = detect_events(s, refractory_s=0.1)
    assert len(ev) == 2
    assert np.allclose(ev.times, [29.5 / 30, 59.5 / 30], atol=0.02)
    b = motion_energy(path, signal="brightness")
    assert len(detect_events(b, refractory_s=0.1)) == 2
