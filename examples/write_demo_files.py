"""Write the synthetic demo session to disk (mic.wav, imu.csv, tactile.tsv, camera.npz, sync.json)
so you can try the command-line workflow end to end:

    python examples/write_demo_files.py demo_session
    anchorsync fit demo_session/sync.json --html sync.html --resample 30 --out aligned.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import wavfile

from anchorsync.synthetic import make_session

out = sys.argv[1] if len(sys.argv) > 1 else "demo_session"
os.makedirs(out, exist_ok=True)
streams = {s.name: s for s in make_session(seed=0)[0]}
wavfile.write(os.path.join(out, "mic.wav"), 16000, (streams["mic"].values[:, 0] * 30000).astype(np.int16))
imu = streams["imu"]
pd.DataFrame({"Time (s)": imu.t, **{f"Acceleration {a} (m/s^2)": imu.values[:, i] for i, a in enumerate("xyz")}}).to_csv(
    os.path.join(out, "imu.csv"), index=False)
tac = streams["tactile"]
df = pd.DataFrame(tac.values, columns=[f"p{i}" for i in range(16)])
df.insert(0, "t_ms", tac.t * 1e3)
df.to_csv(os.path.join(out, "tactile.tsv"), sep="\t", index=False)
cam = streams["camera"]
np.savez(os.path.join(out, "camera.npz"), t=cam.t, values=cam.values, rate=30.0)
cfg = {"reference": "mic", "streams": [
    {"name": "mic", "path": "mic.wav"},
    {"name": "imu", "path": "imu.csv", "kind": "accel", "nominal_rate": 100},
    {"name": "tactile", "path": "tactile.tsv", "time": "t_ms", "time_unit": "ms"},
    {"name": "camera", "path": "camera.npz", "kind": "video"},
]}
with open(os.path.join(out, "sync.json"), "w") as fh:
    json.dump(cfg, fh, indent=2)
print(f"wrote {out}/ — now run: anchorsync fit {out}/sync.json --html sync.html")
