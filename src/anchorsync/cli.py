"""Command-line interface: ``anchorsync fit | check | detect | demo``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional


from . import __version__
from .events import detect_events
from .fit import SyncError
from .health import check_timing
from .session import SyncSession
from .stream import Stream, StreamError

DETECT_KEYS = ("min_strength", "refractory_s", "onset_fraction", "peak_window_s", "smooth_s")
STREAM_KEYS = ("time", "columns", "time_unit", "nominal_rate")


def _load_config(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as fh:
        raw = fh.read()
    if ext == ".toml":
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - python < 3.11
            import tomli as tomllib  # type: ignore[no-redef]
        return tomllib.loads(raw.decode("utf-8"))
    return json.loads(raw.decode("utf-8"))


def session_from_config(path: str) -> SyncSession:
    """Build a :class:`SyncSession` from a JSON or TOML config file (see README)."""
    cfg = _load_config(path)
    base = os.path.dirname(os.path.abspath(path))
    sess = SyncSession(
        reference=cfg.get("reference"),
        tol=float(cfg.get("tolerance_ms", 20)) / 1e3,
        max_drift_ppm=float(cfg.get("max_drift_ppm", 1000)),
        min_anchor_span_s=float(cfg.get("min_anchor_span_s", 10)),
    )
    if not cfg.get("streams"):
        raise StreamError(f"{path}: no streams listed")
    for spec in cfg["streams"]:
        spec = dict(spec)
        p = spec.pop("path")
        p = p if os.path.isabs(p) else os.path.join(base, p)
        name = spec.pop("name", None)
        kind = spec.pop("kind", None)
        detect = {k: spec.pop(k) for k in DETECT_KEYS if k in spec}
        fixed = spec.pop("fixed_drift_ppm", None)
        load_kw = {k: spec.pop(k) for k in STREAM_KEYS if k in spec}
        video_kw = {k: spec.pop(k) for k in ("signal", "roi", "downsample") if k in spec}
        if spec:
            raise StreamError(f"{name or p}: unknown keys {sorted(spec)}")
        if video_kw:
            load_kw.update(video_kw)
        stream = Stream.load(p, name=name, kind=kind, **load_kw)
        sess.add(stream, fixed_drift_ppm=fixed, **detect)
    return sess


def cmd_fit(args: argparse.Namespace) -> int:
    result = session_from_config(args.config).fit()
    print(result.summary())
    if args.html:
        result.to_html(args.html)
        print(f"\nHTML report: {args.html}")
    if args.json:
        result.save(args.json)
        print(f"Fits: {args.json}")
    if args.resample:
        df = result.resample(args.resample, method=args.method)
        out = args.out or "aligned.csv"
        if out.endswith(".parquet"):
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        print(f"Aligned {len(df)} rows × {df.shape[1]} columns at {args.resample:g} Hz → {out}")
    return 0 if result.ok else 1


def cmd_check(args: argparse.Namespace) -> int:
    kw = {}
    if args.time:
        kw["time"] = args.time
    if args.rate:
        kw["nominal_rate"] = args.rate
    s = Stream.load(args.file, kind=args.kind, **kw)
    h = check_timing(s)
    print(f"{s.name}: {h.n_samples} samples over {h.duration_s:.3f} s, median rate {h.median_rate_hz:.3f} Hz, "
          f"jitter {h.jitter_ms:.3f} ms — {h.verdict.upper()}")
    for issue in h.issues:
        print(f"  - {issue}")
    return 0 if h.verdict == "ok" else 1


def cmd_detect(args: argparse.Namespace) -> int:
    kw = {"time": args.time} if args.time else {}
    s = Stream.load(args.file, kind=args.kind, **kw)
    ev = detect_events(s, min_strength=args.min_strength)
    print(f"{s.name}: {len(ev)} events (threshold {args.min_strength:g} MAD)")
    for t, k in zip(ev.times, ev.strengths):
        print(f"  {t:12.4f} s   strength {k:8.1f}")
    return 0 if len(ev) else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from .synthetic import alignment_error_ms, make_session

    streams, truth = make_session(dual_anchor=not args.single, seed=args.seed)
    sess = SyncSession(reference="mic")
    for s in streams:
        sess.add(s)
    result = sess.fit()
    print(result.summary())
    print("\nground truth (synthetic):")
    for n, (scale, offset) in truth.items():
        f = result.fits.get(n)
        if f is None:
            continue
        err = alignment_error_ms(f, (scale, offset), result.streams[n].t)
        print(f"  {n:<8} true offset {offset:+.4f} s, true drift {(scale - 1) * 1e6:+.1f} ppm → "
              f"worst alignment error {err:.2f} ms")
    if args.html:
        result.to_html(args.html)
        print(f"\nHTML report: {args.html}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anchorsync", description="Put every sensor stream on one clock using shared anchor events.")
    p.add_argument("--version", action="version", version=f"anchorsync {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="fit all streams listed in a JSON/TOML config onto the reference clock")
    f.add_argument("config")
    f.add_argument("--html", help="write an HTML report")
    f.add_argument("--json", help="save the clock fits (re-usable with anchorsync.load_fits)")
    f.add_argument("--resample", type=float, metavar="HZ", help="resample all streams onto one timeline at HZ")
    f.add_argument("--method", default="linear", choices=["linear", "nearest", "previous"])
    f.add_argument("--out", help="aligned output (.csv or .parquet; default aligned.csv)")
    f.set_defaults(func=cmd_fit)

    c = sub.add_parser("check", help="timing health of one stream (gaps, jitter, rate error)")
    c.add_argument("file")
    c.add_argument("--time", help="time column (CSV)")
    c.add_argument("--rate", type=float, help="nominal sampling rate in Hz")
    c.add_argument("--kind", default="generic", choices=["audio", "accel", "video", "generic"])
    c.set_defaults(func=cmd_check)

    d = sub.add_parser("detect", help="list anchor events detected in one stream")
    d.add_argument("file")
    d.add_argument("--kind", default="generic", choices=["audio", "accel", "video", "generic"])
    d.add_argument("--time", help="time column (CSV)")
    d.add_argument("--min-strength", type=float, default=12.0)
    d.set_defaults(func=cmd_detect)

    m = sub.add_parser("demo", help="synthetic 4-sensor session with known offsets and drifts")
    m.add_argument("--single", action="store_true", help="only one anchor burst (shows why drift needs two)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--html")
    m.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (StreamError, SyncError, ValueError, ImportError, FileNotFoundError, KeyError) as exc:
        print(f"anchorsync: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
