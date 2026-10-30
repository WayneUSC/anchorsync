"""Self-contained HTML sync report."""

from __future__ import annotations

import html
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

_CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--accent:#3b5bdb;--soft:#c9c8c1;
--ok:#2b8a3e;--warn:#c77700;--fail:#d6336c;--okbg:#ebf7ee;--warnbg:#fff4e0;--failbg:#fdecf2;--refbg:#edf0fd}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;
--muted:#a3a29b;--line:#33332f;--accent:#7b93f0;--soft:#55554f;--ok:#58c27a;--warn:#f0a73c;--fail:#f06595;
--okbg:#1d2e22;--warnbg:#33281a;--failbg:#351c27;--refbg:#22263a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 16px 64px}.tool{font-size:13px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}
h1{font-size:28px;margin:4px 0 8px;font-weight:650}.facts{color:var(--muted);font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:16px 0;overflow-x:auto}
.card h2{font-size:18px;margin:0 0 4px}.q{color:var(--muted);margin:0 0 12px;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--muted);font-weight:600;font-size:13px;padding:6px}td{padding:7px 6px;border-top:1px solid var(--line);vertical-align:top}
td.n{text-align:right}th.n{text-align:right}
.pill{font-size:12px;padding:2px 10px;border-radius:999px;font-weight:700;letter-spacing:.04em;white-space:nowrap}
.pill.ok{background:var(--okbg);color:var(--ok)}.pill.warn{background:var(--warnbg);color:var(--warn)}
.pill.fail{background:var(--failbg);color:var(--fail)}.pill.reference{background:var(--refbg);color:var(--accent)}
.badge{display:inline-block;margin-top:12px;font-weight:700;font-size:14px;padding:6px 14px;border-radius:999px}
ul.notes{margin:8px 0 0;padding-left:18px;color:var(--muted);font-size:14px}
svg{max-width:100%;height:auto}.legend{font-size:13px;color:var(--muted)}
footer{color:var(--muted);font-size:13px;margin-top:32px}
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _raster(rows: List[Tuple[str, np.ndarray, np.ndarray]], x0: float, x1: float, title: str, width: int = 900) -> str:
    """Event ticks per stream. rows: (name, times, matched_mask)."""
    lw, rp, top, rh = 110, 20, 26, 26
    h = top + rh * len(rows) + 28
    span = (x1 - x0) or 1.0

    def X(t: float) -> float:
        return lw + (t - x0) / span * (width - lw - rp)

    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" font-size="12" role="img" aria-label="{_e(title)}">',
           f'<text x="{lw}" y="16" fill="var(--muted)">{_e(title)}</text>']
    for k in range(6):
        t = x0 + span * k / 5
        x = X(t)
        out.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{h - 24}" stroke="var(--line)"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 8}" fill="var(--muted)" text-anchor="middle">{t:.2f} s</text>')
    for i, (name, times, matched) in enumerate(rows):
        y = top + i * rh
        out.append(f'<text x="{lw - 8}" y="{y + 15}" fill="var(--ink)" text-anchor="end">{_e(name)}</text>')
        out.append(f'<line x1="{lw}" y1="{y + 11}" x2="{width - rp}" y2="{y + 11}" stroke="var(--line)"/>')
        for t, m in zip(times, matched):
            if x0 <= t <= x1:
                x = X(t)
                color = "var(--accent)" if m else "var(--soft)"
                out.append(f'<line x1="{x:.1f}" y1="{y + 2}" x2="{x:.1f}" y2="{y + 20}" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>')
    out.append("</svg>")
    return "".join(out)


def _residual_plot(series: Dict[str, np.ndarray], width: int = 900) -> str:
    """series[name] = (M, 2) array of (reference time, residual ms)."""
    if not series:
        return ""
    lw, rp, top, h = 60, 20, 16, 220
    allv = np.concatenate([s for s in series.values() if len(s)]) if any(len(s) for s in series.values()) else np.zeros((1, 2))
    tmin, tmax = float(allv[:, 0].min()), float(allv[:, 0].max())
    if tmax <= tmin:
        tmax = tmin + 1
    lim = float(max(1.0, np.abs(allv[:, 1]).max() * 1.15))

    def X(t):
        return lw + (t - tmin) / (tmax - tmin) * (width - lw - rp)

    def Y(v):
        return top + (lim - v) / (2 * lim) * (h - top - 30)

    palette = ["var(--accent)", "var(--warn)", "var(--ok)", "var(--fail)", "var(--muted)"]
    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" font-size="12" role="img" aria-label="residuals">']
    for v in (-lim, -lim / 2, 0, lim / 2, lim):
        y = Y(v)
        sw = ' stroke-width="1.5"' if v == 0 else ""
        out.append(f'<line x1="{lw}" y1="{y:.1f}" x2="{width - rp}" y2="{y:.1f}" stroke="var(--line)"{sw}/>')
        out.append(f'<text x="{lw - 6}" y="{y + 4:.1f}" fill="var(--muted)" text-anchor="end">{v:+.1f}</text>')
    out.append(f'<text x="{lw}" y="{h - 6}" fill="var(--muted)">reference time {tmin:.1f}–{tmax:.1f} s · residual (ms)</text>')
    lx = width - rp
    for k, (name, s) in enumerate(series.items()):
        c = palette[k % len(palette)]
        for t, r in s:
            out.append(f'<circle cx="{X(t):.1f}" cy="{Y(r):.1f}" r="4" fill="{c}" opacity=".85"/>')
        lx -= 12 + 7.5 * len(name) + 12
        out.append(f'<circle cx="{lx:.1f}" cy="{h - 10}" r="4" fill="{c}"/><text x="{lx + 8:.1f}" y="{h - 6}" fill="var(--muted)">{_e(name)}</text>')
    out.append("</svg>")
    return "".join(out)


def _bursts(times: np.ndarray, gap: float = 3.0) -> List[Tuple[float, float]]:
    if len(times) == 0:
        return []
    times = np.sort(times)
    groups, start = [], times[0]
    for a, b in zip(times[:-1], times[1:]):
        if b - a > gap:
            groups.append((start, a))
            start = b
    groups.append((start, times[-1]))
    return groups


def render_html(result) -> str:
    from . import __version__

    ref = result.reference
    names = list(result.streams)
    status_all = "ok" if result.ok else ("fail" if result.errors else "warn")
    parts = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             f"<title>Sync report — {_e(ref)}</title><style>{_CSS}</style></head><body><main>",
             '<div class="tool">anchorsync · one clock for every sensor</div>',
             f"<h1>{len(names)} streams on the <em>{_e(ref)}</em> clock</h1>",
             f'<div class="facts">tolerance {result.settings.get("tol_s", 0) * 1e3:.0f} ms · max drift searched '
             f'±{result.settings.get("max_drift_ppm", 0):g} ppm · drift needs anchors ≥ '
             f'{result.settings.get("min_anchor_span_s", 0):g} s apart</div>',
             f'<span class="badge pill {status_all}">{status_all.upper()}</span>']

    # clock table
    parts.append('<section class="card"><h2>Clock fits</h2><p class="q">t<sub>ref</sub> = (1 + drift) · t<sub>local</sub> + offset. '
                 "Residuals are how far each matched anchor sits from the fitted line; ±1σ is the worst mapping "
                 "uncertainty over the stream from detection noise alone.</p><table><tr>"
                 '<th>stream</th><th>kind</th><th class="n">rate (Hz)</th><th class="n">events</th><th class="n">matched</th>'
                 '<th class="n">offset (s)</th><th class="n">drift (ppm)</th><th class="n">residual rms (ms)</th><th class="n">±1σ (ms)</th><th>status</th></tr>')
    for n in names:
        s = result.streams[n]
        ev = result.events.get(n)
        v = result.verdict(n)
        if n in result.errors:
            parts.append(f"<tr><td>{_e(n)}</td><td>{_e(s.kind)}</td><td class=\"n\">{s.rate:.1f}</td><td class=\"n\">{len(ev) if ev else 0}</td>"
                         f'<td colspan="5">{_e(result.errors[n])}</td><td><span class="pill fail">FAIL</span></td></tr>')
            continue
        f = result.fits[n]
        if n == ref:
            cells = ["—", "0", "0", "—", "—"]
        else:
            cells = [str(f.n_matched), f"{f.offset:+.4f}",
                     f"{f.drift_ppm:+.1f} ± {f.drift_se_ppm:.1f}" if f.drift_estimated else "not estimated",
                     f"{f.rms_ms:.2f}", f"{result.uncertainty_ms(n):.2f}" if f.drift_estimated else "—"]
        parts.append(f"<tr><td><strong>{_e(n)}</strong></td><td>{_e(s.kind)}</td><td class=\"n\">{s.rate:.1f}</td>"
                     f"<td class=\"n\">{len(ev)}</td>" + "".join(f'<td class="n">{_e(c)}</td>' for c in cells)
                     + f'<td><span class="pill {v}">{v.upper()}</span></td></tr>')
    parts.append("</table>")
    notes = [f"<li><strong>{_e(n)}</strong>: {_e(w)}</li>" for n in names if n in result.fits for w in result.fits[n].warnings]
    if notes:
        parts.append('<ul class="notes">' + "".join(notes) + "</ul>")
    parts.append("</section>")

    # rasters around anchor bursts
    ref_matched = set()
    for n in names:
        if n in result.fits and n != ref:
            for _, tr in result.fits[n].pairs:
                ref_matched.add(round(float(tr), 6))
    bursts = _bursts(np.array(sorted(ref_matched)))
    if bursts:
        parts.append('<section class="card"><h2>Anchors before and after alignment</h2>'
                     '<p class="q">Detected events per stream around each anchor burst. Blue ticks were matched to the '
                     "reference, grey ticks are unmatched (distractors or missed counterparts). Before: raw local clocks. "
                     "After: mapped onto the reference clock.</p>")
        for b0, b1 in bursts:
            x0, x1 = b0 - 0.6, b1 + 0.6
            raw_lo, raw_hi = x0, x1
            before, after = [], []
            for n in names:
                if n not in result.events:
                    continue
                ev = result.events[n].times
                if n in result.fits:
                    f = result.fits[n]
                    if n == ref:
                        mask = np.array([round(float(t), 6) in ref_matched for t in ev])
                    else:
                        in_burst = (f.pairs[:, 1] >= b0 - 1e-6) & (f.pairs[:, 1] <= b1 + 1e-6)
                        local = f.pairs[in_burst, 0]
                        if len(local):
                            raw_lo, raw_hi = min(raw_lo, local.min() - 0.6), max(raw_hi, local.max() + 0.6)
                        matched_local = set(np.round(f.pairs[:, 0], 6))
                        mask = np.array([round(float(t), 6) in matched_local for t in ev])
                    mapped = f.to_ref(ev)
                else:
                    mask = np.zeros(len(ev), bool)
                    mapped = ev
                before.append((n, ev, mask))
                after.append((n, mapped, mask))
            parts.append(_raster(before, raw_lo, raw_hi, f"before — each stream on its own clock ({raw_lo:.1f}–{raw_hi:.1f} s)"))
            parts.append(_raster(after, x0, x1, f"after — all on the {ref} clock ({x0:.1f}–{x1:.1f} s)"))
        parts.append("</section>")

    # residuals
    series = {n: np.column_stack([result.fits[n].pairs[:, 1], result.fits[n].residuals_ms])
              for n in names if n in result.fits and n != ref and result.fits[n].n_matched}
    if series:
        parts.append('<section class="card"><h2>Residuals</h2><p class="q">Matched anchor time minus fitted time. '
                     "A tilt across the recording means drift was not (or not fully) corrected; scatter is detection "
                     "precision, which is bounded by each stream's sample period.</p>")
        parts.append(_residual_plot(series))
        parts.append("</section>")

    # timing health
    parts.append('<section class="card"><h2>Timing health</h2><p class="q">Problems in each stream\'s own timestamps.</p><table><tr>'
                 '<th>stream</th><th class="n">samples</th><th class="n">duration (s)</th><th class="n">median rate</th>'
                 '<th class="n">jitter (ms)</th><th class="n">gaps</th><th>issues</th></tr>')
    for n in names:
        hh = result.health[n]
        pill = {"ok": "ok", "warn": "warn", "fail": "fail"}[hh.verdict]
        parts.append(f"<tr><td><strong>{_e(n)}</strong> <span class=\"pill {pill}\">{hh.verdict.upper()}</span></td>"
                     f'<td class="n">{hh.n_samples}</td><td class="n">{hh.duration_s:.2f}</td><td class="n">{hh.median_rate_hz:.2f}</td>'
                     f'<td class="n">{hh.jitter_ms:.3f}</td><td class="n">{hh.gaps}</td><td>{_e("; ".join(hh.issues) or "—")}</td></tr>')
    parts.append("</table></section>")
    parts.append(f"<footer>anchorsync {__version__}. Fits can be re-applied to new data with <code>anchorsync.load_fits()</code>."
                 "<details><summary>Fit parameters (JSON)</summary><pre style=\"white-space:pre-wrap;font-size:12px\">"
                 + _e(json.dumps({n: {k: v for k, v in f.to_dict().items() if k != 'pairs'} for n, f in result.fits.items()},
                                 indent=2, default=float))
                 + "</pre></details></footer></main></body></html>")
    return "".join(parts)
