"""One self-contained HTML file: charts inlined, no assets, no server."""

from __future__ import annotations

import base64
from pathlib import Path

CSS = """
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px;
margin:40px auto;padding:0 20px;color:#16181d}
h1{margin-bottom:4px}h2{margin-top:38px;border-bottom:1px solid #e3e6ea;
padding-bottom:6px}
.sub{color:#5b6472;margin-top:0}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
.tile{flex:1;min-width:150px;background:#f6f8fa;border:1px solid #e3e6ea;
border-radius:9px;padding:14px 16px}
.tile .v{font-size:25px;font-weight:650;letter-spacing:-.5px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:#5b6472;margin-top:3px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f6f8fa;font-weight:600}
td.good{color:#17803d}td.bad{color:#b4232c}
img{width:100%;margin:14px 0;border:1px solid #e3e6ea;border-radius:7px}
.note{background:#fff8e6;border-left:3px solid #d9a441;padding:11px 15px;
margin:18px 0;font-size:14px}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _img(path: Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{Path(path).stem}">'


def build(*, census, share, stint_counts, fitted, robust, naive, quadratic,
          fwl_gap, recovery_rows, circuit_rows, stint_rows, threshold_rows,
          implied_fuel, seasons, figures, spreads, condition_cut,
          out_path: Path) -> Path:
    low, high = fitted.interval("age")

    tiles = [
        ("laps kept", f"{census['kept']:,}"),
        ("car-races", f"{fitted.groups:,}"),
        ("degradation, s/lap", f"{fitted['age']:+.4f}"),
        ("naive estimate", f"{naive['age']:+.4f}"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in tiles)

    census_html = "".join(
        f"<tr><td>{key}</td><td>{value:,}</td>"
        f"<td>{value / census['total']:.2%}</td></tr>"
        for key, value in census.items() if key != "total")

    estimate_html = "".join(
        f"<tr><td>{label}</td>"
        f'<td class="{klass}">{fit_value:+.4f}</td>'
        f"<td>{error:.4f}</td><td>{lo:+.4f} to {hi:+.4f}</td>"
        f"<td>{lap_value}</td></tr>"
        for label, fit_value, error, lo, hi, lap_value, klass in (
            ("within, with race lap", fitted["age"], fitted.error("age"),
             low, high, f"{fitted['lap']:+.4f}", "good"),
            ("within, Huber", robust["age"], robust.error("age"),
             *robust.interval("age"), f"{robust['lap']:+.4f}", "good"),
            ("naive, age only", naive["age"], naive.error("age"),
             *naive.interval("age"), "—", "bad"),
        ))

    recovery_html = "".join(
        f"<tr><td>{r['stops']}</td><td>{r['degradation']:+.3f}</td>"
        + (f'<td class="bad" colspan="2">refused: not identified</td>'
           if r["degradation_fit"] != r["degradation_fit"] else
           f'<td class="good">{r["degradation_fit"]:+.4f}</td>'
           f'<td class="good">{r["fuel_fit"]:+.4f}</td>')
        + f'<td class="bad">{r["naive"]:+.4f}</td></tr>'
        for r in recovery_rows)

    circuit_html = "".join(
        f"<tr><td>{r['circuit']}</td><td>{r['value']:+.4f}</td>"
        f"<td>{r['low']:+.4f} to {r['high']:+.4f}</td>"
        f"<td>{r['race_lap']:+.4f}</td><td>{r['laps']:,}</td></tr>"
        for r in sorted(circuit_rows, key=lambda row: -row["value"]))

    stint_html = "".join(
        f"<tr><td>stint {r['stint']}</td><td>{r['value']:+.4f}</td>"
        f"<td>{r['low']:+.4f} to {r['high']:+.4f}</td>"
        f"<td>{r['laps']:,}</td></tr>" for r in stint_rows)

    cuts = sorted({r["cut"] for r in threshold_rows})
    threshold_head = "".join(
        f"<th>{'no cut' if c >= 1 else f'cut {c:.2f}'}</th>" for c in cuts)
    threshold_html = ""
    for ratio in sorted({r["ratio"] for r in threshold_rows}):
        cells = "".join(
            f'<td class="{"bad" if r["cut"] >= 1 else "good"}">'
            f"{r['value']:+.4f}</td>"
            for r in sorted((row for row in threshold_rows
                             if row["ratio"] == ratio),
                            key=lambda row: row["cut"]))
        threshold_html += f"<tr><td>safety car {ratio:.2f}</td>{cells}</tr>"

    condition_html = "".join(
        f"<tr><td>{r['circuit']}</td><td>{r['spread']:.3f}</td>"
        f'<td class="{"bad" if r["spread"] > condition_cut else "good"}">'
        f"{'excluded' if r['spread'] > condition_cut else 'kept'}</td></tr>"
        for r in spreads)
    inside = [r["spread"] for r in spreads if r["spread"] <= condition_cut]
    outside = [r["spread"] for r in spreads if r["spread"] > condition_cut]

    with_rule = [r["value"] for r in threshold_rows if r["cut"] < 1.0]
    without = [r["value"] for r in threshold_rows if r["cut"] >= 1.0]
    spread = max(with_rule) - min(with_rule)
    positive = sum(1 for r in circuit_rows if r["low"] > 0)
    stint_mix = ", ".join(f"{k} stint{'s' if k != 1 else ''}: {v}"
                          for k, v in sorted(stint_counts.items()))

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Stint degradation</title><style>{CSS}</style></head><body>
<h1>Stint degradation</h1>
<p class="sub">Tyre degradation from {census['total']:,} F1 lap times,
{"-".join(str(s) for s in seasons)}, with the fuel and track-evolution
effect estimated rather than assumed.</p>

<div class="tiles">{tiles_html}</div>

<div class="note"><b>The whole problem in one line.</b> Within a stint, tyre
age and race lap move together one for one, so leaving the race lap out does
not drop the fuel effect — it hides it inside the degradation coefficient.
The naive estimate below is therefore not a noisier version of the right
answer; it is the right answer <i>plus</i> the fuel effect, which here is
large enough to reverse the sign.</div>

<h2>Which laps are used</h2>
{_img(figures['census'])}
<table><tr><th>rule</th><th>laps</th><th>share</th></tr>{census_html}</table>
<p>{share:.1%} of the laps that survive belong to a car that made at least
one stop — the only cars that can separate tyre age from race lap. Stints
per car-race: {stint_mix}.</p>
{_img(figures['identification'])}

<h2>The estimate</h2>
{_img(figures['degradation_curve'])}
<table><tr><th>estimator</th><th>degradation (s/lap)</th><th>std. error</th>
<th>95% interval</th><th>race-lap term</th></tr>{estimate_html}</table>
<p>Standard errors are clustered on the car-race: {fitted.groups:,} clusters,
{fitted.n:,} laps. Within R&sup2; {fitted.r_squared:.4f}, residual
&sigma; {fitted.sigma:.3f} s. Curvature in age is
{quadratic['age squared']:+.5f} s per lap squared
(p {quadratic.p_value('age squared'):.3f}).</p>
{_img(figures['estimates'])}
<p>The fixed effects are absorbed by the within transformation rather than
by explicit dummies. The two agree to {fwl_gap:.1e} s, which is
Frisch-Waugh-Lovell doing what it says.</p>

<h2>Does the estimator recover a degradation it was given?</h2>
{_img(figures['recovery'])}
<table><tr><th>stops per car</th><th>true degradation</th>
<th>fitted degradation</th><th>fitted fuel</th><th>naive</th></tr>
{recovery_html}</table>
<p>With no stops nothing identifies the split and the fitter refuses. The
naive estimator never refuses, at any stop count, and is wrong by about the
size of the fuel effect at all of them.</p>

<h2>By circuit</h2>
{_img(figures['circuits'])}
<table><tr><th>circuit</th><th>degradation</th><th>95% interval</th>
<th>race-lap term</th><th>laps</th></tr>{circuit_html}</table>
<p>{positive} of {len(circuit_rows)} circuits have an interval entirely
above zero.</p>

<h2>By stint position</h2>
{_img(figures['stints'])}
<table><tr><th>stint</th><th>degradation</th><th>95% interval</th>
<th>laps</th></tr>{stint_html}</table>
<p>One model, with tyre age interacted with the stint number. This is as
close to a compound breakdown as the feed allows: it never records what was
fitted.</p>

<h2>The races that are not a race against the clock</h2>
{_img(figures['conditions'])}
<table><tr><th>race</th><th>condition drift</th><th></th></tr>
{condition_html}</table>
<p>A drying track makes the whole field faster lap after lap, which is a
race-lap effect no fuel load can produce, and it swamps the model. The
measure is the spread of the field median over the race. It runs from
{min(inside):.3f} to {max(inside):.3f} for {len(inside)} races and then
jumps to {min(outside):.3f}; the cut at {condition_cut:.2f} sits in that
empty band, so anywhere in it removes the same races.</p>

<h2>Do either of the cleaning constants matter?</h2>
{_img(figures['neutralisation'])}
<table><tr><th></th>{threshold_head}</tr>{threshold_html}</table>
<p>With the condition rule on, the estimate moves by {spread:.4f} s per lap
across every combination of the two constants. With it off, the safety-car
threshold alone moves it by {max(without) - min(without):.4f} — more than
the effect being measured. The fragile constant was never the safety-car
rule; it was the two races the rule was being asked to paper over.</p>

<h2>Traffic, and why slow laps are not deleted</h2>
{_img(figures['contamination'])}
<p>Being held up adds time to a lap and never subtracts it, so the residuals
are skewed one way. The robust fit down-weights the far tail symmetrically;
removing slow laps as a class would remove the end of every stint, which is
where the degradation is.</p>

<p class="sub">Generated by <code>run.py</code>. Lap times from the Jolpica
mirror of the Ergast schema.</p>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
