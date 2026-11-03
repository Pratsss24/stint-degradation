"""Build the report as a printable PDF.

`paper.py` owns the page furniture; everything here is this project's own
argument, in the order the argument runs.
"""

from __future__ import annotations

from pathlib import Path

from paper import Paper

SOURCE = ("Lap times, pit stops and results from the Jolpica mirror of the "
          "Ergast schema.")


def _census(census: dict) -> tuple[list[str], list[list[str]]]:
    rows = [[key, f"{value:,}", f"{value / census['total']:.2%}"]
            for key, value in census.items() if key != "total"]
    return (["rule", "laps", "share"], rows)


def _estimates(fitted, robust, naive) -> tuple[list[str], list[list[str]]]:
    rows = []
    for label, fit, lap in (("within, with race lap", fitted,
                             f"{fitted['lap']:+.4f}"),
                            ("within, Huber", robust,
                             f"{robust['lap']:+.4f}"),
                            ("naive, age only", naive, "—")):
        low, high = fit.interval("age")
        rows.append([label, f"{fit['age']:+.4f}", f"{fit.error('age'):.4f}",
                     f"{low:+.4f} to {high:+.4f}", lap])
    return (["estimator", "degradation, s/lap", "std. error",
             "95% interval", "race-lap term"], rows)


def _recovery(rows) -> tuple[list[str], list[list[str]]]:
    out = []
    for one in rows:
        identified = one["degradation_fit"] == one["degradation_fit"]
        out.append([str(one["stops"]), f"{one['degradation']:+.3f}",
                    f"{one['degradation_fit']:+.4f}" if identified
                    else "refused: not identified",
                    f"{one['fuel_fit']:+.4f}" if identified else "—",
                    f"{one['naive']:+.4f}"])
    return (["stops per car", "true degradation", "fitted degradation",
             "fitted fuel", "naive"], out)


def _circuits(rows) -> tuple[list[str], list[list[str]]]:
    out = [[one["circuit"], f"{one['value']:+.4f}",
            f"{one['low']:+.4f} to {one['high']:+.4f}",
            f"{one['race_lap']:+.4f}", f"{one['laps']:,}"]
           for one in sorted(rows, key=lambda row: -row["value"])]
    return (["circuit", "degradation", "95% interval", "race-lap term",
             "laps"], out)


def _stints(rows) -> tuple[list[str], list[list[str]]]:
    out = [[f"stint {one['stint']}", f"{one['value']:+.4f}",
            f"{one['low']:+.4f} to {one['high']:+.4f}", f"{one['laps']:,}"]
           for one in rows]
    return (["stint", "degradation", "95% interval", "laps"], out)


def _conditions(spreads, cut: float) -> tuple[list[str], list[list[str]]]:
    out = [[one["circuit"], f"{one['spread']:.3f}",
            "excluded" if one["spread"] > cut else "kept"]
           for one in spreads]
    return (["race", "condition drift", "verdict"], out)


def _thresholds(rows) -> tuple[list[str], list[list[str]]]:
    cuts = sorted({one["cut"] for one in rows})
    head = [""] + ["no cut" if one >= 1 else f"cut {one:.2f}" for one in cuts]
    out = []
    for ratio in sorted({one["ratio"] for one in rows}):
        cells = [f"safety car {ratio:.2f}"]
        here = sorted((row for row in rows if row["ratio"] == ratio),
                      key=lambda row: row["cut"])
        cells += [f"{one['value']:+.4f}" for one in here]
        out.append(cells)
    return (head, out)


def build(*, census, share, stint_counts, fitted, robust, naive, quadratic,
          fwl_gap, recovery_rows, circuit_rows, stint_rows, threshold_rows,
          implied_fuel, seasons, figures, spreads, condition_cut,
          out_path: Path) -> Path:
    low, high = fitted.interval("age")
    inside = [one["spread"] for one in spreads
              if one["spread"] <= condition_cut]
    outside = [one["spread"] for one in spreads
               if one["spread"] > condition_cut]
    with_rule = [one["value"] for one in threshold_rows if one["cut"] < 1.0]
    without = [one["value"] for one in threshold_rows if one["cut"] >= 1.0]
    spread = max(with_rule) - min(with_rule)
    positive = sum(1 for one in circuit_rows if one["low"] > 0)
    stint_mix = ", ".join(f"{key} stint{'s' if key != 1 else ''}: {value}"
                          for key, value in sorted(stint_counts.items()))
    out = Path(out_path)
    out = out.with_name("report.pdf") if out.suffix != ".pdf" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    with Paper(out, "Stint degradation",
               f"Tyre degradation from {census['total']:,} F1 lap times, "
               f"{'-'.join(str(one) for one in seasons)}, with the fuel and "
               f"track-evolution effect estimated rather than assumed.") \
            as paper:

        paper.cover(
            headline=f"Degradation is {fitted['age']:+.4f} s a lap "
                     f"[{low:+.4f}, {high:+.4f}]",
            small=f"The naive estimate, leaving the race lap out, is "
                  f"{naive['age']:+.4f} — the opposite sign. From "
                  f"{census['kept']:,} kept laps over {fitted.groups:,} "
                  f"car-races.",
            tiles=[("laps kept", f"{census['kept']:,}"),
                   ("car-races", f"{fitted.groups:,}"),
                   ("degradation, s/lap", f"{fitted['age']:+.4f}"),
                   ("naive estimate", f"{naive['age']:+.4f}")],
            blurb="The whole problem in one line: within a stint, tyre age "
                  "and race lap move together one for one, so leaving the "
                  "race lap out does not drop the fuel effect — it "
                  "hides it inside the degradation coefficient. The naive "
                  "estimate is therefore not a noisier version of the right "
                  "answer; it is the right answer plus the fuel effect, "
                  "which here is large enough to reverse the sign.")

        paper.section(
            "Which laps are used",
            body=f"{share:.1%} of the laps that survive belong to a car "
                 f"that made at least one stop — the only cars that "
                 f"can separate tyre age from race lap. Stints per "
                 f"car-race: {stint_mix}.",
            figure_path=[figures["census"], figures["identification"]],
            table=_census(census))

        paper.section(
            "The estimate",
            note=f"Standard errors clustered on the car-race: "
                 f"{fitted.groups:,} clusters, {fitted.n:,} laps. Within "
                 f"R² {fitted.r_squared:.4f}, residual sigma "
                 f"{fitted.sigma:.3f} s.",
            body=f"Curvature in age is {quadratic['age squared']:+.5f} s per "
                 f"lap squared (p "
                 f"{quadratic.p_value('age squared'):.3f}). The fixed "
                 f"effects are absorbed by the within transformation rather "
                 f"than by explicit dummies; the two agree to "
                 f"{fwl_gap:.1e} s, which is Frisch–Waugh–Lovell "
                 f"doing what it says.",
            figure_path=[figures["degradation_curve"], figures["estimates"]],
            table=_estimates(fitted, robust, naive))

        paper.section(
            "Does the estimator recover a degradation it was given?",
            warn="With no stops nothing identifies the split and the fitter "
                 "refuses. The naive estimator never refuses, at any stop "
                 "count, and is wrong by about the size of the fuel effect "
                 "at all of them.",
            figure_path=figures["recovery"], table=_recovery(recovery_rows))

        paper.section(
            "By circuit",
            note=f"{positive} of {len(circuit_rows)} circuits have an "
                 f"interval entirely above zero.",
            figure_path=figures["circuits"], table=_circuits(circuit_rows))

        paper.section(
            "By stint position",
            body="One model, with tyre age interacted with the stint "
                 "number. This is as close to a compound breakdown as the "
                 "feed allows: it never records what was fitted.",
            figure_path=figures["stints"], table=_stints(stint_rows))

        paper.section(
            "The races that are not a race against the clock",
            body=f"A drying track makes the whole field faster lap after "
                 f"lap, which is a race-lap effect no fuel load can "
                 f"produce, and it swamps the model. The measure is the "
                 f"spread of the field median over the race. It runs from "
                 f"{min(inside):.3f} to {max(inside):.3f} for "
                 f"{len(inside)} races and then jumps to "
                 f"{min(outside):.3f}; the cut at {condition_cut:.2f} sits "
                 f"in that empty band, so anywhere in it removes the same "
                 f"races.",
            figure_path=figures["conditions"],
            table=_conditions(spreads, condition_cut))

        paper.section(
            "Do either of the cleaning constants matter?",
            warn=f"With the condition rule on, the estimate moves by "
                 f"{spread:.4f} s per lap across every combination of the "
                 f"two constants. With it off, the safety-car threshold "
                 f"alone moves it by {max(without) - min(without):.4f} "
                 f"— more than the effect being measured. The fragile "
                 f"constant was never the safety-car rule; it was the two "
                 f"races the rule was being asked to paper over.",
            figure_path=figures["neutralisation"],
            table=_thresholds(threshold_rows))

        paper.section(
            "Traffic, and why slow laps are not deleted",
            body="Being held up adds time to a lap and never subtracts it, "
                 "so the residuals are skewed one way. The robust fit "
                 "down-weights the far tail symmetrically; removing slow "
                 "laps as a class would remove the end of every stint, "
                 "which is where the degradation is.",
            figure_path=figures["contamination"], after=SOURCE)

    return out
