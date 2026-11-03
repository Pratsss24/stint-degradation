"""Tyre degradation from F1 lap times, with the fuel effect taken out.

    python run.py              # everything (~20 s once the data is cached)
    python run.py --refresh    # re-fetch the season (several minutes)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import charts        # noqa: E402
import data          # noqa: E402
import model         # noqa: E402
import stints        # noqa: E402
import report        # noqa: E402

SEASONS = (2023,)
MIN_CIRCUIT_LAPS = 400
MAX_STINT = 3
THRESHOLDS = (1.08, 1.10, 1.15, 1.20, 1.30)
CONDITION_CUTS = (0.06, 0.08, 0.10, 0.12, 1.00)
# Published physics, used only as a cross-check and never as an input:
# a car starts on about 100 kg of fuel and burns it over the distance, and
# a kilogram is worth roughly 0.03 s a lap.
FUEL_KG_PER_LAP = 1.8
SECONDS_PER_KG = 0.030


def binned(ages, values, *, width: int = 1):
    """Mean of `values` by tyre age, for plotting."""
    table = defaultdict(list)
    for age, value in zip(ages, values):
        table[int(age // width) * width].append(value)
    keys = sorted(table)
    return (np.array(keys, dtype=float),
            np.array([np.mean(table[k]) for k in keys]),
            np.array([len(table[k]) for k in keys]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "out"))
    args = parser.parse_args()
    out_dir = Path(args.out)

    # -- the data ---------------------------------------------------------
    laps, stops, results = data.load(SEASONS, refresh=args.refresh)
    rows, census = stints.build(laps, stops, results)
    columns = stints.arrays(rows)
    share = stints.identifying_share(rows)
    counts = Counter(stints.stint_count(rows).values())

    print(f"{len(laps):,} lap times, {len(stops):,} pit stops, "
          f"{len({(r.season, r.round) for r in results})} races")
    for key, value in census.items():
        print(f"  {key:12s} {value:7,d}  {value / census['total']:6.2%}")
    print(f"  car-races {columns['n_groups']}, "
          f"{share:.1%} of kept laps belong to a car that pitted")
    print("  stints per car-race: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))

    staged, _ = stints.build(laps, stops, results, condition_spread_max=1e9)
    spreads = stints.condition_spread(staged)
    names = {(row.season, row.round): row.circuit for row in staged}
    ordered = sorted(spreads.items(), key=lambda item: -item[1])
    excluded = [key for key, value in ordered
                if value > stints.CONDITION_SPREAD_MAX]
    inside = [value for _, value in ordered
              if value <= stints.CONDITION_SPREAD_MAX]
    print(f"\ncondition drift per race, cut at "
          f"{stints.CONDITION_SPREAD_MAX:.2f}")
    for key, value in ordered[:4]:
        mark = "   EXCLUDED" if value > stints.CONDITION_SPREAD_MAX else ""
        print(f"  {names[key]:16s} {value:.3f}{mark}")
    print(f"  ... {len(inside)} more races between {min(inside):.3f} and "
          f"{max(inside):.3f}")
    print(f"  the gap between the last race kept and the first excluded is "
          f"{min(spreads[k] for k in excluded) - max(inside):.3f}")

    # -- the estimate -----------------------------------------------------
    fitted = model.fit_within(columns)
    robust = model.fit_huber(columns)
    naive = model.fit_naive(columns)
    quadratic = model.fit_within(columns, quadratic=True)
    dummy = model.fit_dummy(columns)
    gap = float(np.max(np.abs(fitted.beta - dummy.beta)))

    low, high = fitted.interval("age")
    print(f"\ndegradation  {fitted['age']:+.4f} s per lap of tyre age "
          f"(95% {low:+.4f} to {high:+.4f}, p {fitted.p_value('age'):.2e})")
    print(f"race lap     {fitted['lap']:+.4f} s per lap of race distance "
          f"(fuel burn and track evolution together)")
    print(f"             sigma {fitted.sigma:.3f} s, "
          f"within R2 {fitted.r_squared:.4f}, "
          f"{fitted.groups} clusters")
    print(f"robust       {robust['age']:+.4f} s (Huber), "
          f"race lap {robust['lap']:+.4f}")
    print(f"naive        {naive['age']:+.4f} s, age only")
    print(f"             within-stint algebra says this must be "
          f"degradation + race lap = "
          f"{fitted['age'] + fitted['lap']:+.4f}")
    print(f"quadratic    age {quadratic['age']:+.4f}, "
          f"age^2 {quadratic['age squared']:+.5f} "
          f"(p {quadratic.p_value('age squared'):.3f})")
    print(f"Frisch-Waugh-Lovell: within vs explicit dummies differ by {gap:.2e}")

    implied = FUEL_KG_PER_LAP * SECONDS_PER_KG
    print(f"\ncross-check: {FUEL_KG_PER_LAP} kg/lap at {SECONDS_PER_KG} s/kg "
          f"is {-implied:+.4f} s/lap of fuel alone, against an estimated "
          f"{fitted['lap']:+.4f} for fuel and track evolution together")

    # -- does it recover a truth it was given? ----------------------------
    recovery_rows = model.recovery()
    print(f"\nrecovery of a known degradation "
          f"({recovery_rows[0]['degradation']:+.3f} s/lap, fuel "
          f"{recovery_rows[0]['fuel']:+.3f})")
    for row in recovery_rows:
        fit_text = ("refused: not identified"
                    if not np.isfinite(row["degradation_fit"])
                    else f"{row['degradation_fit']:+.4f} / "
                         f"{row['fuel_fit']:+.4f}")
        print(f"  {row['stops']} stops   within {fit_text:26s} "
              f"naive {row['naive']:+.4f}")

    # -- by circuit --------------------------------------------------------
    circuit_rows = []
    for circuit in sorted({row.circuit for row in rows}):
        subset = [row for row in rows if row.circuit == circuit]
        if len(subset) < MIN_CIRCUIT_LAPS:
            continue
        sub_columns = stints.arrays(subset)
        try:
            sub_fit = model.fit_within(sub_columns)
        except ValueError:
            continue
        bounds = sub_fit.interval("age")
        circuit_rows.append({
            "circuit": circuit, "value": sub_fit["age"],
            "low": bounds[0], "high": bounds[1], "laps": len(subset),
            "race_lap": sub_fit["lap"]})

    print(f"\nby circuit ({len(circuit_rows)} with >= {MIN_CIRCUIT_LAPS} laps)")
    for row in sorted(circuit_rows, key=lambda r: -r["value"]):
        flag = "  clear of zero" if row["low"] > 0 else ""
        print(f"  {row['circuit']:16s} {row['value']:+.4f} "
              f"({row['low']:+.4f} to {row['high']:+.4f})  "
              f"n={row['laps']:5,d}{flag}")
    positive = sum(1 for row in circuit_rows if row["low"] > 0)
    print(f"  {positive} of {len(circuit_rows)} circuits have an interval "
          f"entirely above zero")

    # -- by stint position -------------------------------------------------
    stint_index = np.minimum(columns["stint"], MAX_STINT)
    interaction = [columns["lap"]]
    interaction_names = ["lap"]
    for k in range(1, MAX_STINT + 1):
        interaction.append(columns["age"] * (stint_index == k))
        interaction_names.append(
            f"age, stint {k}" + ("+" if k == MAX_STINT else ""))
    stint_fit = model.fit_matrix(
        columns["seconds"], np.column_stack(interaction),
        tuple(interaction_names),
        columns["group"], columns["n_groups"],
        name="within, age interacted with stint position")

    stint_rows = []
    for k in range(1, MAX_STINT + 1):
        label = f"age, stint {k}" + ("+" if k == MAX_STINT else "")
        bounds = stint_fit.interval(label)
        stint_rows.append({"stint": k, "value": stint_fit[label],
                           "low": bounds[0], "high": bounds[1],
                           "laps": int((stint_index == k).sum())})

    print("\nby stint position (one model, age interacted with the stint)")
    for row in stint_rows:
        print(f"  stint {row['stint']}{'+' if row['stint'] == MAX_STINT else ' '} "
              f"{row['value']:+.4f} ({row['low']:+.4f} to {row['high']:+.4f})  "
              f"n={row['laps']:,}")

    # -- does the safety-car threshold matter? ----------------------------
    threshold_rows = []
    for ratio in THRESHOLDS:
        for cut in CONDITION_CUTS:
            subset, sub_census = stints.build(
                laps, stops, results, neutralised_ratio=ratio,
                condition_spread_max=cut)
            sub_fit = model.fit_within(stints.arrays(subset))
            threshold_rows.append({
                "ratio": ratio, "cut": cut, "value": sub_fit["age"],
                "laps": len(subset), "removed": sub_census["neutralised"],
                "conditions": sub_census["conditions"]})

    header = "".join(f"  cut {c:.2f}" if c < 1 else "    none"
                     for c in CONDITION_CUTS)
    print("\nsensitivity to both cleaning constants")
    print(f"  safety car |{header}")
    for ratio in THRESHOLDS:
        cells = "".join(f"  {r['value']:+7.4f}" for r in threshold_rows
                        if r["ratio"] == ratio)
        print(f"  ratio {ratio:.2f} |{cells}")

    with_rule = [r["value"] for r in threshold_rows if r["cut"] < 1.0]
    without = [r["value"] for r in threshold_rows if r["cut"] == 1.0]
    print(f"  with the condition rule on, the estimate moves by "
          f"{max(with_rule) - min(with_rule):.4f} s/lap across every "
          f"combination")
    print(f"  with it off, the safety-car threshold alone moves it by "
          f"{max(without) - min(without):.4f}")

    # -- figures and report -----------------------------------------------
    residual = fitted.residuals
    corrected = columns["seconds"] - fitted["lap"] * columns["lap"]
    corrected = model.demean(corrected, columns["group"], columns["n_groups"])
    observed = model.demean(columns["seconds"], columns["group"],
                            columns["n_groups"])

    ratios = []
    neutral = stints.neutralised_laps(laps)
    by_race = defaultdict(lambda: defaultdict(list))
    for lap in laps:
        by_race[(lap.season, lap.round)][lap.lap].append(lap.seconds)
    for race, table in by_race.items():
        medians = {k: float(np.median(v)) for k, v in table.items()}
        baseline = float(np.median(list(medians.values())))
        ratios.extend(value / baseline for value in medians.values())

    example_multi = example_single = None
    for key, n_stints in stints.stint_count(rows).items():
        subset = [row for row in rows if row.key == key]
        payload = {"lap": [r.lap for r in subset], "age": [r.age for r in subset],
                   "label": f"{key[2]}, {subset[0].race}"}
        if n_stints >= 2 and example_multi is None:
            example_multi = payload
        if n_stints == 1 and example_single is None and len(subset) > 25:
            example_single = payload
    if example_single is None:          # every car pitted; make the point anyway
        example_single = {"lap": example_multi["lap"],
                          "age": example_multi["lap"],
                          "label": "hypothetical, no stop"}

    figures = {
        "census": charts.census(census, out_dir),
        "degradation_curve": charts.degradation_curve(
            binned(columns["age"], observed),
            binned(columns["age"], corrected),
            fitted["age"], naive["age"], out_dir),
        "identification": charts.identification(example_multi, example_single,
                                                out_dir),
        "estimates": charts.estimates([
            {"label": "naive: age only", "value": naive["age"],
             "low": naive.interval("age")[0], "high": naive.interval("age")[1],
             "wrong": True},
            {"label": "within, with race lap", "value": fitted["age"],
             "low": low, "high": high},
            {"label": "within, Huber", "value": robust["age"],
             "low": robust.interval("age")[0],
             "high": robust.interval("age")[1]},
        ], out_dir),
        "recovery": charts.recovery(recovery_rows, out_dir),
        "circuits": charts.circuits(circuit_rows, out_dir),
        "stints": charts.stints(stint_rows, out_dir),
        "conditions": charts.conditions(
            [{"circuit": names[key], "spread": value}
             for key, value in spreads.items()],
            stints.CONDITION_SPREAD_MAX, out_dir),
        "neutralisation": charts.neutralisation(ratios, threshold_rows,
                                                out_dir),
        "contamination": charts.contamination(residual, robust.weights,
                                              out_dir),
    }

    report_path = report.build(
        census=census, share=share, stint_counts=counts, fitted=fitted,
        robust=robust, naive=naive, quadratic=quadratic, fwl_gap=gap,
        recovery_rows=recovery_rows, circuit_rows=circuit_rows,
        stint_rows=stint_rows, threshold_rows=threshold_rows,
        implied_fuel=-implied, seasons=SEASONS, figures=figures,
        spreads=[{"circuit": names[key], "spread": value}
                 for key, value in sorted(spreads.items(),
                                          key=lambda item: -item[1])],
        condition_cut=stints.CONDITION_SPREAD_MAX,
        out_path=out_dir / "report.pdf")

    print(f"\nreport  {report_path}")
    for path in figures.values():
        print(f"figure  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
