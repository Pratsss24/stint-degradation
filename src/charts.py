"""Figures. Committed to the repository so the write-up renders anywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

PALETTE = {"ink": "#16181d", "muted": "#5b6472", "grid": "#e3e6ea",
           "good": "#17803d", "bad": "#b4232c", "accent": "#2563a8",
           "warm": "#d9a441", "plum": "#6b4c9a"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"], "axes.titlesize": 10,
    "axes.titleweight": "bold", "figure.facecolor": "white",
})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _grid(ax):
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def degradation_curve(raw, corrected, slope: float, naive: float,
                      out_dir: Path) -> Path:
    """The same laps before and after the race-lap effect is removed.

    Left: lap time against tyre age with nothing taken out — the picture
    the naive estimate is fitted to. Right: the same after the estimated
    race-lap effect is subtracted, which is what degradation actually
    looks like.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for ax, (ages, means, counts), colour, title, slope_value in (
            (axes[0], raw, PALETTE["muted"], "As observed", naive),
            (axes[1], corrected, PALETTE["accent"],
             "Race-lap effect removed", slope)):
        weight = np.clip(np.array(counts, dtype=float), 1, None)
        ax.scatter(ages, means, s=6 + 40 * weight / weight.max(),
                   color=colour, alpha=0.75, zorder=3)
        grid = np.linspace(min(ages), max(ages), 50)
        centre = float(np.average(means, weights=weight))
        mid = float(np.average(ages, weights=weight))
        ax.plot(grid, centre + slope_value * (grid - mid),
                color=PALETTE["bad"], linewidth=1.6, linestyle="--",
                label=f"{slope_value:+.4f} s per lap of age")
        ax.axhline(centre, color=PALETTE["grid"], linewidth=1, zorder=1)
        ax.set_xlabel("tyre age (laps)")
        ax.set_ylabel("lap time relative to the car-race mean (s)")
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper left")
        _grid(ax)

    fig.suptitle("Degradation is only visible once the fuel effect is out",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "degradation_curve")


def identification(multi: dict, single: dict, out_dir: Path) -> Path:
    """Why a car that stops can answer the question and one that cannot, cannot."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)

    for ax, car, title in ((axes[0], multi, "A car that pits"),
                           (axes[1], single, "A car that does not")):
        ax.plot(car["lap"], car["age"], ".", color=PALETTE["accent"],
                markersize=5)
        ax.plot(car["lap"], car["lap"], color=PALETTE["bad"], linewidth=1.0,
                linestyle=":", label="age = lap (perfect collinearity)")
        ax.set_xlabel("race lap")
        ax.set_title(f"{title} — {car['label']}")
        ax.legend(frameon=False, loc="upper left")
        _grid(ax)
    axes[0].set_ylabel("tyre age (laps)")

    fig.suptitle("Tyre age resets; the race lap counter does not",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "identification")


def estimates(rows, out_dir: Path) -> Path:
    """Every estimator of the same quantity, with its interval."""
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(rows) + 1.6))
    positions = np.arange(len(rows))[::-1]

    for y, row in zip(positions, rows):
        colour = PALETTE["bad"] if row.get("wrong") else PALETTE["accent"]
        ax.plot([row["low"], row["high"]], [y, y], color=colour,
                linewidth=2.2, solid_capstyle="round")
        ax.plot(row["value"], y, "o", color=colour, markersize=6)
        ax.text(row["high"], y + 0.22, f"{row['value']:+.4f}",
                color=colour, fontsize=8.5)

    ax.axvline(0, color=PALETTE["muted"], linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_xlabel("degradation (seconds per lap of tyre age)")
    ax.set_title("The estimate, and the estimate with the fuel effect left in")
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "estimates")


def recovery(rows, out_dir: Path) -> Path:
    """Recovering a known degradation at several stop counts."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    stops = [row["stops"] for row in rows]
    truth = rows[0]["degradation"]

    fitted = [row["degradation_fit"] for row in rows]
    naive = [row["naive"] for row in rows]
    errors = [1.96 * row["degradation_se"] for row in rows]

    ax.axhline(truth, color=PALETTE["good"], linewidth=1.6,
               label=f"truth {truth:+.3f}")
    ax.axhline(truth + rows[0]["fuel"], color=PALETTE["grid"], linewidth=1.4,
               linestyle="--",
               label=f"truth + fuel {truth + rows[0]['fuel']:+.3f}")
    ax.errorbar(stops, fitted, yerr=errors, fmt="o-", color=PALETTE["accent"],
                linewidth=1.5, capsize=3, label="within, with race lap")
    ax.plot(stops, naive, "s--", color=PALETTE["bad"], linewidth=1.5,
            label="naive, age only")

    for x, value in zip(stops, fitted):
        if not np.isfinite(value):
            ax.annotate("refused:\nnot identified", (x, truth),
                        textcoords="offset points", xytext=(6, 10),
                        color=PALETTE["muted"], fontsize=8.5)

    ax.set_xticks(stops)
    ax.set_xlabel("pit stops per car in the simulated race")
    ax.set_ylabel("estimated degradation (s per lap)")
    ax.set_title("The naive estimate misses by the fuel effect, at every stop count")
    ax.legend(frameon=False)
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "recovery")


def _ranked(rows, key, out_dir, name, xlabel, title):
    fig, ax = plt.subplots(figsize=(8.6, 0.33 * len(rows) + 1.8))
    ordered = sorted(rows, key=lambda r: r["value"])
    positions = np.arange(len(ordered))

    for y, row in zip(positions, ordered):
        colour = PALETTE["bad"] if row["low"] > 0 else PALETTE["muted"]
        ax.plot([row["low"], row["high"]], [y, y], color=colour,
                linewidth=1.8, solid_capstyle="round")
        ax.plot(row["value"], y, "o", color=colour, markersize=4.5)

    ax.axvline(0, color=PALETTE["ink"], linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([row[key] for row in ordered], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def circuits(rows, out_dir: Path) -> Path:
    return _ranked(rows, "circuit", out_dir, "circuits",
                   "degradation (s per lap of tyre age), 95 % interval",
                   "Degradation by circuit, each fitted on its own races")


def stints(rows, out_dir: Path) -> Path:
    """Degradation by stint position — the only compound proxy available."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    positions = [row["stint"] for row in rows]
    values = [row["value"] for row in rows]
    errors = [[row["value"] - row["low"] for row in rows],
              [row["high"] - row["value"] for row in rows]]

    ax.errorbar(positions, values, yerr=errors, fmt="o", color=PALETTE["plum"],
                capsize=4, markersize=6, linewidth=1.6)
    for row in rows:
        ax.annotate(f"n={row['laps']:,}", (row["stint"], row["high"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", color=PALETTE["muted"], fontsize=8)
    ax.axhline(0, color=PALETTE["muted"], linewidth=1)
    ax.set_xticks(positions)
    ax.set_xlabel("stint number")
    ax.set_ylabel("degradation (s per lap of tyre age)")
    ax.set_title("By stint position, which is as close to a compound as this feed gets")
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "stints")


def conditions(rows, cut: float, out_dir: Path) -> Path:
    """How far each race's own pace drifts, and where the rule cuts.

    The point of the figure is the empty band: the races that get removed
    are not the tail of a continuum, they are somewhere else entirely.
    """
    ordered = sorted(rows, key=lambda r: r["spread"])
    positions = np.arange(len(ordered))
    values = [r["spread"] for r in ordered]
    colours = [PALETTE["bad"] if v > cut else PALETTE["muted"] for v in values]

    fig, ax = plt.subplots(figsize=(8.6, 0.33 * len(ordered) + 1.6))
    ax.barh(positions, values, color=colours, height=0.62)
    ax.axvline(cut, color=PALETTE["accent"], linewidth=1.6,
               label=f"cut at {cut:.2f}")

    inside = [v for v in values if v <= cut]
    outside = [v for v in values if v > cut]
    if inside and outside:
        ax.axvspan(max(inside), min(outside), color=PALETTE["accent"],
                   alpha=0.08, linewidth=0)
        ax.annotate(f"nothing in {max(inside):.3f}-{min(outside):.3f}",
                    ((max(inside) + min(outside)) / 2, len(ordered) * 0.45),
                    rotation=90, ha="center", va="center", fontsize=8,
                    color=PALETTE["muted"])

    ax.set_yticks(positions)
    ax.set_yticklabels([r["circuit"] for r in ordered], fontsize=8)
    ax.set_xlabel("spread of the field median over the race, "
                  "as a fraction of its own level")
    ax.set_title("A dry race holds its pace; a drying one does not")
    ax.legend(frameon=False, loc="lower right")
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "conditions")


def neutralisation(ratios, grid, out_dir: Path) -> Path:
    """The safety-car threshold, and whether either constant matters."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ratios = np.asarray(ratios, dtype=float)
    limit = 2.5
    axes[0].hist(np.clip(ratios, None, limit), bins=90,
                 color=PALETTE["muted"])
    axes[0].axvline(1.15, color=PALETTE["bad"], linewidth=1.6,
                    label="threshold 1.15")
    beyond = int((ratios > limit).sum())
    if beyond:
        # Red-flag laps: a "lap" that contains a stoppage runs to thirty
        # times the baseline and would otherwise own the whole axis.
        axes[0].annotate(f"{beyond} laps beyond {limit:.1f}x, "
                         f"up to {ratios.max():.0f}x (red flags)",
                         (limit, 3), ha="right", va="bottom", fontsize=8,
                         color=PALETTE["muted"])
    axes[0].set_xlim(0.95, limit)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("field median lap time / race baseline")
    axes[0].set_ylabel("laps (log scale)")
    axes[0].set_title("Racing laps and neutralised laps are not a continuum")
    axes[0].legend(frameon=False)
    _grid(axes[0])

    cuts = sorted({row["cut"] for row in grid})
    shades = [PALETTE["accent"], PALETTE["plum"], PALETTE["good"],
              PALETTE["warm"], PALETTE["bad"]]
    for colour, cut in zip(shades, cuts):
        points = sorted((row for row in grid if row["cut"] == cut),
                        key=lambda r: r["ratio"])
        label = (f"condition cut {cut:.2f}" if cut < 1
                 else "no condition rule")
        axes[1].plot([p["ratio"] for p in points],
                     [p["value"] for p in points], marker="o", color=colour,
                     linewidth=1.6,
                     linestyle="--" if cut >= 1 else "-", label=label)

    axes[1].axhline(0, color=PALETTE["ink"], linewidth=1)
    axes[1].set_xlabel("safety-car threshold")
    axes[1].set_ylabel("degradation (s per lap)")
    axes[1].set_title("Stable in both constants — once the wet races are out")
    axes[1].legend(frameon=False, fontsize=8)
    _grid(axes[1])

    fig.tight_layout()
    return _save(fig, out_dir, "neutralisation")


def contamination(residual, weights, out_dir: Path) -> Path:
    """Traffic only ever adds time, and the residuals show it."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    axes[0].hist(residual, bins=140, color=PALETTE["muted"])
    axes[0].axvline(0, color=PALETTE["ink"], linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("residual (s)")
    axes[0].set_ylabel("laps (log scale)")
    axes[0].set_title("Residuals: a long tail on one side only")
    _grid(axes[0])

    order = np.argsort(residual)
    axes[1].plot(residual[order], weights[order], color=PALETTE["plum"],
                 linewidth=1.6)
    axes[1].set_xlabel("residual (s)")
    axes[1].set_ylabel("Huber weight")
    axes[1].set_title("What the robust fit does with them")
    _grid(axes[1])

    fig.tight_layout()
    return _save(fig, out_dir, "contamination")


def census(table, out_dir: Path) -> Path:
    """Where the laps went."""
    labels = [key for key in table if key not in ("total", "kept")]
    values = [table[key] for key in labels]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    bars = ax.bar(labels + ["kept"], values + [table["kept"]],
                  color=[PALETTE["muted"]] * len(labels) + [PALETTE["good"]])
    for bar, value in zip(bars, values + [table["kept"]]):
        ax.annotate(f"{value:,}\n{value / table['total']:.1%}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8, color=PALETTE["ink"])
    ax.set_ylabel("laps")
    ax.set_title(f"What happened to the {table['total']:,} laps in the feed")
    _grid(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "census")
