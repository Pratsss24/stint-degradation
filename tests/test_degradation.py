"""Tests anchored on algebra and on data whose answer is known in advance.

A regression always returns a number. Nothing about a wrong one looks
wrong, so almost nothing here compares against stored output. Instead:

* **Exact algebra.** Within a stint, race lap and tyre age differ by a
  constant, so after the stint means are removed they are the *same
  column*. The naive estimate must therefore equal degradation plus the
  fuel effect — not approximately, exactly, on noiseless data. And the
  within transformation must agree with explicit dummy variables to
  machine precision, because Frisch-Waugh-Lovell says they are the same
  estimator.
* **Recovery of a known truth.** Simulated races with the two coefficients
  fixed in advance. The right model has to find them; the naive one has to
  miss, in the stated direction and by the stated amount.
* **Refusal.** A sample where nothing identifies the split has to raise,
  not return a number.
* **Hand examples.** Stint and tyre-age assignment around a stop on a
  known lap, worked out on paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data          # noqa: E402
import model         # noqa: E402
import stints        # noqa: E402


@pytest.fixture(scope="module")
def feed():
    """The cached season. The suite never goes to the network itself."""
    missing = [season for season in (2023,)
               if not (ROOT / "data" / f"season_{season}.json").exists()]
    if missing:
        pytest.skip("run `python run.py` once to cache the season")
    return data.load()


@pytest.fixture(scope="module")
def rows(feed):
    laps, stops, results = feed
    kept, _ = stints.build(laps, stops, results)
    return kept


@pytest.fixture(scope="module")
def columns(rows):
    return stints.arrays(rows)


# -- the algebra ----------------------------------------------------------

def test_the_naive_estimate_is_exactly_degradation_plus_fuel():
    """The identity the whole project rests on.

    Inside a stint, lap = age + (first lap of the stint) - 1, and that
    offset is constant. Remove the stint mean and the two columns are
    identical, so a regression on age alone cannot distinguish the two
    effects: it returns their sum. With no noise this is exact.
    """
    simulated = model.simulate(n_cars=60, stops=2, noise=0.0, seed=3)
    truth = simulated["truth"]
    naive = model.fit_naive(simulated)["age"]
    assert naive == pytest.approx(truth["degradation"] + truth["fuel"],
                                  abs=1e-12)


def test_the_two_columns_coincide_once_the_stint_mean_is_removed():
    """The same statement one level down, as columns rather than estimates."""
    simulated = model.simulate(n_cars=25, stops=2, noise=0.0, seed=5)
    key = simulated["group"] * 100 + simulated["stint"]
    _, group = np.unique(key, return_inverse=True)
    n_groups = int(group.max()) + 1

    age = model.demean(simulated["age"], group, n_groups)
    lap = model.demean(simulated["lap"], group, n_groups)
    assert np.max(np.abs(age - lap)) < 1e-12


def test_the_within_transformation_equals_explicit_dummy_variables():
    """Frisch-Waugh-Lovell, checked rather than cited."""
    simulated = model.simulate(n_cars=40, stops=2, noise=0.4, seed=7)
    within = model.fit_within(simulated)
    dummies = model.fit_dummy(simulated)
    assert np.max(np.abs(within.beta - dummies.beta)) < 1e-9


def test_demeaning_leaves_every_group_mean_at_zero():
    rng = np.random.default_rng(11)
    values = rng.normal(size=500)
    group = rng.integers(0, 12, size=500)
    centred = model.demean(values, group, 12)
    for g in range(12):
        assert centred[group == g].mean() == pytest.approx(0.0, abs=1e-12)


def test_demeaning_is_idempotent():
    rng = np.random.default_rng(12)
    values = rng.normal(size=300)
    group = rng.integers(0, 7, size=300)
    once = model.demean(values, group, 7)
    twice = model.demean(once, group, 7)
    assert np.max(np.abs(once - twice)) < 1e-12


# -- recovery, and refusal ------------------------------------------------

@pytest.mark.parametrize("stops", [1, 2, 3])
def test_the_model_recovers_a_degradation_it_was_given(stops):
    simulated = model.simulate(n_cars=300, stops=stops, noise=0.45, seed=stops)
    fitted = model.fit_within(simulated)
    truth = simulated["truth"]
    assert fitted["age"] == pytest.approx(truth["degradation"], abs=0.004)
    assert fitted["lap"] == pytest.approx(truth["fuel"], abs=0.004)


def test_a_race_with_no_pit_stops_is_refused_rather_than_answered():
    """Nothing in that sample separates the two effects, so nothing may be
    reported. The failure has to be loud: a silent pseudo-inverse would
    return a number with no meaning."""
    simulated = model.simulate(n_cars=80, stops=0, noise=0.3, seed=9)
    with pytest.raises(ValueError, match="rank deficient"):
        model.fit_within(simulated)


def test_the_naive_estimator_answers_even_when_nothing_is_identified():
    """Which is the point: it never tells you it cannot know."""
    simulated = model.simulate(n_cars=80, stops=0, noise=0.3, seed=9)
    value = model.fit_naive(simulated)["age"]
    assert np.isfinite(value)
    truth = simulated["truth"]
    assert value == pytest.approx(truth["degradation"] + truth["fuel"],
                                  abs=0.01)


def test_the_naive_estimate_has_the_wrong_sign_when_fuel_dominates():
    """Degradation of +0.06 reported as negative: tyres getting faster."""
    simulated = model.simulate(n_cars=200, degradation=0.060, fuel=-0.075,
                               stops=2, noise=0.4, seed=13)
    assert model.fit_naive(simulated)["age"] < 0
    assert model.fit_within(simulated)["age"] > 0


def test_recovery_reports_every_stop_count_including_the_refused_one():
    table = model.recovery(levels=(0, 2))
    assert np.isnan(table[0]["degradation_fit"])
    assert "refused" in table[0]
    assert table[1]["degradation_fit"] == pytest.approx(0.060, abs=0.005)


# -- standard errors ------------------------------------------------------

def test_clustering_widens_the_interval_when_laps_are_correlated():
    """Laps within one car's race are not independent draws.

    A driver nursing a problem is slow for twenty laps together. Treating
    those as twenty independent observations overstates the precision, so
    the clustered error must be the larger one.
    """
    rng = np.random.default_rng(17)
    simulated = model.simulate(n_cars=120, stops=2, noise=0.2, seed=17)
    # A shock that lasts a whole stint: a wrong set-up, a damaged floor, a
    # driver managing a problem. A shock constant across the entire race
    # would prove nothing, because the car-race fixed effect absorbs it.
    key = simulated["group"] * 100 + simulated["stint"]
    _, stint_index = np.unique(key, return_inverse=True)
    simulated["seconds"] = simulated["seconds"] + rng.normal(
        scale=1.0, size=stint_index.max() + 1)[stint_index]

    clustered = model.fit_within(simulated)
    matrix, names = model.design(simulated)
    within = np.column_stack([
        model.demean(matrix[:, j], simulated["group"], simulated["n_groups"])
        for j in range(matrix.shape[1])])
    y = model.demean(simulated["seconds"], simulated["group"],
                     simulated["n_groups"])
    beta = np.linalg.solve(within.T @ within, within.T @ y)
    residual = y - within @ beta
    dof = len(y) - simulated["n_groups"] - len(names)
    plain = np.sqrt(np.diag(np.linalg.inv(within.T @ within))
                    * (residual @ residual) / dof)
    assert clustered.error("age") > 2.0 * plain[0]


def test_the_p_value_counts_clusters_not_laps():
    simulated = model.simulate(n_cars=30, stops=2, noise=0.4, seed=19)
    fitted = model.fit_within(simulated)
    assert fitted.groups == 30
    assert fitted.n > 30 * 20


# -- robustness, and the filter that must not be applied ------------------

def test_huber_and_least_squares_agree_when_nothing_is_contaminated():
    simulated = model.simulate(n_cars=150, stops=2, noise=0.4, seed=21)
    assert (model.fit_huber(simulated)["age"]
            == pytest.approx(model.fit_within(simulated)["age"], abs=0.004))


def test_huber_holds_up_where_least_squares_is_dragged_by_traffic():
    """Contamination here is one-sided: traffic only ever adds time."""
    simulated = model.simulate(n_cars=200, stops=2, noise=0.35, seed=23)
    rng = np.random.default_rng(23)
    blocked = rng.random(len(simulated["seconds"])) < 0.04
    simulated["seconds"] = simulated["seconds"] + blocked * rng.gamma(
        2.0, 2.0, size=len(blocked))

    truth = simulated["truth"]["degradation"]
    ordinary = abs(model.fit_within(simulated)["age"] - truth)
    robust = abs(model.fit_huber(simulated)["age"] - truth)
    assert robust < ordinary


def test_trimming_the_slowest_laps_biases_the_estimate_upward():
    """Why there is no 'remove slow laps' rule in the cleaning step.

    The obvious tidy-up is to drop the slowest laps of each stint as
    traffic. It is a selection on the outcome, so it cannot be neutral,
    and the direction follows from the same confounding the project is
    about: within a stint, lap times *fall* with tyre age, because the
    fuel effect is larger than the degradation. The slowest laps of a
    stint are therefore its earliest ones, on fresh tyres and a heavy car.
    Cutting them removes the start of every stint and tilts the slope up.

    The test is of a mistake, kept so the mistake cannot be reintroduced
    as a cleanup later. It also records the direction, which is the
    opposite of the intuitive one.
    """
    simulated = model.simulate(n_cars=250, stops=2, noise=0.4, seed=29)
    truth = simulated["truth"]["degradation"]
    honest = model.fit_within(simulated)["age"]
    assert honest == pytest.approx(truth, abs=0.005)

    key = simulated["group"] * 100 + simulated["stint"]
    keep = np.ones(len(key), dtype=bool)
    for stint in np.unique(key):
        rows = np.flatnonzero(key == stint)
        cut = np.quantile(simulated["seconds"][rows], 0.90)
        keep[rows[simulated["seconds"][rows] > cut]] = False

    trimmed = {k: (v[keep] if isinstance(v, np.ndarray) else v)
               for k, v in simulated.items() if k != "truth"}
    _, trimmed["group"] = np.unique(trimmed["group"], return_inverse=True)
    trimmed["n_groups"] = int(trimmed["group"].max()) + 1
    assert model.fit_within(trimmed)["age"] > 1.03 * truth


# -- turning a feed into stints -------------------------------------------

def test_stint_and_age_around_a_stop_on_a_known_lap():
    """One stop on lap 18: lap 18 ends stint 1, lap 19 starts stint 2."""
    stops = [18]
    assert stints.assign_stint(1, stops) == (1, 1)
    assert stints.assign_stint(17, stops) == (1, 17)
    assert stints.assign_stint(18, stops) == (1, 18)     # the in-lap
    assert stints.assign_stint(19, stops) == (2, 1)      # the out-lap
    assert stints.assign_stint(20, stops) == (2, 2)


def test_stint_and_age_with_two_stops():
    stops = [14, 33]
    assert stints.assign_stint(14, stops) == (1, 14)
    assert stints.assign_stint(15, stops) == (2, 1)
    assert stints.assign_stint(33, stops) == (2, 19)
    assert stints.assign_stint(34, stops) == (3, 1)
    assert stints.assign_stint(50, stops) == (3, 17)


def test_a_car_that_never_stops_is_one_stint_whose_age_is_its_lap():
    for lap in (1, 5, 40):
        assert stints.assign_stint(lap, []) == (1, lap)


def test_lap_times_parse_and_rubbish_is_refused_rather_than_guessed():
    assert data.parse_lap_time("1:39.019") == pytest.approx(99.019)
    assert data.parse_lap_time("59.512") == pytest.approx(59.512)
    assert data.parse_lap_time("2:00.000") == pytest.approx(120.0)
    assert data.parse_lap_time("") is None
    assert data.parse_lap_time("   ") is None
    assert data.parse_lap_time("n/a") is None


def _fake_feed(times_by_lap, *, stops=(), race_laps=20, drivers=("a", "b")):
    laps = [data.Lap(season=2023, round=1, race="Test", circuit="test",
                     driver=driver, lap=lap, seconds=seconds, position=1)
            for lap, seconds in times_by_lap.items() for driver in drivers]
    stop_rows = [data.Stop(season=2023, round=1, driver=driver, lap=lap,
                           stop=i + 1, duration=22.0)
                 for i, lap in enumerate(stops) for driver in drivers]
    results = [data.Result(season=2023, round=1, race="Test", circuit="test",
                           driver=driver, team="t", grid=1, laps=race_laps,
                           status="Finished", position_text="1")
               for driver in drivers]
    return laps, stop_rows, results


def test_the_safety_car_is_found_from_the_field_not_assumed():
    times = {lap: 90.0 for lap in range(1, 21)}
    for lap in (8, 9, 10):
        times[lap] = 130.0
    laps, _, _ = _fake_feed(times)
    assert stints.neutralised_laps(laps)[(2023, 1)] == {8, 9, 10}


def test_one_slow_car_does_not_neutralise_a_lap():
    """The field median is what moves under a safety car, not one driver."""
    laps = []
    for lap in range(1, 21):
        for driver in ("a", "b", "c", "d", "e"):
            seconds = 140.0 if (driver == "a" and lap == 12) else 90.0
            laps.append(data.Lap(season=2023, round=1, race="Test",
                                 circuit="test", driver=driver, lap=lap,
                                 seconds=seconds, position=1))
    assert stints.neutralised_laps(laps)[(2023, 1)] == set()


def test_the_in_lap_and_the_out_lap_are_both_dropped():
    times = {lap: 90.0 for lap in range(1, 21)}
    laps, stops, results = _fake_feed(times, stops=(10,))
    kept, census = stints.build(laps, stops, results, min_stint_laps=1)

    for row in kept:
        assert row.lap != 10                       # in-lap
        assert not (row.stint == 2 and row.age == 1)   # out-lap
    assert census["in-lap"] == 2
    assert census["out-lap"] == 2
    assert census["lap one"] == 2
    assert min(row.age for row in kept if row.stint == 2) == 2


def test_the_last_lap_of_a_car_that_retired_is_dropped():
    times = {lap: 90.0 for lap in range(1, 21)}
    laps, stops, results = _fake_feed(times)
    results = [data.Result(**{**r.__dict__, "status": "Engine", "laps": 15})
               for r in results]
    kept, census = stints.build(laps, stops, results, min_stint_laps=1)
    assert census["retirement"] == 2
    assert all(row.lap != 15 for row in kept)


def test_identifying_share_is_zero_when_nobody_pits():
    times = {lap: 90.0 for lap in range(1, 21)}
    laps, stops, results = _fake_feed(times)
    kept, _ = stints.build(laps, stops, results, min_stint_laps=1)
    assert stints.identifying_share(kept) == 0.0

    laps, stops, results = _fake_feed(times, stops=(10,))
    kept, _ = stints.build(laps, stops, results, min_stint_laps=1)
    assert stints.identifying_share(kept) == 1.0


# -- the real season ------------------------------------------------------

def test_the_feed_has_enough_pitting_cars_to_identify_anything(rows):
    assert len(rows) > 10_000
    assert stints.identifying_share(rows) > 0.9


def test_degradation_is_positive_and_clear_of_zero(columns):
    """The headline, as an interval rather than a stored number."""
    fitted = model.fit_within(columns)
    low, high = fitted.interval("age")
    assert fitted["age"] > 0
    assert low > 0
    assert fitted.p_value("age") < 1e-6


def test_the_race_lap_effect_is_negative_and_the_right_size(columns):
    """Cars get lighter and tracks rubber in, so lap times fall.

    The published physics — about 1.8 kg of fuel a lap at roughly 0.03 s a
    kilogram — puts the fuel part alone near -0.054 s/lap. The estimate
    also contains track evolution, so it should be negative and at least
    that large, without being absurd.
    """
    fitted = model.fit_within(columns)
    assert fitted["lap"] < 0
    assert -0.25 < fitted["lap"] < -0.02


def test_the_naive_estimate_is_materially_lower_on_the_real_data(columns):
    """Not exactly degradation + fuel here, because the two models absorb
    different fixed effects, but close to it and far from the truth."""
    fitted = model.fit_within(columns)
    naive = model.fit_naive(columns)
    assert naive["age"] < fitted["age"] - 0.02
    assert naive["age"] == pytest.approx(fitted["age"] + fitted["lap"],
                                         abs=0.02)


def test_the_estimate_survives_both_cleaning_constants(feed):
    """If the headline moved with an arbitrary constant it would be a
    property of the constant, not of the tyres."""
    laps, stops, results = feed
    values = []
    for ratio in (1.08, 1.15, 1.30):
        for cut in (0.06, 0.08, 0.12):
            kept, _ = stints.build(laps, stops, results,
                                   neutralised_ratio=ratio,
                                   condition_spread_max=cut)
            values.append(model.fit_within(stints.arrays(kept))["age"])
    assert all(value > 0 for value in values)
    assert max(values) - min(values) < 0.02


def test_without_the_condition_rule_the_answer_is_the_threshold_s(feed):
    """The finding that produced the rule, kept as a test.

    Leave the rain-affected races in and the safety-car threshold — an
    arbitrary constant — swings the estimate across zero, from clearly
    positive to clearly negative. No conclusion drawn from that sample is
    about tyres.
    """
    laps, stops, results = feed
    values = []
    for ratio in (1.08, 1.15, 1.30):
        kept, _ = stints.build(laps, stops, results, neutralised_ratio=ratio,
                               condition_spread_max=1e9)
        values.append(model.fit_within(stints.arrays(kept))["age"])
    assert max(values) > 0 > min(values)
    assert max(values) - min(values) > 0.05


def test_the_condition_measure_separates_a_drying_race_from_a_dry_one():
    """A dry race drifts by a few per cent; a drying one by far more.

    The drying race here is deliberately gradual — no single lap is slow
    enough to be caught by the safety-car rule — because that is exactly
    the case the safety-car rule cannot handle and this measure can.
    """
    dry = {lap: 90.0 - 0.05 * lap for lap in range(1, 51)}
    drying = {lap: 110.0 - 0.4 * lap for lap in range(1, 51)}

    for times, expected in ((dry, "small"), (drying, "large")):
        laps, stops, results = _fake_feed(times, race_laps=50)
        rows, census = stints.build(laps, stops, results, min_stint_laps=1,
                                    condition_spread_max=1e9)
        assert census["neutralised"] == 0        # nothing looks like an SC
        spread = list(stints.condition_spread(rows).values())[0]
        if expected == "small":
            assert spread < 0.05
        else:
            assert spread > 0.15


def test_a_race_with_changing_conditions_is_removed_whole():
    """Not lap by lap: a drying track contaminates the race-lap effect for
    every car in it, including the laps that look ordinary."""
    times = {lap: 110.0 - 0.4 * lap for lap in range(1, 51)}
    laps, stops, results = _fake_feed(times, stops=(20,), race_laps=50)

    kept, census = stints.build(laps, stops, results, min_stint_laps=1,
                                condition_spread_max=0.08)
    assert kept == []
    assert census["conditions"] > 0

    kept, census = stints.build(laps, stops, results, min_stint_laps=1,
                                condition_spread_max=1e9)
    assert kept
    assert census["conditions"] == 0


# -- the real season ------------------------------------------------------

def test_the_feed_has_enough_pitting_cars_to_identify_anything(rows):
    assert len(rows) > 10_000
    assert stints.identifying_share(rows) > 0.9


def test_degradation_is_positive_and_clear_of_zero(columns):
    """The headline, as an interval rather than a stored number."""
    fitted = model.fit_within(columns)
    low, high = fitted.interval("age")
    assert fitted["age"] > 0
    assert low > 0
    assert fitted.p_value("age") < 1e-6


def test_the_race_lap_effect_is_negative_and_the_right_size(columns):
    """Cars get lighter and tracks rubber in, so lap times fall.

    The published physics — about 1.8 kg of fuel a lap at roughly 0.03 s a
    kilogram — puts the fuel part alone near -0.054 s/lap. The estimate
    also contains track evolution, so it should be negative and at least
    that large, without being absurd.
    """
    fitted = model.fit_within(columns)
    assert fitted["lap"] < 0
    assert -0.25 < fitted["lap"] < -0.02


def test_the_naive_estimate_is_materially_lower_on_the_real_data(columns):
    """Not exactly degradation + fuel here, because the two models absorb
    different fixed effects, but close to it and far from the truth."""
    fitted = model.fit_within(columns)
    naive = model.fit_naive(columns)
    assert naive["age"] < fitted["age"] - 0.02
    assert naive["age"] == pytest.approx(fitted["age"] + fitted["lap"],
                                         abs=0.02)


def test_the_estimate_survives_both_cleaning_constants(feed):
    """If the headline moved with an arbitrary constant it would be a
    property of the constant, not of the tyres."""
    laps, stops, results = feed
    values = []
    for ratio in (1.08, 1.15, 1.30):
        for cut in (0.06, 0.08, 0.12):
            kept, _ = stints.build(laps, stops, results,
                                   neutralised_ratio=ratio,
                                   condition_spread_max=cut)
            values.append(model.fit_within(stints.arrays(kept))["age"])
    assert all(value > 0 for value in values)
    assert max(values) - min(values) < 0.02


def test_without_the_condition_rule_the_answer_is_the_threshold_s(feed):
    """The finding that produced the rule, kept as a test.

    Leave the rain-affected races in and the safety-car threshold — an
    arbitrary constant — swings the estimate across zero, from clearly
    positive to clearly negative. No conclusion drawn from that sample is
    about tyres.
    """
    laps, stops, results = feed
    values = []
    for ratio in (1.08, 1.15, 1.30):
        kept, _ = stints.build(laps, stops, results, neutralised_ratio=ratio,
                               condition_spread_max=1e9)
        values.append(model.fit_within(stints.arrays(kept))["age"])
    assert max(values) > 0 > min(values)
    assert max(values) - min(values) > 0.05


def test_the_robust_fit_does_not_overturn_the_result(columns):
    ordinary = model.fit_within(columns)["age"]
    robust = model.fit_huber(columns)["age"]
    assert robust > 0
    assert abs(robust - ordinary) < 0.5 * abs(ordinary)


def test_no_kept_lap_is_an_in_lap_out_lap_or_first_lap(rows, feed):
    _, stops, _ = feed
    table = stints.stop_laps(stops)
    for row in rows:
        assert row.lap != 1
        assert row.lap not in table.get(row.key, ())
        if row.stint > 1:
            assert row.age >= 2
