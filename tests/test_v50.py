import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from inventory_v50 import (
    _assign_fold_splits,
    _baseline_and_single_window,
    _build_fold_schedule,
    _replication_seed,
    _select_historical,
    _window_panel,
)


def _history_frame(fill_rates, costs=None):
    costs = costs or {}
    rows = []
    config_ids = [
        "q50_proportional", "q75_proportional",
        "q50_avg_greedy", "q75_avg_greedy", "q90_avg_greedy",
        "q50_marginal_lot", "q75_marginal_lot", "q90_marginal_lot",
    ]
    for config_id in config_ids:
        rows.append({
            "config_id": config_id,
            "mean_cost": float(costs.get(config_id, 10.0)),
            "mean_fill_rate": float(fill_rates.get(config_id, .50)),
        })
    return pd.DataFrame(rows)


def test_fold_schedule_has_five_expanding_origins_and_disjoint_final_test():
    weeks = list(range(91))
    schedule = _build_fold_schedule(weeks)
    assert len(schedule) == 5
    assert [len(x["train_weeks"]) for x in schedule] == [30, 36, 42, 48, 54]
    assert all(len(x["calibration_weeks"]) == 8 for x in schedule)
    assert all(len(x["selection_weeks"]) == 6 for x in schedule)
    assert all(len(x["forward_weeks"]) == 6 for x in schedule)
    final = set(range(75, 91))
    for fold in schedule:
        used = set(
            fold["train_weeks"] + fold["calibration_weeks"]
            + fold["selection_weeks"] + fold["forward_weeks"]
        )
        assert not used.intersection(final)
    assert schedule[-1]["forward_weeks"] == list(range(68, 74))


def test_fold_schedule_rejects_unsorted_duplicate_or_insufficient_calendar():
    with pytest.raises(ValueError):
        _build_fold_schedule([0, 1, 1, 2])
    with pytest.raises(ValueError):
        _build_fold_schedule([2, 1, 0])
    with pytest.raises(ValueError):
        _build_fold_schedule(list(range(80)))
    with pytest.raises(ValueError):
        _build_fold_schedule(list(range(40)) + list(range(41, 92)))


def test_assign_fold_splits_keeps_series_aligned_and_final_holdout_unused():
    schedule = _build_fold_schedule(list(range(91)))
    fold = schedule[0]
    frame = pd.DataFrame([
        {"series_id": sid, "week": week, "date": f"2020-W{week:02d}"}
        for sid in ("a", "b") for week in range(91)
    ])
    assigned = _assign_fold_splits(frame, fold)
    by_week = assigned.groupby("week").split.nunique()
    assert by_week.max() == 1
    assert set(assigned.loc[assigned.week.isin(range(75, 91)), "split"]) == {"unused"}
    assert set(assigned.loc[assigned.week.isin(fold["train_weeks"]), "split"]) == {"train"}
    assert set(assigned.loc[assigned.week.isin(fold["forward_weeks"]), "split"]) == {"test"}


def test_v49_single_window_selector_uses_cheapest_proportional_baseline_and_service_floor():
    metrics = pd.DataFrame([
        {"config_id": "q50_proportional", "mean_cost": 10.0, "mean_fill_rate": .60},
        {"config_id": "q75_proportional", "mean_cost": 9.0, "mean_fill_rate": .59},
        {"config_id": "q50_avg_greedy", "mean_cost": 7.0, "mean_fill_rate": .58},
        {"config_id": "q75_marginal_lot", "mean_cost": 8.0, "mean_fill_rate": .59},
        {"config_id": "q90_marginal_lot", "mean_cost": 6.0, "mean_fill_rate": .50},
    ])
    choice = _baseline_and_single_window(metrics)
    assert choice["baseline_config_id"] == "q75_proportional"
    assert choice["candidate_config_id"] == "q75_marginal_lot"
    assert choice["abstained"] is False
    assert choice["service_floor"] == pytest.approx(.585)


def test_robust_selector_requires_ceil_eighty_percent_historical_folds():
    history = []
    for i in range(5):
        fills = {
            "q50_proportional": .60,
            "q75_proportional": .59,
            "q50_avg_greedy": .589 if i < 4 else .56,
            "q75_avg_greedy": .57 if i < 3 else .55,
            "q90_avg_greedy": .56,
            "q50_marginal_lot": .59,
            "q75_marginal_lot": .58,
            "q90_marginal_lot": .57,
        }
        costs = {
            "q50_avg_greedy": 5.0,
            "q75_avg_greedy": 6.0,
            "q50_marginal_lot": 7.0,
            "q75_marginal_lot": 8.0,
            "q90_avg_greedy": 9.0,
            "q90_marginal_lot": 10.0,
        }
        history.append({
            "fold_id": f"F{i+1}",
            "baseline_id": "q75_proportional",
            "metrics": _history_frame(fills, costs),
        })
    choice = _select_historical(history, "q75_proportional", "robust")
    assert choice["candidate_config_id"] == "q50_avg_greedy"
    assert choice["feasible_candidate_count"] >= 1
    assert choice["required_folds"] == 4
    assert choice["pass_count"] == 4


def test_robust_selector_abstains_until_three_prior_folds_even_if_all_available_pass():
    for n_history in (1, 2):
        history = []
        for i in range(n_history):
            fills = {
                "q50_proportional": .60, "q75_proportional": .59,
                "q50_avg_greedy": .59, "q75_avg_greedy": .58,
                "q90_avg_greedy": .57, "q50_marginal_lot": .58,
                "q75_marginal_lot": .57, "q90_marginal_lot": .56,
            }
            costs = {
                "q50_avg_greedy": 1.0, "q75_avg_greedy": 2.0,
                "q90_avg_greedy": 3.0, "q50_marginal_lot": 4.0,
                "q75_marginal_lot": 5.0, "q90_marginal_lot": 6.0,
            }
            history.append({
                "fold_id": f"F{i+1}",
                "baseline_id": "q75_proportional",
                "metrics": _history_frame(fills, costs),
            })
        choice = _select_historical(history, "q75_proportional", "robust")
        assert choice["candidate_config_id"] == "q75_proportional"
        assert choice["abstained"]
        assert choice["history_folds"] == n_history
        assert choice["required_folds"] == 3


def test_robust_selector_abstains_without_history_or_when_no_candidate_qualifies():
    no_history = _select_historical([], "q75_proportional", "robust")
    assert no_history["candidate_config_id"] == "q75_proportional"
    assert no_history["abstained"]
    history = [{
        "fold_id": "F1",
        "baseline_id": "q75_proportional",
        "metrics": _history_frame({
            "q50_proportional": .60, "q75_proportional": .59,
            "q50_avg_greedy": .40, "q75_avg_greedy": .41,
            "q90_avg_greedy": .42, "q50_marginal_lot": .43,
            "q75_marginal_lot": .44, "q90_marginal_lot": .45,
        }),
    }]
    choice = _select_historical(history, "q75_proportional", "robust")
    assert choice["candidate_config_id"] == "q75_proportional"
    assert choice["abstained"]


def test_conservative_selector_requires_three_folds_and_uses_one_sided_lcb():
    too_short = _select_historical([], "q75_proportional", "conservative")
    assert too_short["abstained"]
    history = []
    for i, delta in enumerate([.001, .0005, -.001, .0002]):
        fills = {
            "q50_proportional": .60, "q75_proportional": .59,
            "q50_avg_greedy": .59 + delta,
            "q75_avg_greedy": .55,
            "q90_avg_greedy": .57,
            "q50_marginal_lot": .56,
            "q75_marginal_lot": .58,
            "q90_marginal_lot": .54,
        }
        history.append({
            "fold_id": f"F{i+1}",
            "baseline_id": "q75_proportional",
            "metrics": _history_frame(fills, {
                "q50_avg_greedy": 5.0,
                "q75_avg_greedy": 4.0,
                "q90_avg_greedy": 6.0,
                "q50_marginal_lot": 7.0,
                "q75_marginal_lot": 8.0,
                "q90_marginal_lot": 9.0,
            }),
        })
    choice = _select_historical(history, "q75_proportional", "conservative")
    assert choice["candidate_config_id"] == "q50_avg_greedy"
    assert choice["service_delta_lcb"] is not None
    assert choice["service_delta_lcb"] >= -.005
    assert choice["service_delta_sd"] <= .02


def test_conservative_selector_abstains_when_service_is_too_volatile():
    history = []
    for i, delta in enumerate([.04, -.04, .03, -.03]):
        fills = {
            "q50_proportional": .60, "q75_proportional": .59,
            "q50_avg_greedy": .59 + delta,
            "q75_avg_greedy": .50,
            "q90_avg_greedy": .51,
            "q50_marginal_lot": .52,
            "q75_marginal_lot": .53,
            "q90_marginal_lot": .54,
        }
        history.append({
            "fold_id": f"F{i+1}",
            "baseline_id": "q75_proportional",
            "metrics": _history_frame(fills),
        })
    choice = _select_historical(history, "q75_proportional", "conservative")
    assert choice["candidate_config_id"] == "q75_proportional"
    assert choice["abstained"]


def test_window_panel_excludes_future_series_and_future_actual_records():
    import numpy as np
    dates = pd.date_range("2020-01-05", periods=106, freq="W-SUN")
    rows = []
    for date in dates[:71]:
        rows.append({"series_id": "known", "date": date, "demand": 3.0, "price": 5.0})
    for date in dates[50:71]:
        rows.append({"series_id": "future-entry", "date": date, "demand": 99.0, "price": 9.0})
    panel = pd.DataFrame(rows)
    window = _window_panel(
        panel, train_end_date=dates[40], evaluation_end_date=dates[60],
        global_dates=dates,
    )
    assert set(window.series_id) == {"known"}
    assert window.date.max() == dates[61]
    assert (window.loc[window.date == dates[61], "demand"] == 0).all()
    assert not (window.date > dates[61]).any()
    assert not ((window.date > dates[60]) & (window.demand == 99.0)).any()


def test_replication_seed_is_deterministic_and_common_across_policy_configurations():
    a = _replication_seed("fold", 3, 7, 11)
    b = _replication_seed("fold", 3, 7, 11)
    assert a == b
    assert a != _replication_seed("fold", 4, 7, 11)
    assert a != _replication_seed("final", 3, 7, 11)
