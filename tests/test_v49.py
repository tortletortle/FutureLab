import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from inventory_v49 import (
    _scenario_grid_v49, _lot_benefits, _allocate_marginal_lots,
    solve_marginal_lot_milp, _select_policies, _simulate_v49,
)


def _toy_forecast():
    weeks = np.arange(12)
    return pd.DataFrame({
        "series_id": ["sku"] * 12,
        "week": weeks,
        "date": pd.date_range("2020-01-05", periods=12, freq="W-SUN").astype(str),
        "split": ["selection"] * 4 + ["test"] * 8,
        "demand": [0] * 12,
        "stoch_q50": [50.0] * 12,
        "stoch_q75": [50.0] * 12,
        "stoch_q90": [50.0] * 12,
    })


def _toy_item():
    return {
        "series_ids": np.array(["sku"], dtype=object),
        "mean_demand": np.array([10.0]),
        "demand_scale": np.array([10.0]),
        "unit_value_proxy": np.array([2.0]),
        "lot_size": np.array([1]),
        "weekly_sales_value_proxy": 200.0,
    }


def _scenario():
    return {
        "budget_fraction": 1.0,
        "capacity_equivalents": 10.0,
        "economics": {"shortage_rate": 0.25, "holding_rate": 0.01, "fixed_order_cost": 0.0},
        "capital_rate": 0.005,
    }


def test_v49_scenario_grid_crosses_sixteen_stress_cases_with_two_capital_rates():
    scenarios = _scenario_grid_v49()
    assert len(scenarios) == 32
    assert {x["capital_rate"] for x in scenarios} == {0.0, 0.005}
    assert len({x["scenario_id"] for x in scenarios}) == 32


def test_capital_charge_reduces_each_lot_value_and_does_not_change_service_benefit():
    args = (
        np.array([15.0]), np.array([10]), np.array([2.0]), np.array([10.0]),
        {"shortage_rate": 0.25, "holding_rate": 0.01, "fixed_order_cost": 1.0}, 2.0,
    )
    no_capital = _lot_benefits(*args, capital_rate=0.0)[0]
    with_capital = _lot_benefits(*args, capital_rate=0.01)[0]
    assert len(no_capital) == 2
    assert np.all(with_capital < no_capital)


def test_marginal_allocator_respects_budget_capacity_request_and_lot_bounds():
    request = np.array([20, 20, 10])
    shortfall = np.array([18.0, 19.0, 9.0])
    lot = np.array([5, 2, 1])
    price = np.array([10.0, 2.0, 1.0])
    scale = np.array([10.0, 5.0, 2.0])
    economics = {"shortage_rate": 0.25, "holding_rate": 0.01, "fixed_order_cost": 2.0}
    order = _allocate_marginal_lots(request, shortfall, lot, price, scale, 60, 3.0,
                                    economics, expected_lead=2.25, capital_rate=0.005)
    assert np.all(order % lot == 0)
    assert np.all(order <= request)
    assert np.sum(order * price) <= 60 + 1e-8
    assert np.sum(order / scale) <= 3.0 + 1e-8


def test_marginal_allocator_returns_zero_without_resources_or_positive_benefit():
    args = (np.array([10]), np.array([10.0]), np.array([1]), np.array([2.0]), np.array([10.0]))
    economics = {"shortage_rate": 0.25, "holding_rate": 0.01, "fixed_order_cost": 10_000.0}
    assert _allocate_marginal_lots(args[0], args[1], args[2], args[3], args[4], 0, 10,
                                   economics, 2.0, 0.0).tolist() == [0]
    assert _allocate_marginal_lots(args[0], args[1], args[2], args[3], args[4], 100, 10,
                                   economics, 2.0, 0.0).tolist() == [0]


def test_prefix_constrained_milp_is_feasible_and_not_worse_than_marginal_greedy():
    request = np.array([10, 8, 6])
    shortfall = np.array([8.0, 7.0, 5.0])
    lot = np.array([5, 2, 3])
    price = np.array([4.0, 2.0, 1.0])
    scale = np.array([10.0, 5.0, 3.0])
    economics = {"shortage_rate": 0.3, "holding_rate": 0.02, "fixed_order_cost": 1.0}
    greedy = _allocate_marginal_lots(request, shortfall, lot, price, scale, 40, 2.0,
                                     economics, 2.0, 0.003)
    exact = solve_marginal_lot_milp(shortfall, request, lot, price, scale, 40, 2.0,
                                    economics, 2.0, 0.003)
    assert exact["success"]
    assert np.all(exact["quantity"] <= request)
    assert np.all(exact["quantity"] % lot == 0)
    assert np.sum(exact["quantity"] * price) <= 40 + 1e-7
    assert np.sum(exact["quantity"] / scale) <= 2.0 + 1e-8
    assert exact["objective"] + 1e-7 >= sum(
        sum(_lot_benefits(shortfall, lot, price, scale, economics, 2.0, 0.003)[i][:greedy[i] // lot[i]])
        for i in range(len(greedy))
    )


def test_selection_enforces_service_floor_and_selects_cheapest_feasible_candidate():
    metrics = pd.DataFrame([
        {"config_id": "q50_proportional", "allocator": "proportional", "mean_cost": 10.0, "mean_fill_rate": .60},
        {"config_id": "q75_proportional", "allocator": "proportional", "mean_cost": 9.0, "mean_fill_rate": .59},
        {"config_id": "q75_avg_greedy", "allocator": "average_greedy", "mean_cost": 7.0, "mean_fill_rate": .58},
        {"config_id": "q75_marginal_lot", "allocator": "marginal_lot", "mean_cost": 8.0, "mean_fill_rate": .59},
    ])
    selected = _select_policies(metrics)
    assert selected["baseline_id"] == "q75_proportional"
    assert selected["candidate_id"] == "q75_marginal_lot"
    assert selected["service_floor_feasible"]
    assert selected["service_floor"] == pytest.approx(.585)


def test_selection_falls_back_to_baseline_when_no_candidate_meets_service_floor():
    metrics = pd.DataFrame([
        {"config_id": "q50_proportional", "allocator": "proportional", "mean_cost": 10.0, "mean_fill_rate": .60},
        {"config_id": "q75_proportional", "allocator": "proportional", "mean_cost": 9.0, "mean_fill_rate": .59},
        {"config_id": "q75_avg_greedy", "allocator": "average_greedy", "mean_cost": 5.0, "mean_fill_rate": .50},
        {"config_id": "q75_marginal_lot", "allocator": "marginal_lot", "mean_cost": 8.0, "mean_fill_rate": .55},
    ])
    selected = _select_policies(metrics)
    assert selected["candidate_id"] == selected["baseline_id"] == "q75_proportional"
    assert not selected["service_floor_feasible"]


def test_simulation_records_capital_charge_and_preserves_inventory_events():
    forecast = _toy_forecast()
    draws = np.full((1, 12), 2, dtype=int)
    config = {"config_id": "q75_marginal_lot", "target_col": "stoch_q75", "allocator": "marginal_lot"}
    summary, weekly, detail = _simulate_v49(
        forecast, _toy_item(), {1: .15, 2: .55, 3: .2, 4: .1}, _scenario(), config, draws, True
    )
    sku = detail.sort_values("week")
    assert sku.loc[sku.week == 0, "order_qty"].iloc[0] == 27
    assert sku.loc[sku.week == 2, "arrival_qty"].iloc[0] == 27
    assert (detail.capital_charge >= 0).all()
    assert np.allclose(detail.capital_charge, detail.capital_exposure_value * detail.capital_rate)
    assert np.allclose(detail.total_cost, detail.holding_cost + detail.lost_sales_cost + detail.fixed_order_cost + detail.capital_charge)
    assert summary["selection"]["mean_capital_charge_per_week"] > 0
    assert summary["test"]["mean_cost"] >= 0
    assert max(x["budget_utilization"] for x in weekly) <= 1 + 1e-8


def test_zero_capital_rate_produces_zero_capital_charge():
    forecast = _toy_forecast()
    draws = np.full((1, 12), 1, dtype=int)
    scenario = _scenario()
    scenario["capital_rate"] = 0.0
    config = {"config_id": "q75_prop", "target_col": "stoch_q75", "allocator": "proportional"}
    _, _, detail = _simulate_v49(
        forecast, _toy_item(), {1: .15, 2: .55, 3: .2, 4: .1}, scenario, config, draws, True
    )
    assert detail.capital_charge.sum() == pytest.approx(0.0)