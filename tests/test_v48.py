import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from inventory_v48 import (
    _weighted_quantile, _validate_lead_probabilities, _item_parameters,
    _allocate_proportional, _allocate_marginal_greedy, _simulate,
    solve_allocation_milp, _allocation_objective,
)


def test_weighted_quantile_respects_order_and_weights():
    assert _weighted_quantile(np.array([1, 2, 3]), np.array([0.8, 0.1, 0.1]), 0.5) == 1
    assert _weighted_quantile(np.array([1, 2, 3]), np.array([0.1, 0.1, 0.8]), 0.5) == 3


@pytest.mark.parametrize("probabilities", [
    {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
    {1: -0.1, 2: 0.4, 3: 0.4, 4: 0.3},
    {1: 0.2, 2: 0.3, 3: 0.5},
])
def test_invalid_lead_distribution_rejected(probabilities):
    with pytest.raises(ValueError):
        _validate_lead_probabilities(probabilities)


def test_valid_lead_distribution_expected_value():
    p = _validate_lead_probabilities({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4})
    assert sum(k * v for k, v in p.items()) == pytest.approx(3.0)


def test_item_parameters_use_training_period_prices_only():
    d = pd.DataFrame({
        "series_id": ["a", "a", "b", "b"],
        "date": pd.to_datetime(["2020-01-05", "2020-01-12", "2020-01-05", "2020-01-12"]),
        "demand": [10, 20, 5, 5],
        "split": ["train", "selection", "train", "selection"],
    })
    forecast = pd.DataFrame({"series_id": ["a", "b"], "week": [2, 2]})
    panel = pd.DataFrame({
        "series_id": ["a", "a", "b", "b"],
        "date": pd.to_datetime(["2020-01-05", "2020-01-12", "2020-01-05", "2020-01-12"]),
        "price": [2.0, 200.0, 4.0, 400.0],
    })
    item = _item_parameters(panel, d, forecast)
    assert item["unit_value_proxy"].tolist() == [2.0, 4.0]
    assert item["mean_demand"].tolist() == [10.0, 5.0]


def test_proportional_allocator_respects_budget_capacity_and_lots():
    request = np.array([20, 20])
    lot = np.array([5, 2])
    price = np.array([10.0, 2.0])
    scale = np.array([10.0, 5.0])
    order = _allocate_proportional(request, lot, price, scale, budget=50, capacity_room=2.0)
    assert np.all(order % lot == 0)
    assert np.all(order <= request)
    assert np.sum(order * price) <= 50 + 1e-8
    assert np.sum(order / scale) <= 2.0 + 1e-8


def test_proportional_allocator_returns_zero_without_resources():
    order = _allocate_proportional(np.array([10]), np.array([2]), np.array([1.0]), np.array([10.0]), 0, 10)
    assert order.tolist() == [0]


def test_marginal_greedy_respects_budget_capacity_and_lots():
    request = np.array([20, 20, 10])
    shortfall = np.array([18.0, 19.0, 9.0])
    lot = np.array([5, 2, 1])
    price = np.array([10.0, 2.0, 1.0])
    scale = np.array([10.0, 5.0, 2.0])
    economics = {"shortage_rate": 0.25, "holding_rate": 0.01, "fixed_order_cost": 2.0}
    order = _allocate_marginal_greedy(request, shortfall, lot, price, scale, 60, 3.0, economics)
    assert np.all(order % lot == 0)
    assert np.all(order <= request)
    assert np.sum(order * price) <= 60 + 1e-8
    assert np.sum(order / scale) <= 3.0 + 1e-8


def test_greedy_allocator_prefers_higher_value_per_resource_when_tight():
    request = np.array([10, 10])
    shortfall = np.array([10.0, 10.0])
    lot = np.array([5, 5])
    price = np.array([10.0, 1.0])
    scale = np.array([10.0, 10.0])
    economics = {"shortage_rate": 0.25, "holding_rate": 0.0, "fixed_order_cost": 0.0}
    order = _allocate_marginal_greedy(request, shortfall, lot, price, scale, 6.0, 1.0, economics)
    assert order.tolist() == [0, 5]


def test_milp_allocation_is_feasible_and_not_worse_than_greedy():
    shortfall = np.array([8.0, 6.0, 3.0])
    lot = np.array([5, 2, 1])
    price = np.array([10.0, 2.0, 1.0])
    scale = np.array([10.0, 5.0, 2.0])
    economics = {"shortage_rate": 0.25, "holding_rate": 0.01, "fixed_order_cost": 2.0}
    greedy = _allocate_marginal_greedy(
        np.ceil(shortfall / lot).astype(int) * lot, shortfall, lot, price, scale, 50, 2.0, economics
    )
    exact = solve_allocation_milp(shortfall, lot, price, scale, 50, 2.0, economics)
    assert exact["success"]
    assert np.sum(exact["quantity"] * price) <= 50 + 1e-7
    assert np.sum(exact["quantity"] / scale) <= 2.0 + 1e-7
    assert np.all(exact["quantity"] % lot == 0)
    greedy_obj = _allocation_objective(greedy, shortfall, lot, price, economics)
    assert exact["objective"] + 1e-7 >= greedy_obj


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


def _toy_scenario():
    return {
        "budget_fraction": 1.0,
        "capacity_equivalents": 10.0,
        "economics": {"shortage_rate": 0.25, "holding_rate": 0.01, "fixed_order_cost": 0.0},
    }


def test_random_lead_order_arrives_at_sampled_period():
    forecast = _toy_forecast()
    draws = np.full((1, 12), 2, dtype=int)
    probs = {1: 0.15, 2: 0.55, 3: 0.2, 4: 0.1}
    cfg = {"config_id": "q75_greedy", "target_col": "stoch_q75", "allocator": "marginal_greedy"}
    summary, weekly, detail = _simulate(forecast, _toy_item(), probs, _toy_scenario(), cfg, draws, True)
    sku = detail.sort_values("week")
    assert sku.loc[sku.week == 0, "order_qty"].iloc[0] == 27
    assert sku.loc[sku.week == 2, "arrival_qty"].iloc[0] == 27
    assert summary["selection"]["mean_cost"] >= 0


def test_terminal_test_tail_disables_new_orders():
    forecast = _toy_forecast()
    forecast["demand"] = 100
    draws = np.full((1, 12), 4, dtype=int)
    probs = {1: 0.15, 2: 0.55, 3: 0.2, 4: 0.1}
    cfg = {"config_id": "q75_greedy", "target_col": "stoch_q75", "allocator": "marginal_greedy"}
    _, _, detail = _simulate(forecast, _toy_item(), probs, _toy_scenario(), cfg, draws, True)
    test = detail[detail.split == "test"]
    assert test[test.week.isin(sorted(test.week.unique())[-4:])].order_qty.sum() == 0


def test_simulation_never_exceeds_budget_or_capacity():
    forecast = _toy_forecast()
    forecast["demand"] = 25
    item = _toy_item()
    scenario = _toy_scenario()
    scenario["budget_fraction"] = 0.2
    scenario["capacity_equivalents"] = 3.0
    draws = np.full((1, 12), 3, dtype=int)
    cfg = {"config_id": "q75_greedy", "target_col": "stoch_q75", "allocator": "marginal_greedy"}
    _, weekly, detail = _simulate(
        forecast, item, {1: .15, 2: .55, 3: .2, 4: .1}, scenario, cfg, draws, True
    )
    assert max(x["budget_utilization"] for x in weekly) <= 1 + 1e-8
    assert max(x["capacity_usage_after"] / x["capacity_limit"] for x in weekly) <= 1 + 1e-8
    assert (detail.order_qty % detail.lot_size == 0).all()


def test_zero_budget_means_no_new_orders():
    forecast = _toy_forecast()
    draws = np.full((1, 12), 1, dtype=int)
    scenario = _toy_scenario()
    scenario["budget_fraction"] = 0.0
    cfg = {"config_id": "q75_prop", "target_col": "stoch_q75", "allocator": "proportional"}
    _, _, detail = _simulate(
        forecast, _toy_item(), {1: .15, 2: .55, 3: .2, 4: .1}, scenario, cfg, draws, True
    )
    assert detail.order_qty.sum() == 0


def test_simulation_rejects_invalid_lead_draws():
    forecast = _toy_forecast()
    draws = np.full((1, 12), 5, dtype=int)
    cfg = {"config_id": "q75_prop", "target_col": "stoch_q75", "allocator": "proportional"}
    with pytest.raises(ValueError):
        _simulate(forecast, _toy_item(), {1: .15, 2: .55, 3: .2, 4: .1}, _toy_scenario(), cfg, draws)
