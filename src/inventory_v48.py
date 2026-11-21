"""V0.48 stochastic lead-time and constrained replenishment research module.

Demand is historical *recorded sales*, not latent demand. Lead-time distributions,
budget limits, capacity, MOQ/pack sizes and economic weights are explicit scenario
assumptions because the source retailer data has no supplier or inventory records.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.ensemble import HistGradientBoostingRegressor

from inventory_v47 import _fit_forecasts
from retail_v46 import features, COLS, SEED

LEAD_DISTRIBUTIONS = {
    # Probabilities are scenario assumptions, not estimates from the retailer.
    "moderate": {1: 0.15, 2: 0.55, 3: 0.20, 4: 0.10},
    "volatile": {1: 0.10, 2: 0.35, 3: 0.35, 4: 0.20},
}
BUDGET_FRACTIONS = (0.55, 0.80)
CAPACITY_EQUIVALENTS = (300.0, 400.0)
ECONOMIC_SCENARIOS = {
    "balanced": {"shortage_rate": 0.25, "holding_rate": 0.01, "fixed_order_cost": 2.0},
    "capital_pressure": {"shortage_rate": 0.25, "holding_rate": 0.03, "fixed_order_cost": 2.0},
}
CONFIGS = (
    {"config_id": "q50_proportional", "target_col": "stoch_q50", "allocator": "proportional", "family": "proportional"},
    {"config_id": "q75_proportional", "target_col": "stoch_q75", "allocator": "proportional", "family": "proportional"},
    {"config_id": "q50_greedy", "target_col": "stoch_q50", "allocator": "marginal_greedy", "family": "priority"},
    {"config_id": "q75_greedy", "target_col": "stoch_q75", "allocator": "marginal_greedy", "family": "priority"},
    {"config_id": "q90_greedy", "target_col": "stoch_q90", "allocator": "marginal_greedy", "family": "priority"},
)
MAX_LEAD = 4
REPLICATIONS = 20
BOOTSTRAP_REPS = 2000
LOT_FRACTION = 0.10
SEED_V48 = 48048


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or len(values) == 0:
        raise ValueError("weighted quantile requires non-empty aligned vectors")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid weighted quantile values or weights")
    if not 0 <= q <= 1:
        raise ValueError("quantile must be in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    v, w = values[order], weights[order]
    idx = int(np.searchsorted(np.cumsum(w), q * w.sum(), side="left"))
    return float(v[min(idx, len(v) - 1)])


def _validate_lead_probabilities(probabilities: dict[int, float]) -> dict[int, float]:
    if set(probabilities) != {1, 2, 3, 4}:
        raise ValueError("lead-time distribution must define periods 1..4")
    p = {int(k): float(v) for k, v in probabilities.items()}
    if any(not np.isfinite(v) or v < 0 for v in p.values()) or not np.isclose(sum(p.values()), 1.0):
        raise ValueError("lead-time probabilities must be nonnegative and sum to one")
    return p


def _fit_stochastic_targets(d: pd.DataFrame, forecast: pd.DataFrame, probabilities: dict[int, float]) -> tuple[pd.DataFrame, dict]:
    """Calibrate a mixture distribution over 1..4 week cumulative demand.

    HGB is fitted on train only. Residual windows are collected on calibration only.
    For each target row, the mixture weights are the assumed lead-time probabilities.
    """
    probabilities = _validate_lead_probabilities(probabilities)
    train = d[d.split == "train"].copy()
    cal = d[d.split == "calibration"].copy()
    if train.empty or cal.empty or forecast.empty:
        raise ValueError("train, calibration and target partitions must be non-empty")

    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=30,
        learning_rate=0.08, l2_regularization=2, random_state=SEED,
        categorical_features=[True] + [False] * (len(COLS) - 1),
    ).fit(train[COLS], np.log1p(train.demand))
    cal["residual"] = cal.demand.to_numpy(float) - np.expm1(model.predict(cal[COLS])).clip(0)
    train_mean = train.groupby("series_id").demand.mean()
    edges = np.quantile(train_mean.to_numpy(float), [1 / 3, 2 / 3])
    group_map = {
        sid: int(np.searchsorted(edges, value, side="right"))
        for sid, value in train_mean.items()
    }
    cal["group_id"] = cal.series_id.map(group_map).astype(int)
    target = forecast.copy()
    target["group_id"] = target.series_id.map(group_map).astype(int)

    residual_windows: dict[int, dict[int, np.ndarray]] = {g: {} for g in range(3)}
    pooled_windows: dict[int, np.ndarray] = {}
    counts = {}
    for lead in range(1, MAX_LEAD + 1):
        pooled = []
        by_group = {g: [] for g in range(3)}
        for sid, g in cal.sort_values(["series_id", "week"]).groupby("series_id", sort=False):
            g = g.sort_values("week")
            residual = g.residual.to_numpy(float)
            scale = g.scale.to_numpy(float)
            weeks = g.week.to_numpy(int)
            if len(g) < lead:
                continue
            for start in range(len(g) - lead + 1):
                if weeks[start + lead - 1] - weeks[start] != lead - 1:
                    continue
                denom = max(float(scale[start:start + lead].sum()), 1.0)
                value = float(residual[start:start + lead].sum() / denom)
                pooled.append(value)
                by_group[group_map[sid]].append(value)
        if len(pooled) < 30:
            raise ValueError(f"insufficient calibration windows for lead={lead}")
        pooled_arr = np.asarray(pooled, dtype=float)
        pooled_windows[lead] = pooled_arr
        counts[lead] = {"pooled": len(pooled_arr), "groups": {}}
        for group in range(3):
            values = np.asarray(by_group[group], dtype=float)
            if len(values) < 30:
                values = pooled_arr
            residual_windows[group][lead] = values
            counts[lead]["groups"][str(group)] = len(by_group[group])

    # Build weighted scenario pools. Each lead-time distribution contributes its
    # own calibrated cumulative residuals with the predeclared probability weight.
    mixture_meta = {}
    pools = {}
    for group in range(3):
        vals, weights, leads = [], [], []
        for lead, prob in probabilities.items():
            residuals = residual_windows[group][lead]
            vals.append(residuals)
            weights.append(np.full(len(residuals), prob / len(residuals), dtype=float))
            leads.append(np.full(len(residuals), lead, dtype=int))
        pools[group] = (np.concatenate(vals), np.concatenate(weights), np.concatenate(leads))

    for q in (0.50, 0.75, 0.90):
        qtag = int(q * 100)
        target[f"stoch_q{qtag}"] = np.nan
    for idx, row in target.iterrows():
        residuals, weights, leads = pools[int(row.group_id)]
        scenarios = leads * float(row.hgb_mean) + leads * float(row.scale) * residuals
        for q in (0.50, 0.75, 0.90):
            target.at[idx, f"stoch_q{int(q * 100)}"] = max(0.0, _weighted_quantile(scenarios, weights, q))

    meta = {
        "lead_time_probabilities": {str(k): v for k, v in probabilities.items()},
        "expected_lead_time": float(sum(k * v for k, v in probabilities.items())),
        "train_only_model_fit": True,
        "calibration_only_residual_windows": True,
        "training_volume_group_edges": edges.tolist(),
        "window_counts": counts,
        "target_definition": "weighted mixture quantile of L*hgb_mean + L*scale*normalized cumulative calibration residual, L~declared lead-time distribution",
        "limitations": "The one-step forecast and scale are held constant within the lead-time horizon; this is a calibrated approximation, not a direct multi-horizon model.",
    }
    return target, meta


def _item_parameters(panel: pd.DataFrame, d: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Create training-only demand scale, pack-size and retail-value proxies."""
    train = d[d.split == "train"]
    series_ids = np.array(sorted(forecast.series_id.unique()), dtype=object)
    means = train.groupby("series_id").demand.mean().reindex(series_ids).fillna(0.0).to_numpy(float)
    means = np.maximum(means, 0.0)
    demand_scale = np.maximum(means, 1.0)

    p = panel[["series_id", "date", "price"]].copy()
    p["date"] = pd.to_datetime(p.date, errors="raise")
    p["price"] = pd.to_numeric(p.price, errors="coerce")
    cutoff = pd.to_datetime(train.date.max())
    p = p[(p.date <= cutoff) & np.isfinite(p.price) & (p.price > 0)]
    global_price = float(p.price.median()) if len(p) else 1.0
    prices = p.groupby("series_id").price.median().reindex(series_ids).fillna(global_price).to_numpy(float)
    prices = np.maximum(prices, 0.01)
    lots = np.maximum(1, np.ceil(LOT_FRACTION * np.maximum(means, 1.0))).astype(int)
    weekly_sales_value = float(np.sum(means * prices))
    if not np.isfinite(weekly_sales_value) or weekly_sales_value <= 0:
        raise ValueError("invalid training-only weekly sales value proxy")
    return {
        "series_ids": series_ids,
        "mean_demand": means,
        "demand_scale": demand_scale,
        "unit_value_proxy": prices,
        "lot_size": lots,
        "weekly_sales_value_proxy": weekly_sales_value,
        "price_cutoff": str(cutoff.date()),
        "price_source": "training-period median recorded unit selling price; not supplier acquisition cost",
        "lot_rule": f"ceil({LOT_FRACTION:.0%} of training mean weekly recorded sales), minimum 1 unit",
    }


def _allocate_proportional(request: np.ndarray, lot: np.ndarray, price: np.ndarray,
                           demand_scale: np.ndarray, budget: float, capacity_room: float) -> np.ndarray:
    request = np.asarray(request, dtype=int)
    if budget <= 0 or capacity_room <= 0 or request.sum() <= 0:
        return np.zeros_like(request)
    value = float(np.sum(request * price))
    cap = float(np.sum(request / demand_scale))
    factor = min(1.0, budget / value if value > 0 else 1.0,
                 capacity_room / cap if cap > 0 else 1.0)
    lots = np.floor((request * factor) / lot + 1e-10).astype(int)
    order = lots * lot
    # Defensive correction for floating point edge cases.
    while float(np.sum(order * price)) > budget + 1e-7:
        candidates = np.flatnonzero(order > 0)
        if not len(candidates):
            break
        i = candidates[np.argmax(order[candidates] * price[candidates])]
        order[i] -= lot[i]
    while float(np.sum(order / demand_scale)) > capacity_room + 1e-9:
        candidates = np.flatnonzero(order > 0)
        if not len(candidates):
            break
        i = candidates[np.argmax(order[candidates] / demand_scale[candidates])]
        order[i] -= lot[i]
    return order


def _average_lot_benefit(shortfall: np.ndarray, lot: np.ndarray, price: np.ndarray,
                         costs: dict) -> np.ndarray:
    """Linearized average benefit per requested lot, including amortized setup cost."""
    shortfall = np.asarray(shortfall, dtype=float)
    lot = np.asarray(lot, dtype=int)
    max_lots = np.ceil(np.divide(shortfall, lot, out=np.zeros_like(shortfall), where=lot > 0)).astype(int)
    requested = max_lots * lot
    filled = np.minimum(requested, shortfall)
    excess = np.maximum(requested - shortfall, 0)
    variable_benefit = (
        costs["shortage_rate"] * price * filled
        - costs["holding_rate"] * price * excess
    )
    avg = np.divide(
        variable_benefit - costs["fixed_order_cost"],
        max_lots,
        out=np.zeros_like(variable_benefit, dtype=float),
        where=max_lots > 0,
    )
    return avg


def _allocate_marginal_greedy(request: np.ndarray, shortfall: np.ndarray, lot: np.ndarray,
                              price: np.ndarray, demand_scale: np.ndarray,
                              budget: float, capacity_room: float, costs: dict) -> np.ndarray:
    """Two-resource greedy integer-lot allocation by average marginal value density."""
    request = np.asarray(request, dtype=int)
    lot = np.asarray(lot, dtype=int)
    if budget <= 0 or capacity_room <= 0 or request.sum() <= 0:
        return np.zeros_like(request)
    max_lots = np.floor_divide(request, lot)
    benefit = _average_lot_benefit(shortfall, lot, price, costs)
    burden = (
        (lot * price) / max(budget, 1e-9)
        + (lot / demand_scale) / max(capacity_room, 1e-9)
    )
    score = np.divide(benefit, burden, out=np.full_like(benefit, -np.inf), where=burden > 0)
    order = np.zeros_like(request)
    remaining_budget = float(budget)
    remaining_capacity = float(capacity_room)
    # Stable tie-break by the sorted series order; every accepted SKU consumes
    # as many feasible lots as possible before the next lower-density SKU.
    for i in np.lexsort((np.arange(len(score)), -score)):
        if benefit[i] <= 0 or max_lots[i] <= 0:
            continue
        budget_lots = int(math.floor((remaining_budget + 1e-8) / (lot[i] * price[i])))
        capacity_lots = int(math.floor((remaining_capacity + 1e-10) / (lot[i] / demand_scale[i])))
        take = min(int(max_lots[i]), budget_lots, capacity_lots)
        if take <= 0:
            continue
        qty = take * int(lot[i])
        order[i] = qty
        remaining_budget -= qty * float(price[i])
        remaining_capacity -= qty / float(demand_scale[i])
    return order


def _simulate(
    forecast: pd.DataFrame,
    item: dict,
    probabilities: dict[int, float],
    scenario: dict,
    config: dict,
    lead_draws: np.ndarray,
    keep_detail: bool = False,
) -> tuple[dict, list[dict], pd.DataFrame | None]:
    """Simulate one strategy and one common-random-number lead-time path."""
    probabilities = _validate_lead_probabilities(probabilities)
    ids = item["series_ids"]
    weeks = np.array(sorted(forecast.week.unique()), dtype=int)
    n_series, n_periods = len(ids), len(weeks)
    index = pd.MultiIndex.from_product([ids, weeks], names=["series_id", "week"])
    aligned = forecast.set_index(["series_id", "week"]).reindex(index).reset_index()
    required = [config["target_col"], "demand", "date", "split"]
    if aligned[required].isna().any().any():
        raise ValueError("forecast panel is incomplete or missing required fields")
    if lead_draws.shape != (n_series, n_periods) or not np.isin(lead_draws, [1, 2, 3, 4]).all():
        raise ValueError("lead-time draw matrix has invalid shape or values")

    targets = np.maximum(aligned[config["target_col"]].to_numpy(float).reshape(n_series, n_periods), 0)
    demands = np.ceil(np.maximum(aligned.demand.to_numpy(float).reshape(n_series, n_periods), 0)).astype(int)
    dates = aligned.date.astype(str).to_numpy().reshape(n_series, n_periods)
    splits = aligned.split.astype(str).to_numpy().reshape(n_series, n_periods)
    if not (splits == splits[0:1, :]).all():
        raise ValueError("split boundaries must be aligned across all series")

    expected_lead = float(sum(k * v for k, v in probabilities.items()))
    mean_demand = item["mean_demand"]
    demand_scale = item["demand_scale"]
    price = item["unit_value_proxy"]
    lot = item["lot_size"]
    initial_stock = np.ceil(expected_lead * mean_demand).astype(int)
    on_hand = initial_stock.copy()
    pipeline_total = np.zeros(n_series, dtype=int)
    due = np.zeros((n_series, n_periods + MAX_LEAD + 1), dtype=int)

    test_weeks = weeks[splits[0] == "test"]
    if len(test_weeks) == 0:
        raise ValueError("test split is required")
    last_order_week = int(test_weeks.max() - MAX_LEAD)
    budget = float(scenario["budget_fraction"] * item["weekly_sales_value_proxy"])
    capacity_limit = float(scenario["capacity_equivalents"])
    costs = scenario["economics"]

    detail_rows = []
    weekly_rows = []
    total_rows = []
    for t, week in enumerate(weeks):
        arrival = due[:, t].copy()
        on_hand += arrival
        pipeline_total -= arrival
        if (pipeline_total < 0).any():
            raise AssertionError("pipeline became negative after receipt")
        on_hand_start = on_hand.copy()
        pipeline_before = pipeline_total.copy()
        inventory_position = on_hand_start + pipeline_before
        shortfall = np.maximum(targets[:, t] - inventory_position, 0.0)
        raw_request = np.ceil(shortfall).astype(int)
        requested = np.where(raw_request > 0, np.ceil(raw_request / lot).astype(int) * lot, 0)
        order_eligible = (splits[0, t] != "test") or int(week) <= last_order_week
        if not order_eligible:
            requested_for_allocation = np.zeros_like(requested)
        else:
            requested_for_allocation = requested.copy()

        capacity_before = float(np.sum(inventory_position / demand_scale))
        capacity_room = max(0.0, capacity_limit - capacity_before)
        if config["allocator"] == "proportional":
            order = _allocate_proportional(
                requested_for_allocation, lot, price, demand_scale, budget, capacity_room
            )
        elif config["allocator"] == "marginal_greedy":
            order = _allocate_marginal_greedy(
                requested_for_allocation, shortfall, lot, price, demand_scale,
                budget, capacity_room, costs,
            )
        else:
            raise ValueError("unknown allocator")

        order_lead = np.where(order > 0, lead_draws[:, t], 0).astype(int)
        for i in np.flatnonzero(order > 0):
            due_week_idx = t + int(order_lead[i])
            if due_week_idx >= n_periods:
                raise AssertionError("order placed too late to observe its arrival")
            due[i, due_week_idx] += int(order[i])
        pipeline_total += order
        pipeline_after = pipeline_total.copy()

        fulfilled = np.minimum(on_hand_start, demands[:, t])
        lost = demands[:, t] - fulfilled
        on_hand = on_hand_start - fulfilled
        holding_cost = costs["holding_rate"] * on_hand * price
        lost_sales_cost = costs["shortage_rate"] * lost * price
        fixed_order_cost = costs["fixed_order_cost"] * (order > 0)
        total_cost = holding_cost + lost_sales_cost + fixed_order_cost

        budget_spend = float(np.sum(order * price))
        capacity_after = float(np.sum((on_hand_start + pipeline_before + order) / demand_scale))
        if budget_spend > budget + 1e-6:
            raise AssertionError("weekly procurement budget exceeded")
        if capacity_after > capacity_limit + 1e-8:
            raise AssertionError(f"inventory capacity exceeded: {capacity_after} > {capacity_limit}")
        if np.any(order % lot != 0):
            raise AssertionError("order violates pack multiple")
        if np.any(fulfilled + lost != demands[:, t]):
            raise AssertionError("demand conservation failed")

        split = str(splits[0, t])
        weekly_rows.append({
            "week": int(week), "split": split,
            "mean_total_cost": float(total_cost.mean()),
            "fill_rate": float(fulfilled.sum() / demands[:, t].sum()) if demands[:, t].sum() else 1.0,
            "lost_units": int(lost.sum()), "holding_value_proxy": float(holding_cost.sum()),
            "lost_sales_value_proxy": float(lost_sales_cost.sum()),
            "order_value_proxy": budget_spend, "budget_limit_proxy": budget,
            "budget_utilization": budget_spend / budget if budget else 0.0,
            "capacity_usage_after": capacity_after, "capacity_limit": capacity_limit,
        })
        if keep_detail:
            for i, sid in enumerate(ids):
                detail_rows.append({
                    "series_id": sid, "week": int(week), "date": dates[i, t], "split": split,
                    "config_id": config["config_id"], "sampled_lead_time": int(lead_draws[i, t]),
                    "order_lead_time": int(order_lead[i]), "arrival_qty": int(arrival[i]),
                    "initial_stock": int(initial_stock[i]), "on_hand_start": int(on_hand_start[i]),
                    "pipeline_before": int(pipeline_before[i]),
                    "inventory_position": int(inventory_position[i]),
                    "target_level": float(targets[i, t]), "shortfall": float(shortfall[i]),
                    "requested_qty": int(requested[i]), "order_eligible": bool(order_eligible),
                    "lot_size": int(lot[i]), "order_qty": int(order[i]),
                    "arrival_week": int(week + order_lead[i]) if order[i] > 0 else -1,
                    "pipeline_after": int(pipeline_after[i]), "demand": int(demands[i, t]),
                    "fulfilled": int(fulfilled[i]), "lost_sales": int(lost[i]),
                    "on_hand_end": int(on_hand[i]), "unit_value_proxy": float(price[i]),
                    "mean_demand": float(demand_scale[i]),
                    "holding_cost": float(holding_cost[i]),
                    "lost_sales_cost": float(lost_sales_cost[i]),
                    "fixed_order_cost": float(fixed_order_cost[i]),
                    "total_cost": float(total_cost[i]),
                    "budget_spend_week": budget_spend, "budget_limit_week": budget,
                    "capacity_usage_after_week": capacity_after, "capacity_limit": capacity_limit,
                    "capacity_room_before_order": capacity_room,
                    "shortage_rate": costs["shortage_rate"], "holding_rate": costs["holding_rate"],
                    "fixed_order_cost_rate": costs["fixed_order_cost"],
                })

    weekly = pd.DataFrame(weekly_rows)
    summaries = {}
    for split in ("selection", "test"):
        w = weekly[weekly.split == split]
        summaries[split] = {
            "mean_cost": float(w.mean_total_cost.mean()),
            "fill_rate": float(1 - w.lost_units.sum() / max(int(np.sum(demands[:, splits[0] == split])), 1)),
            "lost_units": int(w.lost_units.sum()),
            "mean_order_value_proxy": float(w.order_value_proxy.mean()),
            "mean_budget_utilization": float(w.budget_utilization.mean()),
            "max_capacity_utilization": float((w.capacity_usage_after / w.capacity_limit).max()),
        }
    return summaries, weekly_rows, pd.DataFrame(detail_rows) if keep_detail else None


def _block_bootstrap_ci(weekly_delta: np.ndarray, seed: int, reps: int = BOOTSTRAP_REPS, block: int = 4) -> list[float]:
    values = np.asarray(weekly_delta, dtype=float)
    if values.ndim != 1 or len(values) < block or not np.isfinite(values).all():
        raise ValueError("invalid weekly delta vector")
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, size=(reps, int(math.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]
    return [float(x) for x in np.quantile(values[idx].mean(axis=1), [0.025, 0.975])]


def solve_allocation_milp(
    shortfall: np.ndarray, lot_size: np.ndarray, unit_value: np.ndarray,
    demand_scale: np.ndarray, budget: float, capacity_room: float, economics: dict,
) -> dict:
    """Exact integer solution of the one-period *linearized* lot-allocation subproblem."""
    shortfall = np.asarray(shortfall, dtype=float)
    lot_size = np.asarray(lot_size, dtype=int)
    unit_value = np.asarray(unit_value, dtype=float)
    demand_scale = np.asarray(demand_scale, dtype=float)
    n = len(shortfall)
    if not (lot_size.shape == unit_value.shape == demand_scale.shape == shortfall.shape):
        raise ValueError("MILP input vectors must align")
    if (shortfall < 0).any() or (lot_size <= 0).any() or (unit_value <= 0).any() or (demand_scale <= 0).any():
        raise ValueError("invalid MILP allocation inputs")
    if budget < 0 or capacity_room < 0:
        raise ValueError("resource limits must be nonnegative")
    max_lots = np.ceil(np.divide(shortfall, lot_size, out=np.zeros_like(shortfall), where=lot_size > 0)).astype(int)
    benefit = _average_lot_benefit(shortfall, lot_size, unit_value, economics)
    if max_lots.sum() == 0 or budget <= 0 or capacity_room <= 0:
        return {"success": True, "quantity": np.zeros(n, dtype=int), "objective": 0.0, "status": 0}
    matrix = np.vstack([lot_size * unit_value, lot_size / demand_scale])
    constraints = LinearConstraint(matrix, np.full(2, -np.inf), np.array([budget, capacity_room], dtype=float))
    result = milp(
        c=-benefit,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), max_lots.astype(float)),
        constraints=constraints,
        options={"time_limit": 1.0, "mip_rel_gap": 1e-8},
    )
    if not result.success or result.x is None:
        return {"success": False, "quantity": np.zeros(n, dtype=int), "objective": None, "status": int(result.status)}
    lots = np.rint(result.x).astype(int)
    qty = lots * lot_size
    objective = float(np.sum(benefit * lots))
    feasible = (
        np.sum(qty * unit_value) <= budget + 1e-6
        and np.sum(qty / demand_scale) <= capacity_room + 1e-8
        and np.all(lots <= max_lots)
    )
    if not feasible:
        return {"success": False, "quantity": np.zeros(n, dtype=int), "objective": None, "status": int(result.status)}
    return {"success": True, "quantity": qty, "objective": objective, "status": int(result.status)}


def _allocation_objective(order: np.ndarray, shortfall: np.ndarray, lot: np.ndarray,
                          price: np.ndarray, costs: dict) -> float:
    benefit = _average_lot_benefit(shortfall, lot, price, costs)
    lots = np.floor_divide(np.asarray(order, dtype=int), np.asarray(lot, dtype=int))
    return float(np.sum(benefit * lots))


def _scenario_grid() -> list[dict]:
    scenarios = []
    for dist_name, probs in LEAD_DISTRIBUTIONS.items():
        for budget_fraction in BUDGET_FRACTIONS:
            for capacity in CAPACITY_EQUIVALENTS:
                for economics_name, economics in ECONOMIC_SCENARIOS.items():
                    scenarios.append({
                        "scenario_id": f"{dist_name}_budget{int(budget_fraction*100)}_cap{capacity:g}_{economics_name}",
                        "lead_distribution": dist_name,
                        "lead_probabilities": probs,
                        "budget_fraction": budget_fraction,
                        "capacity_equivalents": capacity,
                        "economics_name": economics_name,
                        "economics": economics,
                    })
    return scenarios


def run(panel: pd.DataFrame, quality: dict, out: Path, replications: int = REPLICATIONS) -> dict:
    if replications < 2:
        raise ValueError("at least two replications are required")
    d, cuts = features(panel)
    forecast, v47_meta = _fit_forecasts(d)
    item = _item_parameters(panel, d, forecast)

    scenario_results = []
    policy_metrics = []
    replicate_metrics = []
    weekly_metrics = []
    selected_rows = []
    detail_samples = []
    scenario_cache = []
    target_cache = {}

    for s_idx, scenario in enumerate(_scenario_grid()):
        probabilities = scenario["lead_probabilities"]
        dist_name = scenario["lead_distribution"]
        if dist_name not in target_cache:
            target_cache[dist_name] = _fit_stochastic_targets(d, forecast, probabilities)
        target, target_meta = target_cache[dist_name]
        scenario_weekly_rows = []
        # Re-attach the per-SKU training-only price/lot data through the fixed sorted ID order.
        dist_metrics = {}
        for cfg in CONFIGS:
            rep_summaries = []
            for rep in range(replications):
                rng = np.random.default_rng(SEED_V48 + s_idx * 100_000 + rep)
                draws = rng.choice(
                    np.array([1, 2, 3, 4], dtype=int),
                    size=(len(item["series_ids"]), target.week.nunique()),
                    p=np.array([probabilities[k] for k in (1, 2, 3, 4)], dtype=float),
                )
                summaries, weekly, _ = _simulate(target, item, probabilities, scenario, cfg, draws, False)
                rep_summaries.append(summaries)
                for row in weekly:
                    tagged = {"scenario_id": scenario["scenario_id"], "config_id": cfg["config_id"],
                              "replicate": rep, **row}
                    weekly_metrics.append(tagged)
                    scenario_weekly_rows.append(tagged)
                for split in ("selection", "test"):
                    replicate_metrics.append({
                        "scenario_id": scenario["scenario_id"], "config_id": cfg["config_id"],
                        "replicate": rep, "split": split, **summaries[split],
                    })
            dist_metrics[cfg["config_id"]] = rep_summaries
            for split in ("selection", "test"):
                values = [x[split]["mean_cost"] for x in rep_summaries]
                policy_metrics.append({
                    "scenario_id": scenario["scenario_id"], "config_id": cfg["config_id"],
                    "allocator": cfg["allocator"], "target_col": cfg["target_col"],
                    "family": cfg["family"], "split": split,
                    "mean_cost": float(np.mean(values)), "replicate_sd": float(np.std(values, ddof=1)),
                    "mean_fill_rate": float(np.mean([x[split]["fill_rate"] for x in rep_summaries])),
                    "mean_lost_units": float(np.mean([x[split]["lost_units"] for x in rep_summaries])),
                    "mean_order_value_proxy": float(np.mean([x[split]["mean_order_value_proxy"] for x in rep_summaries])),
                    "mean_budget_utilization": float(np.mean([x[split]["mean_budget_utilization"] for x in rep_summaries])),
                    "max_capacity_utilization": float(np.max([x[split]["max_capacity_utilization"] for x in rep_summaries])),
                })

        # Selection is frozen using selection-period mean cost across replications.
        selection_costs = {
            cfg["config_id"]: float(np.mean([x["selection"]["mean_cost"] for x in dist_metrics[cfg["config_id"]]]))
            for cfg in CONFIGS
        }
        baseline_id = min(("q50_proportional", "q75_proportional"), key=selection_costs.get)
        candidate_id = min(("q50_greedy", "q75_greedy", "q90_greedy"), key=selection_costs.get)
        selected_rows.extend([
            {"scenario_id": scenario["scenario_id"], "role": "baseline", "config_id": baseline_id, "selection_cost": selection_costs[baseline_id]},
            {"scenario_id": scenario["scenario_id"], "role": "candidate", "config_id": candidate_id, "selection_cost": selection_costs[candidate_id]},
        ])

        rep_deltas = []
        weekly_delta_by_week = {}
        for rep in range(replications):
            base = dist_metrics[baseline_id][rep]["test"]["mean_cost"]
            cand = dist_metrics[candidate_id][rep]["test"]["mean_cost"]
            rep_deltas.append(cand - base)
        weekly_frame = pd.DataFrame(scenario_weekly_rows)
        selected_weekly = weekly_frame[weekly_frame.config_id.isin((baseline_id, candidate_id))]
        base_week = selected_weekly[selected_weekly.config_id == baseline_id].groupby("week").mean_total_cost.mean()
        cand_week = selected_weekly[selected_weekly.config_id == candidate_id].groupby("week").mean_total_cost.mean()
        weeks = sorted(set(base_week.index) & set(cand_week.index))
        test_weeks = sorted(target.loc[target.split == "test", "week"].unique())
        test_weeks = [w for w in test_weeks if w in weeks]
        delta_weeks = np.array([cand_week.loc[w] - base_week.loc[w] for w in test_weeks], dtype=float)
        ci = _block_bootstrap_ci(delta_weeks, SEED_V48 + s_idx)
        baseline_test = float(np.mean([x["test"]["mean_cost"] for x in dist_metrics[baseline_id]]))
        candidate_test = float(np.mean([x["test"]["mean_cost"] for x in dist_metrics[candidate_id]]))
        fixed_base = float(np.mean([x["test"]["mean_cost"] for x in dist_metrics["q75_proportional"]]))
        fixed_candidate = float(np.mean([x["test"]["mean_cost"] for x in dist_metrics["q75_greedy"]]))
        scenario_results.append({
            "scenario_id": scenario["scenario_id"],
            "lead_distribution": scenario["lead_distribution"],
            "lead_probabilities": json.dumps({str(k): v for k, v in probabilities.items()}),
            "expected_lead_time": target_meta["expected_lead_time"],
            "budget_fraction_of_training_weekly_sales_value": scenario["budget_fraction"],
            "capacity_equivalents": scenario["capacity_equivalents"],
            "economics_scenario": scenario["economics_name"],
            "baseline_config_selected_on_selection": baseline_id,
            "candidate_config_selected_on_selection": candidate_id,
            "baseline_test_cost": baseline_test,
            "candidate_test_cost": candidate_test,
            "relative_cost_reduction": (baseline_test - candidate_test) / baseline_test if baseline_test > 0 else None,
            "paired_delta_candidate_minus_baseline": candidate_test - baseline_test,
            "block_bootstrap_95_ci_delta": json.dumps(ci),
            "replicate_delta_sd": float(np.std(rep_deltas, ddof=1)),
            "candidate_test_fill_rate": float(np.mean([x["test"]["fill_rate"] for x in dist_metrics[candidate_id]])),
            "baseline_test_fill_rate": float(np.mean([x["test"]["fill_rate"] for x in dist_metrics[baseline_id]])),
            "candidate_test_mean_budget_utilization": float(np.mean([x["test"]["mean_budget_utilization"] for x in dist_metrics[candidate_id]])),
            "candidate_test_max_capacity_utilization": float(np.max([x["test"]["max_capacity_utilization"] for x in dist_metrics[candidate_id]])),
            "fixed_q75_greedy_vs_proportional_cost_reduction": (fixed_base - fixed_candidate) / fixed_base if fixed_base > 0 else None,
            "fixed_q75_greedy_vs_proportional_delta_ci": json.dumps(_block_bootstrap_ci(
                (
                    weekly_frame[weekly_frame.config_id == "q75_greedy"].groupby("week").mean_total_cost.mean()
                    - weekly_frame[weekly_frame.config_id == "q75_proportional"].groupby("week").mean_total_cost.mean()
                ).reindex(test_weeks).to_numpy(float), SEED_V48 + s_idx + 1000)),
            "evidence_warning": "Recorded-sales replay; stochastic lead-time distribution, budgets, capacity, lot sizes and economic weights are scenario assumptions, not observed supplier/inventory facts.",
        })

        scenario_cache.append((scenario, target, target_meta, baseline_id, candidate_id))

    # Save detailed state-transition samples for one fixed replication of selected policies.
    for s_idx, (scenario, target, target_meta, baseline_id, candidate_id) in enumerate(scenario_cache):
        probabilities = scenario["lead_probabilities"]
        for role, config_id in (("baseline", baseline_id), ("candidate", candidate_id)):
            cfg = next(x for x in CONFIGS if x["config_id"] == config_id)
            rng = np.random.default_rng(SEED_V48 + s_idx * 100_000)
            draws = rng.choice(
                np.array([1, 2, 3, 4], dtype=int),
                size=(len(item["series_ids"]), target.week.nunique()),
                p=np.array([probabilities[k] for k in (1, 2, 3, 4)], dtype=float),
            )
            _, _, detail = _simulate(target, item, probabilities, scenario, cfg, draws, True)
            detail["scenario_id"] = scenario["scenario_id"]
            detail["role"] = role
            detail["replicate"] = 0
            detail_samples.append(detail)

    # MILP comparison on a bounded sample of selected candidate weekly allocation states.
    decisions = pd.concat(detail_samples, ignore_index=True)
    milp_rows = []
    for scenario_id, group in decisions[decisions.role == "candidate"].groupby("scenario_id", sort=True):
        weeks = sorted(group.week.unique())
        sampled_weeks = weeks[::2][:8]
        for week in sampled_weeks:
            state = group[group.week == week].sort_values("series_id")
            if not len(state):
                continue
            budget_limit = float(state.budget_limit_week.iloc[0])
            capacity_room = float(state.capacity_room_before_order.iloc[0])
            economics = {
                "shortage_rate": float(state.shortage_rate.iloc[0]),
                "holding_rate": float(state.holding_rate.iloc[0]),
                "fixed_order_cost": float(state.fixed_order_cost_rate.iloc[0]),
            }
            result = solve_allocation_milp(
                state.shortfall.to_numpy(float),
                state.lot_size.to_numpy(int),
                state.unit_value_proxy.to_numpy(float),
                state.mean_demand.to_numpy(float),
                budget_limit,
                capacity_room,
                economics,
            )
            greedy_qty = state.order_qty.to_numpy(int)
            greedy_obj = _allocation_objective(
                greedy_qty, state.shortfall.to_numpy(float), state.lot_size.to_numpy(int),
                state.unit_value_proxy.to_numpy(float), economics,
            )
            milp_rows.append({
                "scenario_id": scenario_id, "week": int(week), "solver_success": bool(result["success"]),
                "greedy_objective": greedy_obj, "milp_objective": result["objective"],
                "greedy_vs_milp_gap": (
                    (result["objective"] - greedy_obj) / max(abs(result["objective"]), 1e-9)
                    if result["success"] and result["objective"] > 0 else None
                ),
                "greedy_order_units": int(greedy_qty.sum()),
                "milp_order_units": int(result["quantity"].sum()),
                "greedy_budget_value": float(np.sum(greedy_qty * state.unit_value_proxy.to_numpy(float))),
                "milp_budget_value": float(np.sum(result["quantity"] * state.unit_value_proxy.to_numpy(float))),
                "milp_capacity_usage": float(np.sum(result["quantity"] / state.mean_demand.to_numpy(float))),
                "budget_limit": budget_limit, "capacity_room_before_order": capacity_room,
            })

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(policy_metrics).to_csv(out / "policy_grid_metrics.csv", index=False)
    pd.DataFrame(replicate_metrics).to_csv(out / "replicate_metrics.csv", index=False)
    pd.DataFrame(weekly_metrics).to_csv(out / "weekly_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "selected_policies.csv", index=False)
    pd.DataFrame(scenario_results).to_csv(out / "scenario_summary.csv", index=False)
    decisions.to_csv(out / "selected_decisions_audit_sample.csv", index=False)
    pd.DataFrame(milp_rows).to_csv(out / "milp_snapshot_metrics.csv", index=False)

    manifest = {
        "version": "0.48",
        "mode": "stochastic_lead_time_budget_capacity_constrained_inventory_replay",
        "dataset": "UCI Online Retail II previously prepared product-country weekly recorded-sales panel",
        "source_transaction_rows": int(quality.get("raw_rows", 0)),
        "valid_source_transaction_rows": int(quality.get("valid_rows", 0)),
        "panel_rows": int(len(panel)),
        "series_count": int(len(item["series_ids"])),
        "chronological_splits": {
            split: {"rows": int((d.split == split).sum()), "weeks": int(d.loc[d.split == split, "week"].nunique())}
            for split in ("train", "calibration", "selection", "test")
        },
        "calendar_cut_indices": cuts,
        "v47_forecast_pipeline_metadata": v47_meta,
        "lead_distributions": LEAD_DISTRIBUTIONS,
        "budget_fractions": list(BUDGET_FRACTIONS),
        "capacity_equivalents": list(CAPACITY_EQUIVALENTS),
        "economic_scenarios": ECONOMIC_SCENARIOS,
        "replications": replications,
        "random_seed": SEED_V48,
        "common_random_numbers": "Within each scenario and replication, every policy receives the same per-series/per-week lead-time draw matrix.",
        "order_cutoff": f"No new orders during final {MAX_LEAD} test weeks; this allows all orders to arrive within observed evaluation horizon.",
        "budget_definition": "fraction of total training-period mean weekly recorded sales value, using training-only median selling price as a value proxy; not procurement spend or actual cash budget.",
        "capacity_definition": "sum of on-hand plus pipeline divided by each SKU's training mean weekly recorded sales; normalized SKU-week demand equivalents, not cubic/physical warehouse capacity.",
        "lot_size_definition": item["lot_rule"],
        "unit_value_proxy": item["price_source"],
        "stochastic_target": "Weighted mixture quantile over 1..4 week lead-time cumulative residual windows calibrated on calibration split only; current one-step mean/scale held constant across horizon.",
        "policy_selection": "Select proportional-family and marginal-priority-family configurations using selection split mean cost averaged over replications; freeze choices before test.",
        "policy_selection_families": "proportional family q50/q75; priority family q50/q75/q90, selected on selection split only.",
        "primary_ablation": "q75_greedy vs q75_proportional holds stochastic target quantile fixed and isolates cross-SKU allocation rule.",
        "milp_benchmark": "Exact one-period integer-lot allocation on sampled selected-policy states; not a globally optimal multi-period policy.",
        "price_and_cost_warning": "Selling price is not acquisition cost or gross margin. Economic coefficients are assumptions and all costs are proxy units, not realized profit.",
        "demand_warning": "No inventory availability/stockout records; recorded sales can be censored by stockouts. Missing calendar weeks are treated as zero recorded sales in the inherited V47 feature panel.",
        "scenario_results": scenario_results,
        "source_quality": quality,
        "source_hashes": {
            "prepared_panel_sha256": "1d32ccc76a7bb6df0127d0a2d32368a22b79558e33603dfb82afd65eb0293831",
            "quality_manifest_sha256": "5a564a92ac78c8de95511862ea576aaf9aa5784572b8c732d38be78606c19d6f",
        },
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/v48/inventory"))
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    from inventory_v47 import load_prepared_panel
    # V47's verified loader intentionally returns only series/date/demand. V48
    # also needs the already hash-verified training-period price column for its
    # value-budget proxy, so re-read the same pinned CSV after verification.
    _, quality = load_prepared_panel(args.panel, args.quality)
    panel = pd.read_csv(args.panel)
    required = {"series_id", "date", "demand", "price"}
    if not required.issubset(panel.columns):
        parser.error(f"prepared panel missing V0.48 columns: {sorted(required - set(panel.columns))}")
    manifest = run(panel, quality, args.output, args.replications)
    (args.output / "quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False))
    print(json.dumps({
        "version": manifest["version"], "scenario_count": len(manifest["scenario_results"]),
        "replications": manifest["replications"], "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
