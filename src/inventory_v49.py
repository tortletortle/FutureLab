"""V0.49 capital-aware marginal-lot inventory research.

V0.49 preserves V0.48 as an immutable baseline and adds:
- sequential per-lot marginal valuation (rather than a SKU-level average score);
- an explicit working-capital exposure charge on on-hand plus pipeline inventory;
- selection-period service-floor constrained policy selection;
- an exact one-period MILP with lot-prefix constraints for the new marginal objective.

All demand remains recorded sales, and all capital/holding/lead-time parameters are
scenario assumptions because the source panel contains no inventory or acquisition-cost data.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import heapq

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from inventory_v47 import _fit_forecasts, load_prepared_panel
from retail_v46 import features
from inventory_v48 import (
    LEAD_DISTRIBUTIONS, BUDGET_FRACTIONS, CAPACITY_EQUIVALENTS,
    ECONOMIC_SCENARIOS, MAX_LEAD, REPLICATIONS, BOOTSTRAP_REPS,
    SEED_V48, _fit_stochastic_targets, _item_parameters,
    _allocate_proportional, _allocate_marginal_greedy, _block_bootstrap_ci,
)

CAPITAL_RATES = (0.0, 0.005)  # per-week charge on selling-price-value proxy; stress assumptions only
SERVICE_TOLERANCE = 0.005    # candidate may trail baseline selection fill by at most 0.5 pp
SEED_V49 = 49049

CONFIGS = (
    {"config_id": "q50_proportional", "target_col": "stoch_q50", "allocator": "proportional", "family": "baseline"},
    {"config_id": "q75_proportional", "target_col": "stoch_q75", "allocator": "proportional", "family": "baseline"},
    {"config_id": "q50_avg_greedy", "target_col": "stoch_q50", "allocator": "average_greedy", "family": "legacy_priority"},
    {"config_id": "q75_avg_greedy", "target_col": "stoch_q75", "allocator": "average_greedy", "family": "legacy_priority"},
    {"config_id": "q90_avg_greedy", "target_col": "stoch_q90", "allocator": "average_greedy", "family": "legacy_priority"},
    {"config_id": "q50_marginal_lot", "target_col": "stoch_q50", "allocator": "marginal_lot", "family": "capital_marginal"},
    {"config_id": "q75_marginal_lot", "target_col": "stoch_q75", "allocator": "marginal_lot", "family": "capital_marginal"},
    {"config_id": "q90_marginal_lot", "target_col": "stoch_q90", "allocator": "marginal_lot", "family": "capital_marginal"},
)


def _scenario_grid_v49() -> list[dict]:
    scenarios = []
    for dist_name, probabilities in LEAD_DISTRIBUTIONS.items():
        for budget_fraction in BUDGET_FRACTIONS:
            for capacity in CAPACITY_EQUIVALENTS:
                for economics_name, economics in ECONOMIC_SCENARIOS.items():
                    for capital_rate in CAPITAL_RATES:
                        scenarios.append({
                            "scenario_id": (
                                f"{dist_name}_budget{int(budget_fraction * 100)}_cap{capacity:g}_"
                                f"{economics_name}_capital{capital_rate:g}"
                            ),
                            "lead_distribution": dist_name,
                            "lead_probabilities": probabilities,
                            "budget_fraction": budget_fraction,
                            "capacity_equivalents": capacity,
                            "economics_name": economics_name,
                            "economics": economics,
                            "capital_rate": capital_rate,
                        })
    return scenarios


def _lot_benefits(shortfall: np.ndarray, lot: np.ndarray, price: np.ndarray,
                  demand_scale: np.ndarray, economics: dict,
                  expected_lead: float, capital_rate: float) -> list[np.ndarray]:
    """Return ordered incremental value for each feasible lot of each series.

    The first lot bears the one-time setup proxy. Each lot is charged for expected
    excess-inventory holding and capital exposure. The expected dwell-time formula
    is deliberately simple and is an explicit scenario model, not observed data.
    """
    shortfall = np.asarray(shortfall, dtype=float)
    lot = np.asarray(lot, dtype=int)
    price = np.asarray(price, dtype=float)
    demand_scale = np.asarray(demand_scale, dtype=float)
    if not (shortfall.shape == lot.shape == price.shape == demand_scale.shape):
        raise ValueError("lot-benefit inputs must be aligned")
    if (shortfall < 0).any() or (lot <= 0).any() or (price <= 0).any() or (demand_scale <= 0).any():
        raise ValueError("invalid lot-benefit inputs")
    if not np.isfinite(shortfall).all() or not np.isfinite(price).all() or not np.isfinite(demand_scale).all():
        raise ValueError("lot-benefit inputs must be finite")
    if expected_lead < 1 or capital_rate < 0 or not np.isfinite(capital_rate):
        raise ValueError("invalid lead-time or capital-rate assumption")

    result: list[np.ndarray] = []
    for i in range(len(shortfall)):
        n_lots = int(math.ceil(shortfall[i] / lot[i] - 1e-12)) if shortfall[i] > 0 else 0
        values = np.zeros(n_lots, dtype=float)
        for j in range(n_lots):
            remaining = max(float(shortfall[i]) - j * int(lot[i]), 0.0)
            filled = min(float(lot[i]), remaining)
            excess = max(float(lot[i]) - filled, 0.0)
            # A half-week within-period consumption approximation plus the
            # additional time required to consume excess inventory.
            dwell_weeks = 0.5 + excess / (2.0 * float(demand_scale[i]))
            holding_charge = (
                float(economics["holding_rate"]) * float(price[i]) * excess * dwell_weeks
            )
            capital_exposure_weeks = float(expected_lead) + dwell_weeks
            capital_charge = (
                float(capital_rate) * float(price[i]) * int(lot[i]) * capital_exposure_weeks
            )
            setup_charge = float(economics["fixed_order_cost"]) if j == 0 else 0.0
            service_benefit = float(economics["shortage_rate"]) * float(price[i]) * filled
            values[j] = service_benefit - holding_charge - capital_charge - setup_charge
        result.append(values)
    return result


def _allocate_marginal_lots(request: np.ndarray, shortfall: np.ndarray,
                            lot: np.ndarray, price: np.ndarray,
                            demand_scale: np.ndarray, budget: float,
                            capacity_room: float, economics: dict,
                            expected_lead: float, capital_rate: float) -> np.ndarray:
    """Allocate one feasible lot at a time using its current marginal value density."""
    request = np.asarray(request, dtype=int)
    lot = np.asarray(lot, dtype=int)
    price = np.asarray(price, dtype=float)
    demand_scale = np.asarray(demand_scale, dtype=float)
    if not (request.shape == lot.shape == price.shape == demand_scale.shape == np.asarray(shortfall).shape):
        raise ValueError("allocator inputs must be aligned")
    if (request < 0).any() or (lot <= 0).any() or (request % lot != 0).any():
        raise ValueError("requests must be nonnegative multiples of positive lots")
    if budget < 0 or capacity_room < 0:
        raise ValueError("resource limits must be nonnegative")
    if budget == 0 or capacity_room == 0 or request.sum() == 0:
        return np.zeros_like(request)

    benefits = _lot_benefits(shortfall, lot, price, demand_scale, economics, expected_lead, capital_rate)
    max_lots = request // lot
    next_lot = np.zeros(len(request), dtype=int)
    order = np.zeros_like(request)
    remaining_budget = float(budget)
    remaining_capacity = float(capacity_room)

    # A heap is equivalent to repeated global re-ranking because each lot's
    # score is fixed against the original resource envelope. Only the selected
    # SKU's next lot enters the frontier after an allocation.
    heap = []
    for i in range(len(request)):
        if int(max_lots[i]) <= 0 or len(benefits[i]) == 0:
            continue
        qty = int(lot[i])
        spend = qty * float(price[i])
        capacity = qty / float(demand_scale[i])
        burden = spend / max(float(budget), 1e-9) + capacity / max(float(capacity_room), 1e-9)
        benefit = float(benefits[i][0])
        if benefit > 0:
            heapq.heappush(heap, (-benefit / max(burden, 1e-12), i, 0))

    while heap:
        neg_score, i, j = heapq.heappop(heap)
        qty = int(lot[i])
        spend = qty * float(price[i])
        capacity = qty / float(demand_scale[i])
        # Resources only decrease, so a currently infeasible lot can never
        # become feasible later and is safely discarded.
        if spend > remaining_budget + 1e-8 or capacity > remaining_capacity + 1e-10:
            continue
        # Consecutive full lots have identical marginal value. Since this
        # SKU's next lot wins exact score ties by stable index, allocate that
        # run in one operation instead of iterating once per physical lot.
        max_j = min(int(max_lots[i]), len(benefits[i]))
        base_benefit = float(benefits[i][j])
        tail = benefits[i][j:max_j]
        different = np.flatnonzero(~np.isclose(tail, base_benefit, rtol=1e-12, atol=1e-12))
        same_run = int(different[0]) if len(different) else len(tail)
        budget_lots = int(math.floor((remaining_budget + 1e-8) / spend))
        capacity_lots = int(math.floor((remaining_capacity + 1e-10) / capacity))
        take = min(same_run, budget_lots, capacity_lots)
        if take <= 0:
            continue
        order[i] += take * qty
        next_lot[i] = j + take
        remaining_budget -= take * spend
        remaining_capacity -= take * capacity
        next_j = j + take
        if next_j < max_j:
            benefit = float(benefits[i][next_j])
            if benefit > 0:
                burden = spend / max(float(budget), 1e-9) + capacity / max(float(capacity_room), 1e-9)
                heapq.heappush(heap, (-benefit / max(burden, 1e-12), i, next_j))

    if (order > request).any() or (order % lot != 0).any():
        raise AssertionError("marginal allocator violated request or lot bounds")
    if float(np.sum(order * price)) > budget + 1e-6:
        raise AssertionError("marginal allocator exceeded budget")
    if float(np.sum(order / demand_scale)) > capacity_room + 1e-8:
        raise AssertionError("marginal allocator exceeded capacity")
    return order


def solve_marginal_lot_milp(shortfall: np.ndarray, request: np.ndarray,
                            lot: np.ndarray, price: np.ndarray,
                            demand_scale: np.ndarray, budget: float,
                            capacity_room: float, economics: dict,
                            expected_lead: float, capital_rate: float,
                            time_limit: float = 2.0) -> dict:
    """Exact binary-lot solution for the same per-lot objective and prefix rules."""
    request = np.asarray(request, dtype=int)
    lot = np.asarray(lot, dtype=int)
    price = np.asarray(price, dtype=float)
    demand_scale = np.asarray(demand_scale, dtype=float)
    benefits = _lot_benefits(shortfall, lot, price, demand_scale, economics, expected_lead, capital_rate)
    variables = [(i, j, float(value)) for i, vals in enumerate(benefits)
                 for j, value in enumerate(vals) if j < int(request[i] // lot[i])]
    n = len(variables)
    if n == 0 or budget <= 0 or capacity_room <= 0:
        return {"success": True, "quantity": np.zeros_like(request), "objective": 0.0, "status": 0}

    # Two resource rows, followed by x[i,j+1] - x[i,j] <= 0 prefix rows.
    prefix = []
    variable_index = {(i, j): k for k, (i, j, _) in enumerate(variables)}
    for i in range(len(request)):
        for j in range(int(request[i] // lot[i]) - 1):
            if (i, j) in variable_index and (i, j + 1) in variable_index:
                prefix.append((variable_index[(i, j)], variable_index[(i, j + 1)]))
    rows = 2 + len(prefix)
    rr, cc, vv = [], [], []
    for k, (i, j, _) in enumerate(variables):
        rr.extend((0, 1))
        cc.extend((k, k))
        vv.extend((int(lot[i]) * float(price[i]), int(lot[i]) / float(demand_scale[i])))
    for r, (prev_idx, next_idx) in enumerate(prefix, start=2):
        rr.extend((r, r))
        cc.extend((next_idx, prev_idx))
        vv.extend((1.0, -1.0))
    matrix = coo_matrix((vv, (rr, cc)), shape=(rows, n)).tocsr()
    lower = np.full(rows, -np.inf, dtype=float)
    upper = np.array([budget, capacity_room] + [0.0] * len(prefix), dtype=float)
    c = -np.array([x[2] for x in variables], dtype=float)
    result = milp(
        c=c, integrality=np.ones(n, dtype=int), bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": 1e-8},
    )
    if not result.success or result.x is None:
        return {"success": False, "quantity": np.zeros_like(request), "objective": None, "status": int(result.status)}
    selected = np.rint(result.x).astype(int)
    quantity = np.zeros_like(request)
    objective = 0.0
    for flag, (i, j, value) in zip(selected, variables):
        if flag:
            quantity[i] += int(lot[i])
            objective += value
    feasible = (
        (quantity <= request).all()
        and (quantity % lot == 0).all()
        and float(np.sum(quantity * price)) <= budget + 1e-6
        and float(np.sum(quantity / demand_scale)) <= capacity_room + 1e-8
    )
    if not feasible:
        return {"success": False, "quantity": np.zeros_like(request), "objective": None, "status": int(result.status)}
    return {"success": True, "quantity": quantity, "objective": float(objective), "status": int(result.status)}


def _simulate_v49(forecast: pd.DataFrame, item: dict, probabilities: dict[int, float],
                  scenario: dict, config: dict, lead_draws: np.ndarray,
                  keep_detail: bool = False):
    probabilities = {int(k): float(v) for k, v in probabilities.items()}
    if set(probabilities) != {1, 2, 3, 4} or any(v < 0 for v in probabilities.values()) or not np.isclose(sum(probabilities.values()), 1):
        raise ValueError("invalid lead-time distribution")
    ids = item["series_ids"]
    weeks = np.array(sorted(forecast.week.unique()), dtype=int)
    n_series, n_periods = len(ids), len(weeks)
    index = pd.MultiIndex.from_product([ids, weeks], names=["series_id", "week"])
    aligned = forecast.set_index(["series_id", "week"]).reindex(index).reset_index()
    required = [config["target_col"], "demand", "date", "split"]
    if aligned[required].isna().any().any():
        raise ValueError("forecast panel is incomplete")
    if lead_draws.shape != (n_series, n_periods) or not np.isin(lead_draws, [1, 2, 3, 4]).all():
        raise ValueError("invalid lead-time draw matrix")

    targets = np.maximum(aligned[config["target_col"]].to_numpy(float).reshape(n_series, n_periods), 0)
    demands = np.ceil(np.maximum(aligned.demand.to_numpy(float).reshape(n_series, n_periods), 0)).astype(int)
    dates = aligned.date.astype(str).to_numpy().reshape(n_series, n_periods)
    splits = aligned.split.astype(str).to_numpy().reshape(n_series, n_periods)
    if not (splits == splits[0:1, :]).all():
        raise ValueError("split boundaries must align across series")
    if not np.isfinite(targets).all():
        raise ValueError("targets must be finite")

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
    if not len(test_weeks):
        raise ValueError("test split is required")
    last_order_week = int(test_weeks.max() - MAX_LEAD)
    budget = float(scenario["budget_fraction"] * item["weekly_sales_value_proxy"])
    capacity_limit = float(scenario["capacity_equivalents"])
    costs = scenario["economics"]
    capital_rate = float(scenario["capital_rate"])
    if capital_rate < 0 or not np.isfinite(capital_rate):
        raise ValueError("capital rate must be finite and nonnegative")

    weekly_rows, detail_rows = [], []
    for t, week in enumerate(weeks):
        arrival = due[:, t].copy()
        on_hand += arrival
        pipeline_total -= arrival
        if (pipeline_total < 0).any():
            raise AssertionError("pipeline became negative")
        on_hand_start = on_hand.copy()
        pipeline_before = pipeline_total.copy()
        inventory_position = on_hand_start + pipeline_before
        shortfall = np.maximum(targets[:, t] - inventory_position, 0.0)
        raw_request = np.ceil(shortfall).astype(int)
        requested = np.where(raw_request > 0, np.ceil(raw_request / lot).astype(int) * lot, 0)
        order_eligible = (splits[0, t] != "test") or int(week) <= last_order_week
        request_for_allocation = requested.copy() if order_eligible else np.zeros_like(requested)
        capacity_room = max(0.0, capacity_limit - float(np.sum(inventory_position / demand_scale)))

        if config["allocator"] == "proportional":
            order = _allocate_proportional(request_for_allocation, lot, price, demand_scale, budget, capacity_room)
        elif config["allocator"] == "average_greedy":
            order = _allocate_marginal_greedy(
                request_for_allocation, shortfall, lot, price, demand_scale,
                budget, capacity_room, costs,
            )
        elif config["allocator"] == "marginal_lot":
            order = _allocate_marginal_lots(
                request_for_allocation, shortfall, lot, price, demand_scale,
                budget, capacity_room, costs, expected_lead, capital_rate,
            )
        else:
            raise ValueError("unknown V0.49 allocator")

        order_lead = np.where(order > 0, lead_draws[:, t], 0).astype(int)
        for i in np.flatnonzero(order > 0):
            due_week = t + int(order_lead[i])
            if due_week >= n_periods:
                raise AssertionError("order arrives outside observed horizon")
            due[i, due_week] += int(order[i])
        pipeline_total += order
        pipeline_after = pipeline_total.copy()
        fulfilled = np.minimum(on_hand_start, demands[:, t])
        lost = demands[:, t] - fulfilled
        on_hand = on_hand_start - fulfilled
        holding_cost = costs["holding_rate"] * on_hand * price
        lost_sales_cost = costs["shortage_rate"] * lost * price
        fixed_order_cost = costs["fixed_order_cost"] * (order > 0)
        capital_exposure_units = 0.5 * (
            on_hand_start + pipeline_before + on_hand + pipeline_after
        )
        capital_exposure_value = capital_exposure_units * price
        capital_charge = capital_rate * capital_exposure_value
        total_cost = holding_cost + lost_sales_cost + fixed_order_cost + capital_charge

        budget_spend = float(np.sum(order * price))
        capacity_after = float(np.sum((on_hand_start + pipeline_before + order) / demand_scale))
        if budget_spend > budget + 1e-6:
            raise AssertionError("weekly budget exceeded")
        if capacity_after > capacity_limit + 1e-8:
            raise AssertionError("weekly capacity exceeded")
        if (order % lot != 0).any() or (order > requested).any():
            raise AssertionError("MOQ/request bound violated")
        if np.any(fulfilled + lost != demands[:, t]):
            raise AssertionError("demand balance violated")

        split = str(splits[0, t])
        weekly_rows.append({
            "week": int(week), "split": split,
            "mean_total_cost": float(total_cost.mean()),
            "total_cost_proxy": float(total_cost.sum()),
            "fill_rate": float(fulfilled.sum() / demands[:, t].sum()) if demands[:, t].sum() else 1.0,
            "lost_units": int(lost.sum()),
            "holding_cost_total": float(holding_cost.sum()),
            "lost_sales_cost_total": float(lost_sales_cost.sum()),
            "fixed_order_cost_total": float(fixed_order_cost.sum()),
            "capital_charge_total": float(capital_charge.sum()),
            "capital_exposure_value_total": float(capital_exposure_value.sum()),
            "order_value_proxy": budget_spend, "budget_limit_proxy": budget,
            "budget_utilization": budget_spend / budget if budget else 0.0,
            "capacity_usage_after": capacity_after, "capacity_limit": capacity_limit,
        })
        if keep_detail:
            for i, sid in enumerate(ids):
                detail_rows.append({
                    "series_id": sid, "week": int(week), "date": dates[i, t], "split": split,
                    "config_id": config["config_id"], "allocator": config["allocator"],
                    "sampled_lead_time": int(lead_draws[i, t]), "order_lead_time": int(order_lead[i]),
                    "arrival_qty": int(arrival[i]), "initial_stock": int(initial_stock[i]),
                    "on_hand_start": int(on_hand_start[i]), "pipeline_before": int(pipeline_before[i]),
                    "inventory_position": int(inventory_position[i]), "target_level": float(targets[i, t]),
                    "shortfall": float(shortfall[i]), "requested_qty": int(requested[i]),
                    "order_eligible": bool(order_eligible), "lot_size": int(lot[i]),
                    "order_qty": int(order[i]), "arrival_week": int(week + order_lead[i]) if order[i] > 0 else -1,
                    "pipeline_after": int(pipeline_after[i]), "demand": int(demands[i, t]),
                    "fulfilled": int(fulfilled[i]), "lost_sales": int(lost[i]),
                    "on_hand_end": int(on_hand[i]), "unit_value_proxy": float(price[i]),
                    "mean_demand": float(demand_scale[i]), "holding_cost": float(holding_cost[i]),
                    "lost_sales_cost": float(lost_sales_cost[i]),
                    "fixed_order_cost": float(fixed_order_cost[i]),
                    "capital_exposure_units": float(capital_exposure_units[i]),
                    "capital_exposure_value": float(capital_exposure_value[i]),
                    "capital_rate": capital_rate, "capital_charge": float(capital_charge[i]),
                    "total_cost": float(total_cost[i]),
                    "budget_spend_week": budget_spend, "budget_limit_week": budget,
                    "capacity_usage_after_week": capacity_after, "capacity_limit": capacity_limit,
                    "capacity_room_before_order": capacity_room,
                    "shortage_rate": float(costs["shortage_rate"]),
                    "holding_rate": float(costs["holding_rate"]),
                    "fixed_order_cost_rate": float(costs["fixed_order_cost"]),
                    "expected_lead_time": expected_lead,
                })

    weekly = pd.DataFrame(weekly_rows)
    summaries = {}
    for split in ("selection", "test"):
        w = weekly[weekly.split == split]
        mask = splits[0] == split
        summaries[split] = {
            "mean_cost": float(w.mean_total_cost.mean()),
            "fill_rate": float(1 - w.lost_units.sum() / max(int(np.sum(demands[:, mask])), 1)),
            "lost_units": int(w.lost_units.sum()),
            "mean_order_value_proxy": float(w.order_value_proxy.mean()),
            "mean_budget_utilization": float(w.budget_utilization.mean()),
            "max_capacity_utilization": float((w.capacity_usage_after / w.capacity_limit).max()),
            "mean_capital_charge_per_week": float(w.capital_charge_total.mean()),
            "mean_capital_exposure_value": float(w.capital_exposure_value_total.mean()),
        }
    return summaries, weekly_rows, pd.DataFrame(detail_rows) if keep_detail else None


def _select_policies(selection_metrics: pd.DataFrame) -> dict:
    """Select the proportional baseline, then the cheapest candidate under a fill floor."""
    baseline_pool = selection_metrics[selection_metrics.config_id.isin(
        ["q50_proportional", "q75_proportional"]
    )]
    if baseline_pool.empty:
        raise ValueError("baseline selection metrics are missing")
    baseline = baseline_pool.sort_values(["mean_cost", "config_id"]).iloc[0]
    service_floor = max(0.0, float(baseline.mean_fill_rate) - SERVICE_TOLERANCE)
    candidate_pool = selection_metrics[~selection_metrics.config_id.isin(
        ["q50_proportional", "q75_proportional"]
    )]
    feasible = candidate_pool[candidate_pool.mean_fill_rate >= service_floor]
    if feasible.empty:
        return {
            "baseline_id": str(baseline.config_id), "candidate_id": str(baseline.config_id),
            "service_floor": service_floor, "service_floor_feasible": False,
            "candidate_selection_cost": float(baseline.mean_cost),
            "candidate_selection_fill_rate": float(baseline.mean_fill_rate),
        }
    candidate = feasible.sort_values(["mean_cost", "config_id"]).iloc[0]
    return {
        "baseline_id": str(baseline.config_id), "candidate_id": str(candidate.config_id),
        "service_floor": service_floor, "service_floor_feasible": True,
        "candidate_selection_cost": float(candidate.mean_cost),
        "candidate_selection_fill_rate": float(candidate.mean_fill_rate),
    }


def run(panel: pd.DataFrame, quality: dict, out: Path, replications: int = REPLICATIONS) -> dict:
    if replications < 2:
        raise ValueError("at least two replications are required")
    d, cuts = features(panel)
    forecast, forecast_meta = _fit_forecasts(d)
    item = _item_parameters(panel, d, forecast)
    scenario_results, policy_metrics, replicate_metrics, weekly_metrics = [], [], [], []
    selected_rows, detail_samples, scenario_cache = [], [], []
    target_cache = {}

    for s_idx, scenario in enumerate(_scenario_grid_v49()):
        probabilities = scenario["lead_probabilities"]
        dist_name = scenario["lead_distribution"]
        if dist_name not in target_cache:
            target_cache[dist_name] = _fit_stochastic_targets(d, forecast, probabilities)
        target, target_meta = target_cache[dist_name]
        dist_metrics = {}
        scenario_weekly_rows = []
        for cfg in CONFIGS:
            rep_summaries = []
            for rep in range(replications):
                rng = np.random.default_rng(SEED_V49 + s_idx * 100_000 + rep)
                draws = rng.choice(
                    np.array([1, 2, 3, 4], dtype=int),
                    size=(len(item["series_ids"]), target.week.nunique()),
                    p=np.array([probabilities[k] for k in (1, 2, 3, 4)], dtype=float),
                )
                summaries, weekly, _ = _simulate_v49(target, item, probabilities, scenario, cfg, draws, False)
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
                policy_metrics.append({
                    "scenario_id": scenario["scenario_id"], "config_id": cfg["config_id"],
                    "allocator": cfg["allocator"], "family": cfg["family"], "split": split,
                    "mean_cost": float(np.mean([x[split]["mean_cost"] for x in rep_summaries])),
                    "replicate_sd": float(np.std([x[split]["mean_cost"] for x in rep_summaries], ddof=1)),
                    "mean_fill_rate": float(np.mean([x[split]["fill_rate"] for x in rep_summaries])),
                    "mean_lost_units": float(np.mean([x[split]["lost_units"] for x in rep_summaries])),
                    "mean_order_value_proxy": float(np.mean([x[split]["mean_order_value_proxy"] for x in rep_summaries])),
                    "mean_budget_utilization": float(np.mean([x[split]["mean_budget_utilization"] for x in rep_summaries])),
                    "max_capacity_utilization": float(np.max([x[split]["max_capacity_utilization"] for x in rep_summaries])),
                    "mean_capital_charge_per_week": float(np.mean([x[split]["mean_capital_charge_per_week"] for x in rep_summaries])),
                    "mean_capital_exposure_value": float(np.mean([x[split]["mean_capital_exposure_value"] for x in rep_summaries])),
                })

        selection_frame = pd.DataFrame([x for x in policy_metrics if x["scenario_id"] == scenario["scenario_id"] and x["split"] == "selection"])
        selected = _select_policies(selection_frame)
        baseline_id, candidate_id = selected["baseline_id"], selected["candidate_id"]
        for role, config_id in (("baseline", baseline_id), ("candidate", candidate_id)):
            row = selection_frame[selection_frame.config_id == config_id].iloc[0]
            selected_rows.append({
                "scenario_id": scenario["scenario_id"], "role": role, "config_id": config_id,
                "selection_cost": float(row.mean_cost), "selection_fill_rate": float(row.mean_fill_rate),
                "service_floor": selected["service_floor"],
                "service_floor_feasible": selected["service_floor_feasible"],
            })

        baseline_test = float(np.mean([x["test"]["mean_cost"] for x in dist_metrics[baseline_id]]))
        candidate_test = float(np.mean([x["test"]["mean_cost"] for x in dist_metrics[candidate_id]]))
        baseline_fill = float(np.mean([x["test"]["fill_rate"] for x in dist_metrics[baseline_id]]))
        candidate_fill = float(np.mean([x["test"]["fill_rate"] for x in dist_metrics[candidate_id]]))
        deltas = [dist_metrics[candidate_id][r]["test"]["mean_cost"] - dist_metrics[baseline_id][r]["test"]["mean_cost"] for r in range(replications)]
        wf = pd.DataFrame(scenario_weekly_rows)
        base_week = wf[wf.config_id == baseline_id].groupby("week").mean_total_cost.mean()
        cand_week = wf[wf.config_id == candidate_id].groupby("week").mean_total_cost.mean()
        test_weeks = sorted(target.loc[target.split == "test", "week"].unique())
        delta = np.array([cand_week.loc[w] - base_week.loc[w] for w in test_weeks], dtype=float)
        ci = _block_bootstrap_ci(delta, SEED_V49 + s_idx, reps=BOOTSTRAP_REPS, block=4)
        candidate_test_summaries = [x["test"] for x in dist_metrics[candidate_id]]
        scenario_results.append({
            "scenario_id": scenario["scenario_id"],
            "lead_distribution": dist_name,
            "lead_probabilities": json.dumps({str(k): v for k, v in probabilities.items()}),
            "expected_lead_time": target_meta["expected_lead_time"],
            "budget_fraction": scenario["budget_fraction"],
            "capacity_equivalents": scenario["capacity_equivalents"],
            "economics_scenario": scenario["economics_name"],
            "capital_rate_per_week": scenario["capital_rate"],
            "baseline_config_selected_on_selection": baseline_id,
            "candidate_config_selected_on_selection": candidate_id,
            "candidate_allocator": next(x["allocator"] for x in CONFIGS if x["config_id"] == candidate_id),
            "selection_service_floor": selected["service_floor"],
            "selection_service_floor_feasible": selected["service_floor_feasible"],
            "baseline_test_cost": baseline_test,
            "candidate_test_cost": candidate_test,
            "relative_cost_reduction": (baseline_test - candidate_test) / baseline_test if baseline_test > 0 else None,
            "paired_delta_candidate_minus_baseline": candidate_test - baseline_test,
            "block_bootstrap_95_ci_delta": json.dumps(ci),
            "replicate_delta_sd": float(np.std(deltas, ddof=1)),
            "baseline_test_fill_rate": baseline_fill,
            "candidate_test_fill_rate": candidate_fill,
            "candidate_test_service_delta_pp": 100.0 * (candidate_fill - baseline_fill),
            "baseline_test_selection_floor_met": bool(baseline_fill >= selected["service_floor"]),
            "candidate_test_selection_floor_met": bool(candidate_fill >= selected["service_floor"]),
            "test_service_floor_met": bool(candidate_fill >= selected["service_floor"]),
            "test_service_noninferiority_vs_baseline": bool(candidate_fill >= baseline_fill - SERVICE_TOLERANCE),
            "candidate_test_mean_capital_charge_per_week": float(np.mean([x["mean_capital_charge_per_week"] for x in candidate_test_summaries])),
            "candidate_test_mean_capital_exposure_value": float(np.mean([x["mean_capital_exposure_value"] for x in candidate_test_summaries])),
            "candidate_test_mean_budget_utilization": float(np.mean([x["mean_budget_utilization"] for x in candidate_test_summaries])),
            "candidate_test_max_capacity_utilization": float(np.max([x["max_capacity_utilization"] for x in candidate_test_summaries])),
            "evidence_warning": "Recorded-sales replay; capital rate, lead-time distribution, budget, capacity, pack size and economic coefficients are scenario assumptions, not observed supplier/inventory facts.",
        })
        scenario_cache.append((scenario, target, target_meta, baseline_id, candidate_id))

    for s_idx, (scenario, target, target_meta, baseline_id, candidate_id) in enumerate(scenario_cache):
        probabilities = scenario["lead_probabilities"]
        for role, config_id in (("baseline", baseline_id), ("candidate", candidate_id)):
            cfg = next(x for x in CONFIGS if x["config_id"] == config_id)
            rng = np.random.default_rng(SEED_V49 + s_idx * 100_000)
            draws = rng.choice(
                np.array([1, 2, 3, 4], dtype=int),
                size=(len(item["series_ids"]), target.week.nunique()),
                p=np.array([probabilities[k] for k in (1, 2, 3, 4)], dtype=float),
            )
            _, _, detail = _simulate_v49(target, item, probabilities, scenario, cfg, draws, True)
            detail["scenario_id"] = scenario["scenario_id"]
            detail["role"] = role
            detail["replicate"] = 0
            detail_samples.append(detail)

    decisions = pd.concat(detail_samples, ignore_index=True)
    milp_rows = []
    for scenario_index, (scenario, _, _, _, _) in enumerate(scenario_cache):
        scenario_id = scenario["scenario_id"]
        group = decisions[(decisions.role == "candidate") & (decisions.scenario_id == scenario_id)]
        test_weeks = sorted(group.loc[group.split == "test", "week"].unique())
        sampled_weeks = test_weeks[:1]
        expected_lead = float(sum(k * v for k, v in scenario["lead_probabilities"].items()))
        for week in sampled_weeks:
            state = group[group.week == week].sort_values("series_id")
            request_all = state.requested_qty.to_numpy(int)
            shortfall_all = state.shortfall.to_numpy(float)
            lot_all = state.lot_size.to_numpy(int)
            price_all = state.unit_value_proxy.to_numpy(float)
            scale_all = state.mean_demand.to_numpy(float)
            budget_limit = float(state.budget_limit_week.iloc[0])
            capacity_room = float(state.capacity_room_before_order.iloc[0])
            economics = {
                "shortage_rate": float(state.shortage_rate.iloc[0]),
                "holding_rate": float(state.holding_rate.iloc[0]),
                "fixed_order_cost": float(state.fixed_order_cost_rate.iloc[0]),
            }
            active = np.flatnonzero(request_all > 0)
            sample_rng = np.random.default_rng(SEED_V49 + scenario_index * 1009 + int(week))
            chosen = np.sort(sample_rng.choice(active, size=min(10, len(active)), replace=False)) if len(active) else active
            # Bound the exact subproblem to at most four lots per sampled series.
            # Scale both resource envelopes by the sample's share of requested
            # spend/coverage so the sampled problem retains the full-state tightness.
            request = np.minimum(request_all[chosen], 2 * lot_all[chosen])
            shortfall = shortfall_all[chosen]
            lot = lot_all[chosen]
            price = price_all[chosen]
            scale = scale_all[chosen]
            all_spend = float(np.sum(request_all * price_all))
            sample_spend = float(np.sum(request * price))
            all_capacity = float(np.sum(request_all / scale_all))
            sample_capacity = float(np.sum(request / scale))
            sample_budget_limit = budget_limit * sample_spend / all_spend if all_spend > 0 else 0.0
            sample_capacity_room = capacity_room * sample_capacity / all_capacity if all_capacity > 0 else 0.0
            greedy = _allocate_marginal_lots(
                request, shortfall, lot, price, scale, sample_budget_limit, sample_capacity_room,
                economics, expected_lead, float(scenario["capital_rate"]),
            )
            exact = solve_marginal_lot_milp(
                shortfall, request, lot, price, scale, sample_budget_limit, sample_capacity_room,
                economics, expected_lead, float(scenario["capital_rate"]), time_limit=0.5,
            )
            benefits = _lot_benefits(shortfall, lot, price, scale, economics, expected_lead, float(scenario["capital_rate"]))
            def objective(qty):
                total = 0.0
                for i, q in enumerate(qty):
                    for j in range(int(q // lot[i])):
                        total += float(benefits[i][j])
                return total
            greedy_obj = objective(greedy)
            milp_obj = exact["objective"]
            milp_rows.append({
                "scenario_id": scenario_id, "week": int(week), "solver_success": bool(exact["success"]),
                "sampled_series_count": int(len(chosen)), "sampled_lot_variable_upper_bound": int(np.sum(request // lot)),
                "sample_request_fraction_by_spend": sample_spend / all_spend if all_spend > 0 else 0.0,
                "marginal_greedy_objective": greedy_obj, "milp_objective": milp_obj,
                "greedy_vs_milp_gap": ((milp_obj - greedy_obj) / max(abs(milp_obj), 1e-9)) if exact["success"] and milp_obj > 0 else None,
                "greedy_order_units": int(greedy.sum()), "milp_order_units": int(exact["quantity"].sum()),
                "greedy_budget_value": float(np.sum(greedy * price)),
                "milp_budget_value": float(np.sum(exact["quantity"] * price)),
                "milp_capacity_usage": float(np.sum(exact["quantity"] / scale)),
                "budget_limit": sample_budget_limit, "capacity_room_before_order": sample_capacity_room,
                "capital_rate_per_week": float(scenario["capital_rate"]),
                "subproblem_scope": "random sample up to 10 active series, capped at two lots per series; resources scaled by sample share",
            })

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(policy_metrics).to_csv(out / "policy_grid_metrics.csv", index=False)
    pd.DataFrame(replicate_metrics).to_csv(out / "replicate_metrics.csv", index=False)
    pd.DataFrame(weekly_metrics).to_csv(out / "weekly_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "selected_policies.csv", index=False)
    pd.DataFrame(scenario_results).to_csv(out / "scenario_summary.csv", index=False)
    decisions.to_csv(out / "selected_decisions_audit_sample.csv.gz", index=False, compression="gzip")
    pd.DataFrame(milp_rows).to_csv(out / "milp_snapshot_metrics.csv", index=False)

    manifest = {
        "version": "0.49",
        "mode": "capital_aware_marginal_lot_and_service_floor_constrained_policy_selection",
        "dataset": "UCI Online Retail II prepared product-country weekly recorded-sales panel",
        "source_transaction_rows": int(quality.get("raw_rows", 0)),
        "valid_source_transaction_rows": int(quality.get("valid_rows", 0)),
        "panel_rows": int(len(panel)), "series_count": int(len(item["series_ids"])),
        "chronological_splits": {
            split: {"rows": int((d.split == split).sum()), "weeks": int(d.loc[d.split == split, "week"].nunique())}
            for split in ("train", "calibration", "selection", "test")
        },
        "calendar_cut_indices": cuts,
        "lead_distributions": LEAD_DISTRIBUTIONS,
        "capital_rates_per_week": list(CAPITAL_RATES),
        "service_floor_tolerance_vs_baseline": SERVICE_TOLERANCE,
        "replications": replications, "random_seed": SEED_V49,
        "common_random_numbers": "All eight policies share the same lead-time draws within each scenario and replication.",
        "policy_configs": list(CONFIGS),
        "selection_rule": "Choose the lowest selection-cost proportional baseline; choose the lowest-cost non-proportional policy whose selection fill rate is no more than 0.5 percentage points below the baseline. If none qualifies, retain the baseline.",
        "capital_charge_definition": "Per week, capital_rate times selling-price-value proxy of average opening/closing on-hand plus pipeline inventory. It is a working-capital stress proxy, not acquisition cost or measured financing expense.",
        "marginal_lot_objective": "Per additional lot: shortage proxy benefit minus estimated excess holding, capital exposure and first-lot setup charges; re-rank after every accepted lot.",
        "marginal_dwell_assumption": "0.5 week within-period consumption plus excess quantity divided by twice training mean weekly demand; capital exposure also includes expected supplier lead time.",
        "milp_benchmark": "Exact one-period binary-lot allocation on a random sample of up to 10 active series and two lots per series, with lot-prefix constraints; not a global full-portfolio or multi-period optimum.",
        "data_warning": "Recorded sales can be censored by stockouts. No on-hand, stockout, supplier, PO, acquisition-cost or gross-margin ground truth is present.",
        "scenario_results": scenario_results,
        "source_quality": quality,
        "source_hashes": {
            "prepared_panel_sha256": "1d32ccc76a7bb6df0127d0a2d32368a22b79558e33603dfb82afd65eb0293831",
            "quality_manifest_sha256": "5a564a92ac78c8de95511862ea576aaf9aa5784572b8c732d38be78606c19d6f",
        },
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/v49/inventory"))
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    _, quality = load_prepared_panel(args.panel, args.quality)
    panel = pd.read_csv(args.panel)
    required = {"series_id", "date", "demand", "price"}
    if not required.issubset(panel.columns):
        parser.error(f"prepared panel missing V0.49 columns: {sorted(required - set(panel.columns))}")
    run(panel, quality, args.output, args.replications)
    print(f"V0.49 benchmark written to {args.output}")


if __name__ == "__main__":
    main()