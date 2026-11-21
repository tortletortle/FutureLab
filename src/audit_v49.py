"""Independent state, service-floor and MILP audit for V0.49."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def audit(directory: str | Path) -> dict:
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(path / "policy_grid_metrics.csv")
    selected = pd.read_csv(path / "selected_policies.csv")
    summary = pd.read_csv(path / "scenario_summary.csv")
    decisions = pd.read_csv(path / "selected_decisions_audit_sample.csv.gz", compression="gzip")
    milp = pd.read_csv(path / "milp_snapshot_metrics.csv")

    if manifest.get("version") != "0.49":
        raise AssertionError("wrong experiment version")
    scenario_count = int(summary.scenario_id.nunique())
    if scenario_count != 32:
        raise AssertionError(f"expected 32 scenarios, got {scenario_count}")
    if len(metrics) != scenario_count * 8 * 2:
        raise AssertionError("policy metric row count mismatch")
    if len(selected) != scenario_count * 2 or selected.duplicated(["scenario_id", "role"]).any():
        raise AssertionError("selected policy rows malformed")
    if decisions.duplicated(["scenario_id", "role", "replicate", "series_id", "week"]).any():
        raise AssertionError("duplicate decision rows")

    # Reproduce all selection decisions from selection-only metrics.
    for scenario_id, rows in selected.groupby("scenario_id"):
        sm = metrics[(metrics.scenario_id == scenario_id) & (metrics.split == "selection")]
        baseline_pool = sm[sm.allocator == "proportional"].sort_values(["mean_cost", "config_id"])
        if baseline_pool.empty:
            raise AssertionError("missing proportional baseline metrics")
        baseline_id = str(baseline_pool.iloc[0].config_id)
        baseline_fill = float(baseline_pool.iloc[0].mean_fill_rate)
        floor = max(0.0, baseline_fill - float(manifest["service_floor_tolerance_vs_baseline"]))
        candidate_pool = sm[sm.allocator != "proportional"]
        feasible = candidate_pool[candidate_pool.mean_fill_rate >= floor]
        baseline_row = rows[rows.role == "baseline"].iloc[0]
        candidate_row = rows[rows.role == "candidate"].iloc[0]
        if str(baseline_row.config_id) != baseline_id:
            raise AssertionError("baseline was not selected on selection-period cost")
        if abs(float(baseline_row.selection_fill_rate) - baseline_fill) > 1e-10:
            raise AssertionError("baseline selection fill mismatch")
        if abs(float(candidate_row.service_floor) - floor) > 1e-10:
            raise AssertionError("service floor mismatch")
        if feasible.empty:
            if str(candidate_row.config_id) != baseline_id or bool(candidate_row.service_floor_feasible):
                raise AssertionError("infeasible service floor must retain baseline")
        else:
            expected = feasible.sort_values(["mean_cost", "config_id"]).iloc[0]
            if str(candidate_row.config_id) != str(expected.config_id):
                raise AssertionError("candidate is not the lowest-cost service-feasible policy")
            if not bool(candidate_row.service_floor_feasible):
                raise AssertionError("feasible candidate incorrectly marked infeasible")

    checks = 0
    for (scenario_id, role), group in decisions.groupby(["scenario_id", "role"], sort=False):
        group = group.sort_values(["series_id", "week"])
        for week, wg in group.groupby("week", sort=True):
            spend = float(np.sum(wg.order_qty.to_numpy(float) * wg.unit_value_proxy.to_numpy(float)))
            budget = float(wg.budget_limit_week.iloc[0])
            if spend > budget + 1e-6 or not np.allclose(wg.budget_spend_week, spend, atol=1e-5):
                raise AssertionError("weekly budget constraint/accounting failed")
            capacity = float(np.sum(
                (wg.on_hand_start.to_numpy(float) + wg.pipeline_before.to_numpy(float) + wg.order_qty.to_numpy(float))
                / wg.mean_demand.to_numpy(float)
            ))
            if capacity > float(wg.capacity_limit.iloc[0]) + 1e-8:
                raise AssertionError("weekly capacity constraint failed")
            if not np.allclose(wg.capacity_usage_after_week, capacity, atol=1e-6):
                raise AssertionError("capacity accounting mismatch")
            if (wg.order_qty.to_numpy(int) % wg.lot_size.to_numpy(int)).any():
                raise AssertionError("MOQ/pack multiple violated")
            if (wg.order_qty > wg.requested_qty).any():
                raise AssertionError("allocated order exceeds request")
            if (wg.fulfilled + wg.lost_sales != wg.demand).any():
                raise AssertionError("demand conservation failed")
            if (wg.on_hand_end != wg.on_hand_start - wg.fulfilled).any():
                raise AssertionError("on-hand balance failed")
            if (wg.pipeline_after != wg.pipeline_before + wg.order_qty).any():
                raise AssertionError("pipeline balance failed")
            exposure_units = 0.5 * (
                wg.on_hand_start.to_numpy(float) + wg.pipeline_before.to_numpy(float)
                + wg.on_hand_end.to_numpy(float) + wg.pipeline_after.to_numpy(float)
            )
            exposure_value = exposure_units * wg.unit_value_proxy.to_numpy(float)
            capital_charge = exposure_value * wg.capital_rate.to_numpy(float)
            if not np.allclose(wg.capital_exposure_units, exposure_units, atol=1e-8):
                raise AssertionError("capital exposure units mismatch")
            if not np.allclose(wg.capital_exposure_value, exposure_value, atol=1e-7):
                raise AssertionError("capital exposure value mismatch")
            if not np.allclose(wg.capital_charge, capital_charge, atol=1e-7):
                raise AssertionError("capital charge mismatch")
            expected_cost = (
                wg.holding_cost.to_numpy(float) + wg.lost_sales_cost.to_numpy(float)
                + wg.fixed_order_cost.to_numpy(float) + capital_charge
            )
            if not np.allclose(wg.total_cost.to_numpy(float), expected_cost, atol=1e-7):
                raise AssertionError("cost decomposition failed")
            checks += len(wg)

        for sid, sku in group.groupby("series_id", sort=False):
            sku = sku.sort_values("week").reset_index(drop=True)
            due = {}
            for row in sku.itertuples(index=False):
                expected_arrival = int(due.get(int(row.week), 0))
                if int(row.arrival_qty) != expected_arrival:
                    raise AssertionError("arrival schedule mismatch")
                if row.order_qty > 0:
                    if int(row.order_lead_time) not in (1, 2, 3, 4):
                        raise AssertionError("invalid realized lead time")
                    if int(row.arrival_week) != int(row.week) + int(row.order_lead_time):
                        raise AssertionError("arrival week mismatch")
                    due[int(row.arrival_week)] = due.get(int(row.arrival_week), 0) + int(row.order_qty)
                elif int(row.order_lead_time) != 0 or int(row.arrival_week) != -1:
                    raise AssertionError("non-order row contains lead-time metadata")
            for j in range(1, len(sku)):
                prev, cur = sku.iloc[j - 1], sku.iloc[j]
                if int(cur.on_hand_start) != int(prev.on_hand_end) + int(cur.arrival_qty):
                    raise AssertionError("on-hand continuity failed")
                if int(cur.pipeline_before) != int(prev.pipeline_after) - int(cur.arrival_qty):
                    raise AssertionError("pipeline continuity failed")
            if int(sku.iloc[-1].pipeline_after) != 0:
                raise AssertionError("terminal pipeline not cleared")
        test = group[group.split == "test"]
        final_weeks = sorted(test.week.unique())[-4:]
        if test[test.week.isin(final_weeks)].order_qty.sum() != 0:
            raise AssertionError("orders exist in terminal test tail")

    if checks != len(decisions):
        raise AssertionError("not all selected decision rows audited")
    if len(milp) != scenario_count:
        raise AssertionError("expected one MILP snapshot per scenario")
    if (milp.sampled_series_count > 10).any() or (milp.sampled_lot_variable_upper_bound > 20).any():
        raise AssertionError("MILP sample exceeded its declared size bound")
    if not milp.solver_success.astype(bool).all():
        raise AssertionError("one or more MILP snapshots did not solve to optimal status")
    gaps = milp.greedy_vs_milp_gap.dropna()
    if (gaps < -1e-6).any():
        raise AssertionError("greedy objective exceeds exact MILP objective")
    if (milp.milp_budget_value > milp.budget_limit + 1e-6).any():
        raise AssertionError("MILP allocation exceeds budget")
    if (milp.milp_capacity_usage > milp.capacity_room_before_order + 1e-8).any():
        raise AssertionError("MILP allocation exceeds capacity")

    result = {
        "version": "0.49", "status": "passed", "scenario_count": scenario_count,
        "policy_metric_rows": int(len(metrics)), "selected_policy_roles": int(len(selected)),
        "decision_rows_checked": int(checks), "milp_snapshots": int(len(milp)),
        "milp_successes": int(milp.solver_success.astype(bool).sum()),
        "median_marginal_greedy_to_milp_gap": float(gaps.median()) if len(gaps) else None,
        "max_marginal_greedy_to_milp_gap": float(gaps.max()) if len(gaps) else None,
        "service_floor_scenarios_feasible": int(summary.selection_service_floor_feasible.astype(bool).sum()),
        "test_selection_floor_scenarios_met": int(summary.test_service_floor_met.astype(bool).sum()),
        "test_service_noninferiority_scenarios_met": int(summary.test_service_noninferiority_vs_baseline.astype(bool).sum()),
        "checks": [
            "selection-only baseline and service-floor candidate reproduction",
            "weekly budget and normalized capacity constraints",
            "MOQ/pack and request upper bounds",
            "demand and on-hand/pipeline conservation",
            "shipment arrival timing and terminal pipeline clearance",
            "capital exposure and financing-charge recomputation",
            "holding/lost-sales/setup/capital cost decomposition",
            "terminal no-order horizon",
            "marginal-lot greedy objective bounded by exact prefix-constrained MILP",
        ],
        "scope_warning": "This remains a recorded-sales replay with assumed costs and supplier lead times. The MILP solves only a random sample of up to 10 active series and two lots per series in each one-period state; it does not establish global multi-period optimality or real enterprise savings.",
    }
    (path / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()