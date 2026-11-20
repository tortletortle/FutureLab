"""Independent audit for V0.48 constrained stochastic inventory experiments."""
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
    decisions = pd.read_csv(path / "selected_decisions_audit_sample.csv")
    milp = pd.read_csv(path / "milp_snapshot_metrics.csv")

    if manifest.get("version") != "0.48":
        raise AssertionError("wrong experiment version")
    if len(metrics) != len(manifest["scenario_results"]) * 5 * 2:
        raise AssertionError("policy metric row count mismatch")
    if selected.duplicated(["scenario_id", "role"]).any():
        raise AssertionError("duplicate selected policy role")
    if decisions.duplicated(["scenario_id", "role", "replicate", "series_id", "week"]).any():
        raise AssertionError("duplicate decision sample rows")
    if not decisions.order_qty.ge(0).all() or not decisions.on_hand_end.ge(0).all():
        raise AssertionError("negative order or inventory")

    # Recompute frozen family champions using selection metrics only.
    selection = metrics[metrics.split == "selection"]
    for scenario_id, chosen in selected.groupby("scenario_id"):
        base = chosen[chosen.role == "baseline"].iloc[0].config_id
        cand = chosen[chosen.role == "candidate"].iloc[0].config_id
        s = selection[selection.scenario_id == scenario_id]
        expected_base = s[s.allocator == "proportional"].sort_values("mean_cost").iloc[0].config_id
        expected_cand = s[s.allocator == "marginal_greedy"].sort_values("mean_cost").iloc[0].config_id
        if base != expected_base or cand != expected_cand:
            raise AssertionError("selected policy was not the selection-period family champion")

    # Recompute the full state transition and global constraints for the saved
    # representative replication of each selected policy.
    checks = 0
    for (scenario_id, role), group in decisions.groupby(["scenario_id", "role"], sort=False):
        group = group.sort_values(["series_id", "week"])
        budget_limit = float(group.budget_limit_week.iloc[0])
        capacity_limit = float(group.capacity_limit.iloc[0])
        for week, wg in group.groupby("week", sort=True):
            if not np.allclose(wg.budget_spend_week, wg.budget_spend_week.iloc[0]):
                raise AssertionError("weekly budget spend not constant across SKU rows")
            spend = float(np.sum(wg.order_qty.to_numpy(float) * wg.unit_value_proxy.to_numpy(float)))
            if spend > budget_limit + 1e-6 or abs(spend - float(wg.budget_spend_week.iloc[0])) > 1e-5:
                raise AssertionError("weekly budget constraint/accounting failed")
            capacity = float(np.sum(
                (wg.on_hand_start.to_numpy(float) + wg.pipeline_before.to_numpy(float) + wg.order_qty.to_numpy(float))
                / wg.mean_demand.to_numpy(float)
            ))
            if capacity > capacity_limit + 1e-8 or abs(capacity - float(wg.capacity_usage_after_week.iloc[0])) > 1e-6:
                raise AssertionError("weekly capacity constraint/accounting failed")
            if (wg.order_qty.to_numpy(int) % wg.lot_size.to_numpy(int)).any():
                raise AssertionError("pack multiple violated")
            if (wg.order_qty > wg.requested_qty).any():
                raise AssertionError("allocated quantity exceeds requested quantity")
            if (wg.fulfilled + wg.lost_sales != wg.demand).any():
                raise AssertionError("demand conservation failed")
            if (wg.on_hand_end != wg.on_hand_start - wg.fulfilled).any():
                raise AssertionError("on-hand inventory balance failed")
            if (wg.pipeline_after != wg.pipeline_before + wg.order_qty).any():
                raise AssertionError("pipeline balance failed")
            expected_cost = (
                wg.holding_cost.to_numpy(float)
                + wg.lost_sales_cost.to_numpy(float)
                + wg.fixed_order_cost.to_numpy(float)
            )
            if not np.allclose(wg.total_cost.to_numpy(float), expected_cost, atol=1e-8):
                raise AssertionError("cost decomposition failed")
            checks += len(wg)

        # SKU-level arrival schedule and inventory state continuity.
        for sid, sku in group.groupby("series_id", sort=False):
            sku = sku.sort_values("week").reset_index(drop=True)
            due_map = {}
            for row in sku.itertuples(index=False):
                expected_arrival = int(due_map.get(int(row.week), 0))
                if int(row.arrival_qty) != expected_arrival:
                    raise AssertionError("shipment arrival does not match earlier order schedule")
                if row.order_qty > 0:
                    if int(row.order_lead_time) not in (1, 2, 3, 4):
                        raise AssertionError("invalid realized lead time")
                    if int(row.arrival_week) != int(row.week) + int(row.order_lead_time):
                        raise AssertionError("order arrival date mismatch")
                    due_map[int(row.arrival_week)] = due_map.get(int(row.arrival_week), 0) + int(row.order_qty)
                elif int(row.order_lead_time) != 0 or int(row.arrival_week) != -1:
                    raise AssertionError("non-order row contains an order lead time")
            for j in range(1, len(sku)):
                prev, cur = sku.iloc[j - 1], sku.iloc[j]
                if int(cur.on_hand_start) != int(prev.on_hand_end) + int(cur.arrival_qty):
                    raise AssertionError("on-hand state continuity failed")
                if int(cur.pipeline_before) != int(prev.pipeline_after) - int(cur.arrival_qty):
                    raise AssertionError("pipeline state continuity failed")
            last = sku.iloc[-1]
            if int(last.pipeline_after) != 0:
                raise AssertionError("terminal pipeline not fully observed")
        # No orders are placed in the terminal max-lead-time test tail.
        test = group[group.split == "test"]
        if len(test):
            final_weeks = sorted(test.week.unique())[-4:]
            if test[test.week.isin(final_weeks)].order_qty.sum() != 0:
                raise AssertionError("orders placed in terminal test tail")
    if checks != len(decisions):
        raise AssertionError("not all saved decision rows were audited")

    successful_milp = milp[milp.solver_success.astype(bool)]
    if len(successful_milp) != len(milp):
        raise AssertionError("one or more MILP snapshots did not reach an optimal solution")
    valid_gaps = successful_milp.greedy_vs_milp_gap.dropna()
    if (valid_gaps < -1e-6).any():
        raise AssertionError("greedy objective exceeds MILP optimum; objective mismatch")
    if (successful_milp.milp_budget_value > successful_milp.budget_limit + 1e-6).any():
        raise AssertionError("MILP benchmark exceeds budget")
    if (successful_milp.milp_capacity_usage > successful_milp.capacity_room_before_order + 1e-8).any():
        raise AssertionError("MILP benchmark exceeds capacity room")

    result = {
        "version": "0.48",
        "status": "passed",
        "decision_rows_checked": int(checks),
        "scenario_count": int(selected.scenario_id.nunique()),
        "selected_policy_roles_checked": int(len(selected)),
        "milp_snapshots": int(len(milp)),
        "milp_successes": int(len(successful_milp)),
        "median_greedy_to_milp_objective_gap": float(valid_gaps.median()) if len(valid_gaps) else None,
        "max_greedy_to_milp_objective_gap": float(valid_gaps.max()) if len(valid_gaps) else None,
        "checks": [
            "selection-only policy-family winner reproduction",
            "weekly procurement budget",
            "normalized aggregate capacity",
            "MOQ/pack-size multiples",
            "request upper bounds",
            "demand fulfillment balance",
            "on-hand and pipeline state conservation",
            "shipment arrival timing",
            "cost decomposition",
            "terminal pipeline clearance",
            "terminal no-order horizon",
            "greedy allocation objective does not exceed exact MILP snapshot optimum",
        ],
        "scope_warning": "MILP snapshots solve a single-period allocation subproblem only; they do not prove global multi-period optimality.",
    }
    (path / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
