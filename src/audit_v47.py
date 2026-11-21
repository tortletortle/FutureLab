"""Independent state-transition and selection audit for V47 policy outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd

from retail_v46 import block_ci, SEED
from inventory_v47 import COST_SCENARIOS, CANDIDATE_PREFIX


def audit(directory: str | Path) -> dict:
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("cost_scenarios") != COST_SCENARIOS:
        raise AssertionError("cost scenario definitions differ from the audited implementation")
    metrics = pd.read_csv(path / "policy_grid_metrics.csv")
    selected = pd.read_csv(path / "selected_policies.csv")
    decisions = pd.read_csv(path / "selected_policy_decisions.csv")

    expected_metrics = len(manifest["lead_times"]) * len(manifest["cost_scenarios"]) * int(manifest["policy_grid_size_per_scenario"])
    if len(metrics) != expected_metrics:
        raise AssertionError("policy-grid metric row count mismatch")
    if metrics.duplicated(["scenario", "lead_time", "config_id"]).any():
        raise AssertionError("duplicate policy-grid metric rows")
    if selected.duplicated(["scenario", "lead_time", "role"]).any():
        raise AssertionError("duplicate selected policy roles")
    if decisions.duplicated(["scenario", "lead_time", "role", "series_id", "week"]).any():
        raise AssertionError("duplicate selected policy decisions")
    if not (decisions.order_qty >= 0).all() or not (decisions.on_hand_end >= 0).all():
        raise AssertionError("negative inventory/order quantity")
    if not (decisions.fulfilled + decisions.lost_sales == decisions.demand).all():
        raise AssertionError("demand fulfillment balance failed")

    checked_series_periods = 0
    scenario_audits = []
    for scenario, lead, role in selected[["scenario", "lead_time", "role"]].itertuples(index=False, name=None):
        cfg = selected[(selected.scenario == scenario) & (selected.lead_time == lead) & (selected.role == role)].iloc[0]
        config_id = cfg.config_id
        d = decisions[(decisions.scenario == scenario) & (decisions.lead_time == lead) & (decisions.role == role)].copy()
        if set(d.config_id.unique()) != {config_id}:
            raise AssertionError("decision rows do not match selected config")
        cost = COST_SCENARIOS[scenario]
        for sid, g in d.sort_values("week").groupby("series_id", sort=False):
            g = g.sort_values("week")
            first = g.iloc[0]
            initial = int(first.initial_stock)
            expected_initial = int(math.ceil(int(lead) * float(first.initial_scale)))
            if initial < 0 or initial != expected_initial:
                raise AssertionError("initial stock is not the common lead-time historical scale")
            if not (g.initial_stock == initial).all() or not (g.initial_scale == float(first.initial_scale)).all():
                raise AssertionError("initial stock changed within a SKU trajectory")
            on_hand = initial
            pipeline: dict[int, int] = {}
            for row in g.itertuples(index=False):
                week = int(row.week)
                arrival = int(pipeline.pop(week, 0))
                on_hand_start = on_hand + arrival
                pipeline_before = int(sum(pipeline.values()))
                inventory_position = on_hand_start + pipeline_before
                target = float(row.order_up_to)
                if row.policy == "base_stock":
                    expected_order = int(math.ceil(max(0.0, target - inventory_position)))
                elif row.policy == "sS":
                    threshold = float(row.trigger_ratio) * target
                    expected_order = int(math.ceil(max(0.0, target - inventory_position))) if inventory_position <= threshold else 0
                else:
                    raise AssertionError("unknown policy in decisions")
                if expected_order != int(row.order_qty):
                    raise AssertionError("order decision does not follow selected policy")
                if int(row.arrival_qty) != arrival:
                    raise AssertionError("supplier arrival violates lead-time pipeline")
                if int(row.on_hand_start) != on_hand_start:
                    raise AssertionError("opening inventory balance mismatch")
                if int(row.pipeline_before) != pipeline_before:
                    raise AssertionError("pipeline-before balance mismatch")
                if int(row.inventory_position_start) != inventory_position:
                    raise AssertionError("inventory position mismatch")
                fulfilled = min(on_hand_start, int(row.demand))
                lost = int(row.demand) - fulfilled
                on_hand = on_hand_start - fulfilled
                if int(row.fulfilled) != fulfilled or int(row.lost_sales) != lost or int(row.on_hand_end) != on_hand:
                    raise AssertionError("demand allocation or on-hand balance mismatch")
                if expected_order:
                    arrival_week = week + int(lead)
                    if int(row.arrival_week) != arrival_week:
                        raise AssertionError("order arrival week mismatch")
                    pipeline[arrival_week] = pipeline.get(arrival_week, 0) + expected_order
                elif int(row.arrival_week) != -1:
                    raise AssertionError("zero order has a scheduled arrival")
                pipeline_after = int(sum(pipeline.values()))
                if int(row.pipeline_after) != pipeline_after:
                    raise AssertionError("pipeline-after balance mismatch")
                holding = float(cost["holding_cost"]) * on_hand
                lost_cost = float(cost["shortage_cost"]) * lost
                fixed = float(cost["fixed_order_cost"]) if expected_order else 0.0
                if not np.isclose(row.holding_cost, holding) or not np.isclose(row.lost_sales_cost, lost_cost) or not np.isclose(row.fixed_order_cost, fixed) or not np.isclose(row.total_cost, holding + lost_cost + fixed):
                    raise AssertionError("cost calculation mismatch")
                checked_series_periods += 1

        # Every policy in the same scenario starts each series with the same
        # initial inventory state, independent of the forecast method.
        peer = decisions[(decisions.scenario == scenario) & (decisions.lead_time == lead)]
        pivot = peer.groupby(["role", "series_id"]).initial_stock.first().unstack("role")
        if not (pivot.baseline == pivot.candidate).all():
            raise AssertionError("baseline and candidate initial inventories differ")
        scale_peer = peer.groupby(["role", "series_id"]).initial_scale.first().unstack("role")
        if not np.allclose(scale_peer.baseline, scale_peer.candidate):
            raise AssertionError("baseline and candidate initial scales differ")

    for result in manifest["scenario_results"]:
        scenario, lead = result["scenario"], int(result["lead_time"])
        group = metrics[(metrics.scenario == scenario) & (metrics.lead_time == lead)]
        baseline_grid = group[~group.method.str.startswith(CANDIDATE_PREFIX)]
        candidate_grid = group[group.method.str.startswith(CANDIDATE_PREFIX)]
        baseline_row = baseline_grid.loc[baseline_grid.selection_cost.idxmin()]
        candidate_row = candidate_grid.loc[candidate_grid.selection_cost.idxmin()]
        if baseline_row.config_id != result["baseline_config"] or candidate_row.config_id != result["candidate_config"]:
            raise AssertionError("selection winner differs from manifest")
        chosen = selected[(selected.scenario == scenario) & (selected.lead_time == lead)].set_index("role")
        if chosen.loc["baseline", "config_id"] != baseline_row.config_id or chosen.loc["candidate", "config_id"] != candidate_row.config_id:
            raise AssertionError("selected policy table does not match selection winner")
        b = decisions[(decisions.scenario == scenario) & (decisions.lead_time == lead) & (decisions.role == "baseline") & (decisions.split == "test")].sort_values(["series_id", "week"])
        c = decisions[(decisions.scenario == scenario) & (decisions.lead_time == lead) & (decisions.role == "candidate") & (decisions.split == "test")].sort_values(["series_id", "week"])
        if not np.array_equal(b[["series_id", "week"]].to_numpy(), c[["series_id", "week"]].to_numpy()):
            raise AssertionError("baseline/candidate test panel is not paired")
        bcost, ccost = float(b.total_cost.mean()), float(c.total_cost.mean())
        delta = float((c.total_cost.to_numpy() - b.total_cost.to_numpy()).mean())
        if not np.isclose(bcost, result["baseline_test_cost"]) or not np.isclose(ccost, result["candidate_test_cost"]) or not np.isclose(delta, result["paired_delta_candidate_minus_baseline"]):
            raise AssertionError("test cost summary does not match detailed decisions")
        ci = block_ci(c.total_cost.to_numpy() - b.total_cost.to_numpy(), c.week.to_numpy(int), seed=SEED + lead)
        if not np.allclose(ci, result["block_bootstrap_descriptive_ci"]):
            raise AssertionError("paired block bootstrap interval mismatch")
        scenario_audits.append({"scenario": scenario, "lead_time": lead, "baseline_test_cost": bcost, "candidate_test_cost": ccost, "paired_delta": delta, "ci": ci})

    return {
        "status": "passed",
        "checked_series_period_transitions": checked_series_periods,
        "selected_policy_pairs": len(scenario_audits),
        "scenario_audits": scenario_audits,
        "sha256": {
            name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ("policy_grid_metrics.csv", "selected_policies.csv", "selected_policy_decisions.csv", "manifest.json")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    result = audit(args.directory)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    (Path(args.directory) / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
