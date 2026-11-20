"""Multi-period, periodic-review inventory policy benchmark.

Demand is the UCI Online Retail II *recorded positive sales* proxy from V46.
This module simulates lead-time pipelines, lost sales, holding cost and fixed
order cost. It is not a reconstruction of the retailer's actual inventory.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from retail_v46 import prepare, features, COLS, block_ci, SEED

Q_LEVELS = (0.50, 0.75, 0.90)
LEAD_TIMES = (1, 2, 4)
CANDIDATE_PREFIX = "group_horizon_q"
PREPARED_PANEL_SHA = "1d32ccc76a7bb6df0127d0a2d32368a22b79558e33603dfb82afd65eb0293831"
PREPARED_QUALITY_SHA = "5a564a92ac78c8de95511862ea576aaf9aa5784572b8c732d38be78606c19d6f"

COST_SCENARIOS = {
    "balanced": {"shortage_cost": 3.0, "holding_cost": 1.0, "fixed_order_cost": 0.0},
    "service_priority": {"shortage_cost": 6.0, "holding_cost": 1.0, "fixed_order_cost": 0.0},
    "capital_pressure": {"shortage_cost": 3.0, "holding_cost": 2.0, "fixed_order_cost": 0.0},
    "order_setup_cost": {"shortage_cost": 3.0, "holding_cost": 1.0, "fixed_order_cost": 10.0},
}


def load_prepared_panel(panel_path: Path, quality_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the already audited V46 aggregation without reparsing 1.07M XLSX rows."""
    from retail_v46 import sha
    if sha(panel_path) != PREPARED_PANEL_SHA:
        raise ValueError("prepared panel hash mismatch")
    if sha(quality_path) != PREPARED_QUALITY_SHA:
        raise ValueError("prepared quality manifest hash mismatch")
    quality = json.loads(Path(quality_path).read_text())
    if quality.get("zip_sha256") != "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb":
        raise ValueError("prepared panel source ZIP provenance mismatch")
    if quality.get("xlsx_sha256") != "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980":
        raise ValueError("prepared panel source XLSX provenance mismatch")
    panel = pd.read_csv(panel_path)
    required = {"series_id", "date", "demand", "time_index"}
    if not required.issubset(panel.columns):
        raise ValueError("prepared panel schema mismatch")
    panel = panel[["series_id", "date", "demand"]].copy()
    panel["date"] = pd.to_datetime(panel.date, errors="raise")
    panel["demand"] = pd.to_numeric(panel.demand, errors="raise")
    if panel.empty or panel.duplicated(["series_id", "date"]).any() or panel.series_id.isna().any():
        raise ValueError("prepared panel has duplicate or missing keys")
    if not np.isfinite(panel.demand).all() or (panel.demand < 0).any() or (panel.date.dt.dayofweek != 6).any():
        raise ValueError("prepared panel contains invalid demand or calendar values")
    if len(panel) != quality.get("selected_panel_rows") or panel.series_id.nunique() != quality.get("selected_series"):
        raise ValueError("prepared panel does not match its quality manifest")
    quality = {**quality, "prepared_panel_sha256": PREPARED_PANEL_SHA, "prepared_quality_sha256": PREPARED_QUALITY_SHA}
    return panel, quality


def _fit_forecasts(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit on train only; calibrate cumulative residuals on calibration only."""
    train = d[d.split == "train"].copy()
    cal = d[d.split == "calibration"].copy()
    target = d[d.split.isin(["selection", "test"])].copy()
    if min(len(train), len(cal), len(target)) == 0:
        raise ValueError("all chronological partitions must be non-empty")

    common = dict(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=30,
        learning_rate=0.08, l2_regularization=2, random_state=SEED,
        categorical_features=[True] + [False] * (len(COLS) - 1),
    )
    point = HistGradientBoostingRegressor(**common).fit(
        train[COLS], np.log1p(train.demand)
    )
    hgb_q = HistGradientBoostingRegressor(
        **common, loss="quantile", quantile=0.75
    ).fit(train[COLS], train.demand)
    xgb_q = XGBRegressor(
        n_estimators=180, max_depth=4, learning_rate=0.055,
        objective="reg:quantileerror", quantile_alpha=0.75,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=5,
        n_jobs=2, random_state=SEED,
    ).fit(train[COLS], train.demand)

    for frame in (cal, target):
        frame["hgb_mean"] = np.expm1(point.predict(frame[COLS])).clip(0)
        frame["hgb_q75"] = hgb_q.predict(frame[COLS]).clip(0)
        frame["xgb_q75"] = xgb_q.predict(frame[COLS]).clip(0)

    train_mean = train.groupby("series_id").demand.mean()
    edges = np.quantile(train_mean.to_numpy(), [1 / 3, 2 / 3])
    sid_group = {
        sid: int(np.searchsorted(edges, value, side="right"))
        for sid, value in train_mean.items()
    }
    cal["residual"] = cal.demand - cal.hgb_mean
    cal = cal.sort_values(["series_id", "week"]).reset_index(drop=True)

    horizon_calibration = {}
    for lead in LEAD_TIMES:
        pooled: list[float] = []
        by_group: dict[int, list[float]] = {0: [], 1: [], 2: []}
        window_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for sid, g in cal.groupby("series_id", sort=False):
            g = g.sort_values("week")
            residual = g.residual.to_numpy(float)
            scale = g.scale.to_numpy(float)
            if len(g) < lead:
                continue
            group_id = sid_group[sid]
            for start in range(len(g) - lead + 1):
                denom = max(float(scale[start:start + lead].sum()), 1.0)
                normalized = float(residual[start:start + lead].sum() / denom)
                pooled.append(normalized)
                by_group[group_id].append(normalized)
                window_counts[group_id] += 1
        if len(pooled) < 50:
            raise ValueError(f"insufficient horizon calibration windows for lead={lead}")
        global_q = {q: float(np.quantile(pooled, q)) for q in Q_LEVELS}
        group_q = {}
        for group_id, values in by_group.items():
            group_q[group_id] = {
                q: float(np.quantile(values, q)) if len(values) >= 50 else global_q[q]
                for q in Q_LEVELS
            }
        horizon_calibration[lead] = {
            "global": global_q,
            "group": group_q,
            "window_counts": window_counts,
            "pooled_windows": len(pooled),
        }

    # Generate all one-step and cumulative lead-time target levels. Each lead-time
    # horizon assumes the current one-step mean/scale remains representative.
    target["group_id"] = target.series_id.map(sid_group).astype(int)
    for lead in LEAD_TIMES:
        target[f"rolling13_L{lead}"] = lead * target.quantile13.clip(lower=0)
        target[f"hgb_mean_L{lead}"] = lead * target.hgb_mean.clip(lower=0)
        target[f"hgb_q75_L{lead}"] = lead * target.hgb_q75.clip(lower=0)
        target[f"xgb_q75_L{lead}"] = lead * target.xgb_q75.clip(lower=0)
        meta = horizon_calibration[lead]
        for q in Q_LEVELS:
            qtag = int(round(q * 100))
            target[f"global_horizon_q{qtag}_L{lead}"] = (
                lead * target.hgb_mean + lead * target.scale * meta["global"][q]
            ).clip(lower=0)
            group_quantile = np.array([meta["group"][int(group_id)][q] for group_id in target.group_id], dtype=float)
            target[f"group_horizon_q{qtag}_L{lead}"] = np.maximum(
                0.0,
                lead * target.hgb_mean.to_numpy(float)
                + lead * target.scale.to_numpy(float) * group_quantile,
            )
    return target.reset_index(drop=True), {
        "training_series_group_edges": edges.tolist(),
        "horizon_calibration": {
            str(k): {
                "global": {str(q): v for q, v in val["global"].items()},
                "group": {
                    str(g): {str(q): v for q, v in qv.items()}
                    for g, qv in val["group"].items()
                },
                "window_counts": val["window_counts"],
                "pooled_windows": val["pooled_windows"],
            }
            for k, val in horizon_calibration.items()
        },
        "models_fit_on": "train_only",
        "residual_calibration_on": "calibration_only",
    }


def policy_grid(lead: int) -> list[dict]:
    methods = [
        f"rolling13_L{lead}", f"hgb_mean_L{lead}", f"hgb_q75_L{lead}",
        f"xgb_q75_L{lead}", f"global_horizon_q50_L{lead}",
        f"global_horizon_q75_L{lead}", f"global_horizon_q90_L{lead}",
        f"group_horizon_q50_L{lead}", f"group_horizon_q75_L{lead}",
        f"group_horizon_q90_L{lead}",
    ]
    configs = []
    for method in methods:
        configs.append({"method": method, "policy": "base_stock", "trigger_ratio": None})
        for ratio in (0.50, 0.75, 0.90):
            configs.append({"method": method, "policy": "sS", "trigger_ratio": ratio})
    return configs


def _simulate_one(target: pd.DataFrame, lead: int, config: dict, costs: dict) -> pd.DataFrame:
    """Vectorized periodic review: receive -> order -> demand -> charge costs."""
    method = config["method"]
    if method not in target.columns:
        raise ValueError(f"missing order-up-to forecast: {method}")
    if lead < 1:
        raise ValueError("lead time must be at least one period")

    series_ids = np.array(sorted(target.series_id.unique()), dtype=object)
    weeks = np.array(sorted(target.week.unique()), dtype=int)
    index = pd.MultiIndex.from_product([series_ids, weeks], names=["series_id", "week"])
    aligned = target.set_index(["series_id", "week"]).reindex(index).reset_index()
    if aligned[[method, "demand", "scale", "date", "split"]].isna().any().any():
        raise ValueError("inventory simulator requires a complete SKU-week panel")
    n_series, n_periods = len(series_ids), len(weeks)
    forecasts = np.maximum(aligned[method].to_numpy(float).reshape(n_series, n_periods), 0)
    demands = np.ceil(np.maximum(aligned.demand.to_numpy(float).reshape(n_series, n_periods), 0)).astype(int)
    scales = aligned.scale.to_numpy(float).reshape(n_series, n_periods)
    dates = aligned.date.astype(str).to_numpy().reshape(n_series, n_periods)
    splits = aligned.split.astype(str).to_numpy().reshape(n_series, n_periods)
    if not (splits == splits[0:1, :]).all():
        raise ValueError("split boundaries must be aligned across SKU series")

    # Common initial inventory state, independent of policy forecast method.
    initial_scale = scales[:, 0]
    initial_stock = np.ceil(lead * initial_scale).astype(int)
    on_hand = initial_stock.copy()
    pipeline = np.zeros((n_series, lead), dtype=int)
    rows = []

    for t, week in enumerate(weeks):
        arrival = pipeline[:, 0].copy()
        if lead > 1:
            pipeline[:, :-1] = pipeline[:, 1:]
            pipeline[:, -1] = 0
        else:
            pipeline[:, 0] = 0
        on_hand += arrival
        on_hand_start = on_hand.copy()
        pipeline_before = pipeline.sum(axis=1)
        inventory_position = on_hand_start + pipeline_before
        target_level = forecasts[:, t]
        if config["policy"] == "base_stock":
            trigger_level = None
            order = np.ceil(np.maximum(0.0, target_level - inventory_position)).astype(int)
        elif config["policy"] == "sS":
            ratio = float(config["trigger_ratio"])
            trigger_level = ratio * target_level
            order = np.where(
                inventory_position <= trigger_level,
                np.ceil(np.maximum(0.0, target_level - inventory_position)),
                0,
            ).astype(int)
        else:
            raise ValueError("unknown inventory policy")

        pipeline[:, -1] += order
        pipeline_after = pipeline.sum(axis=1)
        fulfilled = np.minimum(on_hand_start, demands[:, t])
        lost = demands[:, t] - fulfilled
        on_hand = on_hand_start - fulfilled
        holding = float(costs["holding_cost"]) * on_hand
        lost_cost = float(costs["shortage_cost"]) * lost
        fixed = float(costs["fixed_order_cost"]) * (order > 0)
        total = holding + lost_cost + fixed
        rows.append(pd.DataFrame({
            "series_id": series_ids, "week": int(week), "date": dates[:, t],
            "split": splits[0, t], "lead_time": lead, "method": method,
            "policy": config["policy"], "trigger_ratio": (float(config["trigger_ratio"]) if config["trigger_ratio"] is not None else -1.0),
            "order_up_to": target_level, "initial_stock": initial_stock,
            "initial_scale": initial_scale, "arrival_qty": arrival,
            "on_hand_start": on_hand_start, "pipeline_before": pipeline_before,
            "inventory_position_start": inventory_position,
            "order_qty": order, "arrival_week": np.where(order > 0, int(week) + lead, -1),
            "pipeline_after": pipeline_after, "demand": demands[:, t],
            "fulfilled": fulfilled, "lost_sales": lost, "on_hand_end": on_hand,
            "holding_cost": holding, "lost_sales_cost": lost_cost,
            "fixed_order_cost": fixed, "total_cost": total,
        }))
    return pd.concat(rows, ignore_index=True)


def summarize(decisions: pd.DataFrame, split: str) -> dict:
    d = decisions[decisions.split == split]
    total_demand = float(d.demand.sum())
    return {
        "n": int(len(d)),
        "mean_cost_per_sku_week": float(d.total_cost.mean()),
        "mean_holding_cost": float(d.holding_cost.mean()),
        "mean_lost_sales_cost": float(d.lost_sales_cost.mean()),
        "mean_fixed_order_cost": float(d.fixed_order_cost.mean()),
        "lost_sales_units": int(d.lost_sales.sum()),
        "fill_rate": float(1 - d.lost_sales.sum() / total_demand) if total_demand else 1.0,
        "mean_on_hand": float(d.on_hand_end.mean()),
        "mean_order_qty": float(d.order_qty.mean()),
        "order_frequency": float((d.order_qty > 0).mean()),
    }


def run(panel: pd.DataFrame, quality: dict, out: Path) -> dict:
    d, cuts = features(panel)
    forecast, calibration_meta = _fit_forecasts(d)
    all_metrics = []
    selected_rows = []
    selected_decisions = []
    scenario_results = []

    for lead in LEAD_TIMES:
        configs = policy_grid(lead)
        for scenario_name, costs in COST_SCENARIOS.items():
            scenario_runs = []
            for config in configs:
                dec = _simulate_one(forecast, lead, config, costs)
                sel = summarize(dec, "selection")
                tst = summarize(dec, "test")
                config_id = f"{config['method']}|{config['policy']}|{config['trigger_ratio']}"
                row = {
                    "scenario": scenario_name, "lead_time": lead, "config_id": config_id,
                    "method": config["method"], "policy": config["policy"],
                    "trigger_ratio": config["trigger_ratio"],
                    "selection_cost": sel["mean_cost_per_sku_week"],
                    "test_cost": tst["mean_cost_per_sku_week"],
                    "selection_fill_rate": sel["fill_rate"], "test_fill_rate": tst["fill_rate"],
                    "selection_lost_sales_units": sel["lost_sales_units"],
                    "test_lost_sales_units": tst["lost_sales_units"],
                    "selection_mean_on_hand": sel["mean_on_hand"],
                    "test_mean_on_hand": tst["mean_on_hand"],
                    "selection_order_frequency": sel["order_frequency"],
                    "test_order_frequency": tst["order_frequency"],
                }
                all_metrics.append(row)
                scenario_runs.append((config, config_id, dec, row))

            baseline_runs = [x for x in scenario_runs if not x[0]["method"].startswith(CANDIDATE_PREFIX)]
            candidate_runs = [x for x in scenario_runs if x[0]["method"].startswith(CANDIDATE_PREFIX)]
            if not baseline_runs or not candidate_runs:
                raise RuntimeError("baseline or candidate grid unexpectedly empty")
            baseline = min(baseline_runs, key=lambda x: x[3]["selection_cost"])
            candidate = min(candidate_runs, key=lambda x: x[3]["selection_cost"])
            for role, chosen in (("baseline", baseline), ("candidate", candidate)):
                config, config_id, dec, metric_row = chosen
                selected_rows.append({"scenario": scenario_name, "lead_time": lead, "role": role, "config_id": config_id, **config})
                kept = dec.copy()
                kept["scenario"] = scenario_name
                kept["role"] = role
                kept["config_id"] = config_id
                selected_decisions.append(kept)
            base_test = baseline[2][baseline[2].split == "test"].sort_values(["series_id", "week"])
            cand_test = candidate[2][candidate[2].split == "test"].sort_values(["series_id", "week"])
            paired = cand_test.total_cost.to_numpy(float) - base_test.total_cost.to_numpy(float)
            if len(paired) != len(base_test) or not np.array_equal(cand_test[["series_id", "week"]].to_numpy(), base_test[["series_id", "week"]].to_numpy()):
                raise AssertionError("candidate and baseline test decisions are not paired")
            ci = block_ci(paired, cand_test.week.to_numpy(int), seed=SEED + lead)
            bcost = float(base_test.total_cost.mean())
            ccost = float(cand_test.total_cost.mean())
            scenario_results.append({
                "scenario": scenario_name, "lead_time": lead,
                "baseline_config": baseline[1], "candidate_config": candidate[1],
                "baseline_test_cost": bcost, "candidate_test_cost": ccost,
                "paired_delta_candidate_minus_baseline": float(paired.mean()),
                "relative_cost_reduction": float((bcost - ccost) / bcost) if bcost > 0 else None,
                "block_bootstrap_descriptive_ci": ci,
                "candidate_test_fill_rate": summarize(candidate[2], "test")["fill_rate"],
                "baseline_test_fill_rate": summarize(baseline[2], "test")["fill_rate"],
                "candidate_beats_baseline_on_test": bool(ccost < bcost),
                "evidence_warning": "Historical recorded-sales proxy; assumed costs; one retailer; no stockout truth or actual inventory records.",
            })

    metrics = pd.DataFrame(all_metrics)
    selected = pd.DataFrame(selected_rows)
    decisions = pd.concat(selected_decisions, ignore_index=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(out / "policy_grid_metrics.csv", index=False)
    selected.to_csv(out / "selected_policies.csv", index=False)
    decisions.to_csv(out / "selected_policy_decisions.csv", index=False)
    summary = {
        "mode": "multi_period_inventory_simulation_on_recorded_sales_proxy",
        "quality": quality,
        "calendar_cut_indices": cuts,
        "forecast_calibration": calibration_meta,
        "splits": {
            split: {
                "n": int((forecast.split == split).sum()),
                "weeks": int(forecast.loc[forecast.split == split, "week"].nunique()),
                "start": str(forecast.loc[forecast.split == split, "date"].min().date()),
                "end": str(forecast.loc[forecast.split == split, "date"].max().date()),
            }
            for split in ("selection", "test")
        },
        "lead_times": list(LEAD_TIMES),
        "cost_scenarios": COST_SCENARIOS,
        "policy_grid_size_per_scenario": len(policy_grid(1)),
        "policy_grid_definition": {
            "forecast_methods": [
                "rolling13", "hgb_mean", "hgb_q75", "xgb_q75",
                "global_horizon_q50/q75/q90", "group_horizon_q50/q75/q90",
            ],
            "policies": ["base_stock", "sS trigger ratios 0.50/0.75/0.90"],
            "candidate_family": "group_horizon_q50/q75/q90",
            "baseline_family": "all other forecast methods",
            "selection_rule": "minimum selection-period average total cost within each family and scenario",
        },
        "inventory_event_order": "receive due shipment -> review inventory position -> place replenishment order -> observe recorded sales -> fulfill/lost sales -> charge end-of-period holding and fixed order cost",
        "initial_inventory_rule": "ceil(lead_time * trailing_13_week_mean) shared across all policy methods per series",
        "rounding_rule": "orders and recorded weekly demand rounded upward to integer units",
        "random_seed": SEED,
        "versions": {
            "numpy": __import__("numpy").__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
        },
        "scenario_results": scenario_results,
        "limitations": [
            "Observed positive transactions are not latent demand; stockouts/censoring are unknown.",
            "No actual inventory, replenishment, supplier, capacity, shelf-life or stockout records exist.",
            "Costs are hypothetical dimensionless scenario weights, not currency or realized profit.",
            "One retailer and the same historical dataset previously used by V25/V46; not an external replication.",
            "Lead time is deterministic; no supplier disruption or stochastic delivery delay.",
            "Lead-time forecast approximates cumulative demand as L times current one-step mean plus calibration residual-window quantile.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Original UCI ZIP; full source re-parse")
    source.add_argument("--panel", type=Path, help="Hash-pinned V46 aggregated panel CSV")
    parser.add_argument("--quality", type=Path, help="Quality manifest required with --panel")
    parser.add_argument("--output", type=Path, default=Path("results/v47/inventory"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    if args.data:
        panel, quality = prepare(args.data)
    else:
        if not args.quality:
            parser.error("--quality is required with --panel")
        panel, quality = load_prepared_panel(args.panel, args.quality)
    run(panel, quality, args.output)


if __name__ == "__main__":
    main()
