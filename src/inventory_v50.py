"""V0.50 walk-forward policy selection under temporal drift.

This module evaluates selectors on five expanding-origin folds before the frozen
16-week holdout. Every fold refits the forecast model and lead-time calibration
using only that fold's training and calibration windows. The frozen test is used
once, after all selector rules are frozen from prior walk-forward evidence.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from inventory_v47 import _fit_forecasts
from inventory_v48 import (
    LEAD_DISTRIBUTIONS,
    _fit_stochastic_targets,
    _item_parameters,
)
from inventory_v49 import (
    CONFIGS,
    SEED_V49,
    SERVICE_TOLERANCE,
    _scenario_grid_v49,
    _select_policies,
    _simulate_v49,
)
from retail_v46 import features

VERSION = "0.50"
SEED_V50 = 50050
FOLD_FIT_LENGTHS = (30, 36, 42, 48, 54)
CALIBRATION_LENGTH = 8
SELECTION_LENGTH = 6
FORWARD_LENGTH = 6
FINAL_TEST_LENGTH = 16
ROBUST_PASS_FRACTION = 0.80
CONSERVATIVE_CONFIDENCE = 0.90
CONSERVATIVE_MAX_SERVICE_SD = 0.02
BASELINE_CONFIGS = ("q50_proportional", "q75_proportional")
CANDIDATE_CONFIGS = tuple(
    x["config_id"] for x in CONFIGS if x["config_id"] not in BASELINE_CONFIGS
)
SELECTORS = (
    "v49_single_window",
    "walkforward_robust_80pct",
    "walkforward_conservative_lcb",
    "baseline_only",
)


def _build_fold_schedule(weeks, final_test_count: int = FINAL_TEST_LENGTH) -> list[dict]:
    """Build expanding-origin folds without consulting any final-test outcomes."""
    values = list(weeks)
    if len(values) != len(set(values)) or values != sorted(values):
        raise ValueError("weeks must be unique and strictly increasing")
    if final_test_count < 1 or len(values) - final_test_count < 74:
        raise ValueError("insufficient pre-test history for the frozen five-fold schedule")
    pre_test = values[:-final_test_count]
    final_test = values[-final_test_count:]
    schedule = []
    for fold_index, fit_length in enumerate(FOLD_FIT_LENGTHS, start=1):
        cal_start = fit_length
        selection_start = cal_start + CALIBRATION_LENGTH
        forward_start = selection_start + SELECTION_LENGTH
        end = forward_start + FORWARD_LENGTH
        if end > len(pre_test):
            raise ValueError(f"fold {fold_index} would overlap the final test")
        schedule.append({
            "fold_id": f"F{fold_index}",
            "fold_index": fold_index,
            "train_weeks": pre_test[:fit_length],
            "calibration_weeks": pre_test[cal_start:selection_start],
            "selection_weeks": pre_test[selection_start:forward_start],
            "forward_weeks": pre_test[forward_start:end],
            "final_test_weeks": final_test,
        })
    return schedule


def _assign_fold_splits(d: pd.DataFrame, fold: dict) -> pd.DataFrame:
    """Assign causal train/calibration/selection/forward labels to feature rows."""
    frame = d.copy()
    frame["split"] = "unused"
    assignments = (
        ("train_weeks", "train"),
        ("calibration_weeks", "calibration"),
        ("selection_weeks", "selection"),
        ("forward_weeks", "test"),
    )
    for key, label in assignments:
        weeks = fold[key]
        if not weeks:
            raise ValueError(f"{key} cannot be empty")
        frame.loc[frame.week.isin(weeks), "split"] = label
    for key in ("train_weeks", "calibration_weeks", "selection_weeks", "forward_weeks"):
        selected = set(fold[key])
        if selected.intersection(fold["final_test_weeks"]):
            raise ValueError("walk-forward window overlaps frozen final test")
    counts = frame.groupby("week").split.nunique()
    if (counts != 1).any():
        raise ValueError("split labels are not aligned across series")
    return frame


def _baseline_and_single_window(selection_metrics: pd.DataFrame) -> dict:
    """Return the V0.49 proportional baseline and single-window candidate."""
    selected = _select_policies(selection_metrics)
    return {
        "baseline_config_id": selected["baseline_id"],
        "candidate_config_id": selected["candidate_id"],
        "abstained": not bool(selected["service_floor_feasible"]),
        "feasible_candidate_count": int(
            (
                selection_metrics[
                    ~selection_metrics.config_id.isin(BASELINE_CONFIGS)
                ].mean_fill_rate
                >= selected["service_floor"]
            ).sum()
        ),
        "required_folds": 0,
        "history_folds": 0,
        "service_floor": float(selected["service_floor"]),
        "service_delta_lcb": None,
        "service_delta_sd": None,
        "selection_cost": float(selected["candidate_selection_cost"]),
        "selection_fill_rate": float(selected["candidate_selection_fill_rate"]),
        "selection_source": "current_fold_selection_window",
    }


def _select_historical(history: list[dict], baseline_id: str, mode: str) -> dict:
    """Select from prior forward windows only; abstain when evidence is inadequate."""
    if mode not in ("robust", "conservative"):
        raise ValueError("mode must be robust or conservative")
    n_history = len(history)
    required = int(math.ceil(ROBUST_PASS_FRACTION * n_history)) if n_history else 1
    if mode == "robust" and n_history == 0:
        return _fallback_choice(baseline_id, n_history, required, "no_prior_forward_folds")
    if mode == "conservative" and n_history < 3:
        return _fallback_choice(baseline_id, n_history, 3, "fewer_than_three_prior_folds")

    candidates = []
    for config_id in CANDIDATE_CONFIGS:
        costs, service_deltas = [], []
        valid = True
        for item in history:
            metrics = item["metrics"].set_index("config_id")
            if config_id not in metrics.index or item["baseline_id"] not in metrics.index:
                valid = False
                break
            candidate = metrics.loc[config_id]
            baseline = metrics.loc[item["baseline_id"]]
            costs.append(float(candidate.mean_cost))
            service_deltas.append(float(candidate.mean_fill_rate - baseline.mean_fill_rate))
        if not valid:
            continue
        deltas = np.asarray(service_deltas, dtype=float)
        mean_delta = float(deltas.mean())
        sd_delta = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        if mode == "robust":
            pass_count = int(np.sum(deltas >= -SERVICE_TOLERANCE))
            feasible = pass_count >= required
            lcb = None
        else:
            critical = float(student_t.ppf(CONSERVATIVE_CONFIDENCE, df=n_history - 1))
            lcb = mean_delta - critical * sd_delta / math.sqrt(n_history)
            pass_count = int(np.sum(deltas >= -SERVICE_TOLERANCE))
            feasible = (
                lcb >= -SERVICE_TOLERANCE
                and sd_delta <= CONSERVATIVE_MAX_SERVICE_SD
            )
        if feasible:
            candidates.append({
                "config_id": config_id,
                "mean_historical_cost": float(np.mean(costs)),
                "mean_service_delta": mean_delta,
                "service_delta_sd": sd_delta,
                "service_delta_lcb": lcb,
                "pass_count": pass_count,
            })

    if not candidates:
        return _fallback_choice(
            baseline_id, n_history, required if mode == "robust" else 3,
            "no_candidate_met_historical_service_rule",
        )
    chosen = sorted(candidates, key=lambda x: (x["mean_historical_cost"], x["config_id"]))[0]
    return {
        "baseline_config_id": baseline_id,
        "candidate_config_id": chosen["config_id"],
        "abstained": False,
        "feasible_candidate_count": len(candidates),
        "required_folds": required if mode == "robust" else 3,
        "history_folds": n_history,
        "service_floor": None,
        "service_delta_lcb": chosen["service_delta_lcb"],
        "service_delta_sd": chosen["service_delta_sd"],
        "pass_count": chosen["pass_count"],
        "selection_cost": chosen["mean_historical_cost"],
        "selection_fill_rate": None,
        "selection_source": "prior_walkforward_forward_evaluation_only",
    }


def _fallback_choice(baseline_id: str, n_history: int, required: int, reason: str) -> dict:
    return {
        "baseline_config_id": baseline_id,
        "candidate_config_id": baseline_id,
        "abstained": True,
        "feasible_candidate_count": 0,
        "required_folds": required,
        "history_folds": n_history,
        "service_floor": None,
        "service_delta_lcb": None,
        "service_delta_sd": None,
        "pass_count": 0,
        "selection_cost": None,
        "selection_fill_rate": None,
        "selection_source": reason,
    }


def _aggregate_replicates(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, sort=True)
    result = grouped.agg(
        mean_cost=("mean_cost", "mean"),
        replicate_cost_sd=("mean_cost", "std"),
        mean_fill_rate=("fill_rate", "mean"),
        replicate_fill_sd=("fill_rate", "std"),
        mean_lost_units=("lost_units", "mean"),
        mean_capital_charge_per_week=("mean_capital_charge_per_week", "mean"),
        mean_budget_utilization=("mean_budget_utilization", "mean"),
        max_capacity_utilization=("max_capacity_utilization", "max"),
    ).reset_index()
    return result.fillna({"replicate_cost_sd": 0.0, "replicate_fill_sd": 0.0})


def _calendar_rows(schedule: list[dict], d: pd.DataFrame) -> list[dict]:
    date_map = d.groupby("week").date.first().astype(str).to_dict()
    rows = []
    for fold in schedule:
        row = {"fold_id": fold["fold_id"], "fold_index": fold["fold_index"]}
        for key in ("train_weeks", "calibration_weeks", "selection_weeks", "forward_weeks"):
            weeks = fold[key]
            row[key] = json.dumps(weeks)
            row[key.replace("_weeks", "_date_start")] = date_map[weeks[0]]
            row[key.replace("_weeks", "_date_end")] = date_map[weeks[-1]]
        rows.append(row)
    return rows


def _run_policy_grid(
    d: pd.DataFrame,
    panel: pd.DataFrame,
    scenarios: list[dict],
    replications: int,
    seed_offset: int,
    phase: str,
) -> list[dict]:
    """Run all policy configurations with common random numbers per scenario/replicate."""
    forecast, _ = _fit_forecasts(d)
    item = _item_parameters(panel, d, forecast)
    target_cache = {}
    rows = []
    for dist_name, probabilities in LEAD_DISTRIBUTIONS.items():
        target_cache[dist_name], _ = _fit_stochastic_targets(d, forecast, probabilities)
    for scenario_index, scenario in enumerate(scenarios):
        target = target_cache[scenario["lead_distribution"]]
        probabilities = scenario["lead_probabilities"]
        n_series = len(item["series_ids"])
        n_periods = target.week.nunique()
        for config in CONFIGS:
            for replicate in range(replications):
                seed = SEED_V50 + seed_offset + scenario_index * 100_000 + replicate
                rng = np.random.default_rng(seed)
                draws = rng.choice(
                    np.array([1, 2, 3, 4], dtype=int),
                    size=(n_series, n_periods),
                    p=np.array([probabilities[k] for k in (1, 2, 3, 4)], dtype=float),
                )
                summaries, _, _ = _simulate_v49(
                    target, item, probabilities, scenario, config, draws, False
                )
                for source_split, output_split in (
                    ("selection", "selection"),
                    ("test", "forward_evaluation" if phase == "fold" else "final_test"),
                ):
                    summary = summaries[source_split]
                    rows.append({
                        "phase": phase,
                        "scenario_id": scenario["scenario_id"],
                        "config_id": config["config_id"],
                        "replicate": replicate,
                        "split": output_split,
                        "mean_cost": float(summary["mean_cost"]),
                        "fill_rate": float(summary["fill_rate"]),
                        "lost_units": int(summary["lost_units"]),
                        "mean_capital_charge_per_week": float(summary["mean_capital_charge_per_week"]),
                        "mean_budget_utilization": float(summary["mean_budget_utilization"]),
                        "max_capacity_utilization": float(summary["max_capacity_utilization"]),
                    })
    return rows


def _make_selector_rows(
    phase: str,
    fold_id: str,
    scenario_id: str,
    selection_metrics: pd.DataFrame,
    evaluation_metrics: pd.DataFrame,
    history: list[dict],
) -> list[dict]:
    current = _baseline_and_single_window(selection_metrics)
    baseline_id = current["baseline_config_id"]
    choices = {
        "v49_single_window": current,
        "walkforward_robust_80pct": _select_historical(history, baseline_id, "robust"),
        "walkforward_conservative_lcb": _select_historical(history, baseline_id, "conservative"),
        "baseline_only": {
            "baseline_config_id": baseline_id,
            "candidate_config_id": baseline_id,
            "abstained": False,
            "feasible_candidate_count": 0,
            "required_folds": 0,
            "history_folds": len(history),
            "service_floor": None,
            "service_delta_lcb": None,
            "service_delta_sd": None,
            "selection_cost": None,
            "selection_fill_rate": None,
            "selection_source": "proportional_baseline_only",
        },
    }
    metrics = evaluation_metrics.set_index("config_id")
    baseline_eval = metrics.loc[baseline_id]
    rows = []
    for selector, choice in choices.items():
        selected_id = choice["candidate_config_id"]
        selected_eval = metrics.loc[selected_id]
        rows.append({
            "phase": phase,
            "fold_id": fold_id,
            "scenario_id": scenario_id,
            "selector": selector,
            "baseline_config_id": baseline_id,
            "selected_config_id": selected_id,
            "abstained": bool(choice["abstained"]),
            "selection_source": choice["selection_source"],
            "history_folds": int(choice["history_folds"]),
            "required_folds": int(choice["required_folds"]),
            "feasible_candidate_count": int(choice["feasible_candidate_count"]),
            "service_floor": choice["service_floor"],
            "service_delta_lcb": choice["service_delta_lcb"],
            "service_delta_sd": choice["service_delta_sd"],
            "selected_evaluation_cost": float(selected_eval.mean_cost),
            "baseline_evaluation_cost": float(baseline_eval.mean_cost),
            "cost_delta_vs_baseline": float(selected_eval.mean_cost - baseline_eval.mean_cost),
            "selected_evaluation_fill_rate": float(selected_eval.mean_fill_rate),
            "baseline_evaluation_fill_rate": float(baseline_eval.mean_fill_rate),
            "service_delta_vs_baseline": float(selected_eval.mean_fill_rate - baseline_eval.mean_fill_rate),
        })
    return rows


def run(panel: pd.DataFrame, quality: dict, out: Path, replications: int = 20) -> dict:
    if replications < 2:
        raise ValueError("at least two replications are required")
    d, cuts = features(panel)
    weeks = sorted(d.week.unique().tolist())
    if len(weeks) != 91:
        raise ValueError(f"V0.50 frozen protocol requires 91 feature weeks, got {len(weeks)}")
    schedule = _build_fold_schedule(weeks)
    scenarios = _scenario_grid_v49()
    if len(scenarios) != 32:
        raise AssertionError("unexpected V0.49 scenario grid")

    fold_replicate_rows = []
    fold_policy_frames = []
    fold_selector_rows = []
    history_by_scenario = {x["scenario_id"]: [] for x in scenarios}
    baseline_by_fold = {}
    for fold in schedule:
        fold_d = _assign_fold_splits(d, fold)
        fold_rows = _run_policy_grid(
            fold_d, panel, scenarios, replications,
            seed_offset=fold["fold_index"] * 1_000_000,
            phase="fold",
        )
        fold_replicate_rows.extend(
            [{"fold_id": fold["fold_id"], **row} for row in fold_rows]
        )
        fold_metrics = _aggregate_replicates(
            pd.DataFrame(fold_rows),
            ["scenario_id", "config_id", "split"],
        )
        fold_metrics.insert(0, "fold_id", fold["fold_id"])
        fold_policy_frames.append(fold_metrics)

        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            selection = fold_metrics[
                (fold_metrics.scenario_id == scenario_id)
                & (fold_metrics.split == "selection")
            ]
            forward = fold_metrics[
                (fold_metrics.scenario_id == scenario_id)
                & (fold_metrics.split == "forward_evaluation")
            ]
            baseline = _baseline_and_single_window(selection)["baseline_config_id"]
            history = history_by_scenario[scenario_id]
            fold_selector_rows.extend(_make_selector_rows(
                "fold", fold["fold_id"], scenario_id, selection, forward, history
            ))
            history.append({
                "fold_id": fold["fold_id"],
                "baseline_id": baseline,
                "metrics": forward.copy(),
            })
            baseline_by_fold[(fold["fold_id"], scenario_id)] = baseline

    fold_replicate_frame = pd.DataFrame(fold_replicate_rows)
    fold_policy_frame = pd.concat(fold_policy_frames, ignore_index=True)

    # Final holdout: reproduce the original V0.49 chronological split exactly.
    final_d = d.copy()
    expected_counts = final_d.groupby("split").week.nunique().to_dict()
    if expected_counts != {"calibration": 11, "selection": 10, "test": 16, "train": 54}:
        raise ValueError(f"unexpected final frozen split counts: {expected_counts}")
    final_rows = _run_policy_grid(
        final_d, panel, scenarios, replications,
        seed_offset=9_000_000,
        phase="final",
    )
    final_replicate_frame = pd.DataFrame(final_rows)
    final_policy_frame = _aggregate_replicates(
        final_replicate_frame,
        ["scenario_id", "config_id", "split"],
    )

    final_selector_rows = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        selection = final_policy_frame[
            (final_policy_frame.scenario_id == scenario_id)
            & (final_policy_frame.split == "selection")
        ]
        final_test = final_policy_frame[
            (final_policy_frame.scenario_id == scenario_id)
            & (final_policy_frame.split == "final_test")
        ]
        history = history_by_scenario[scenario_id]
        final_selector_rows.extend(_make_selector_rows(
            "final", "FINAL_TEST", scenario_id, selection, final_test, history
        ))

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(_calendar_rows(schedule, d)).to_csv(out / "fold_calendar.csv", index=False)
    fold_replicate_frame.to_csv(out / "fold_replicate_metrics.csv", index=False)
    fold_policy_frame.to_csv(out / "fold_policy_metrics.csv", index=False)
    pd.DataFrame(fold_selector_rows).to_csv(out / "fold_selector_decisions.csv", index=False)
    final_replicate_frame.to_csv(out / "final_replicate_metrics.csv", index=False)
    final_policy_frame.to_csv(out / "final_policy_metrics.csv", index=False)
    pd.DataFrame(final_selector_rows).to_csv(out / "final_selector_results.csv", index=False)

    final_test_weeks = schedule[0]["final_test_weeks"]
    date_map = d.groupby("week").date.first().astype(str).to_dict()
    manifest = {
        "version": VERSION,
        "status": "research_benchmark",
        "dataset": "UCI Online Retail II prepared product-country weekly recorded-sales panel",
        "panel_rows": int(len(panel)),
        "series_count": int(d.series_id.nunique()),
        "feature_week_count": len(weeks),
        "chronological_split_counts_final_holdout": expected_counts,
        "fold_count": len(schedule),
        "fold_fit_lengths": list(FOLD_FIT_LENGTHS),
        "calibration_length": CALIBRATION_LENGTH,
        "selection_length": SELECTION_LENGTH,
        "forward_evaluation_length": FORWARD_LENGTH,
        "final_test_length": FINAL_TEST_LENGTH,
        "final_test_week_ids": final_test_weeks,
        "final_test_date_start": date_map[final_test_weeks[0]],
        "final_test_date_end": date_map[final_test_weeks[-1]],
        "fold_schedule": [
            {
                "fold_id": x["fold_id"],
                "train_weeks": x["train_weeks"],
                "calibration_weeks": x["calibration_weeks"],
                "selection_weeks": x["selection_weeks"],
                "forward_weeks": x["forward_weeks"],
            } for x in schedule
        ],
        "replications": replications,
        "scenario_count": len(scenarios),
        "policy_config_count": len(CONFIGS),
        "selectors": list(SELECTORS),
        "common_random_numbers": "All eight policies share identical lead-time draws within each scenario, fold and replication.",
        "selector_rules": {
            "v49_single_window": "V0.49 selector applied to the current fold selection window; final test selector uses the original final selection window.",
            "walkforward_robust_80pct": "Select the lowest historical mean-cost non-proportional policy that was within 0.5 percentage points of that fold's proportional baseline in at least ceil(0.8*n) prior forward folds; otherwise fall back to current proportional baseline.",
            "walkforward_conservative_lcb": "Requires at least three prior folds; select the lowest historical mean-cost non-proportional policy whose one-sided 90% t lower confidence bound for paired fill-rate difference is at least -0.5 percentage points and whose across-fold service-delta SD is at most 2 percentage points; otherwise fall back.",
            "baseline_only": "Always use the lower selection-cost q50/q75 proportional baseline.",
        },
        "final_test_firewall": "The final 16 weeks are excluded from all walk-forward fitting, calibration, selector construction and threshold selection. They are evaluated only after the selectors are frozen from the five prior folds.",
        "fit_protocol": "Each fold refits demand models on its train weeks only, calibrates residual/lead-time distributions on its calibration weeks only, and evaluates policy selection on its own selection and subsequent forward windows.",
        "source_hashes": {
            "prepared_panel_sha256": "1d32ccc76a7bb6df0127d0a2d32368a22b79558e33603dfb82afd65eb0293831",
            "quality_manifest_sha256": "5a564a92ac78c8de95511862ea576aaf9aa5784572b8c732d38be78606c19d6f",
        },
        "source_quality": quality,
        "limitations": "Recorded sales may be censored by stockouts. No ground truth for on-hand inventory, latent demand, purchase orders, supplier receipts, acquisition cost, gross margin or financing expense is available. Scenario parameters are assumptions; results do not establish realized profit or cross-retailer generalization.",
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/v50/walkforward"))
    parser.add_argument("--replications", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    _, quality = __import__("inventory_v47").load_prepared_panel(args.panel, args.quality)
    panel = pd.read_csv(args.panel)
    required = {"series_id", "date", "demand", "price"}
    if not required.issubset(panel.columns):
        parser.error(f"prepared panel missing columns: {sorted(required - set(panel.columns))}")
    manifest = run(panel, quality, args.output, args.replications)
    print(json.dumps({
        "version": manifest["version"],
        "folds": manifest["fold_count"],
        "scenarios": manifest["scenario_count"],
        "replications": manifest["replications"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
