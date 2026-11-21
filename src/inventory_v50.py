"""V0.50 walk-forward policy selection under temporal drift.

This module evaluates selectors on five expanding-origin folds before the frozen
16-week holdout. Every fold refits the forecast model and lead-time calibration
using only that fold's training and calibration windows. The frozen test is used
once, after all selector rules are frozen from prior walk-forward evidence.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from inventory_v47 import _fit_forecasts
from inventory_v48 import (
    BOOTSTRAP_REPS,
    LEAD_DISTRIBUTIONS,
    _block_bootstrap_ci,
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
    try:
        if any(float(right) - float(left) != 1.0 for left, right in zip(values, values[1:])):
            raise ValueError("missing or non-consecutive calendar weeks")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("weeks must be numeric consecutive calendar indices") from exc
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


def _replication_seed(phase: str, fold_index: int, scenario_index: int, replicate: int) -> int:
    if phase == "fold":
        offset = int(fold_index) * 1_000_000
    elif phase == "final":
        offset = 9_000_000
    else:
        raise ValueError("phase must be fold or final")
    return SEED_V50 + offset + int(scenario_index) * 100_000 + int(replicate)


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


def _window_panel(
    panel: pd.DataFrame,
    train_end_date,
    evaluation_end_date,
    global_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a no-lookahead cohort panel ending at the evaluation horizon.

    Only series observed by the training cutoff are included. Actual records after
    the evaluation horizon are removed; one zero-demand calendar anchor after the
    horizon preserves the feature builder's final-week exclusion convention.
    """
    source = panel.copy()
    source["date"] = pd.to_datetime(source.date, errors="raise")
    train_end = pd.Timestamp(train_end_date)
    evaluation_end = pd.Timestamp(evaluation_end_date)
    known_ids = source.loc[source.date <= train_end, "series_id"].dropna().unique()
    if len(known_ids) == 0:
        raise ValueError("no series are known by this fold's training cutoff")
    window = source[
        source.series_id.isin(known_ids) & (source.date <= evaluation_end)
    ].copy()
    anchor_id = known_ids[0]

    def placeholder(date_value):
        row = {column: np.nan for column in source.columns}
        row["series_id"] = anchor_id
        row["date"] = pd.Timestamp(date_value)
        row["demand"] = 0.0
        return row

    global_start = pd.Timestamp(global_dates[0])
    pad_date = evaluation_end + pd.Timedelta(days=7)
    present_dates = set(pd.to_datetime(window.date).unique())
    additions = []
    if np.datetime64(global_start) not in present_dates:
        additions.append(placeholder(global_start))
    if np.datetime64(pad_date) not in present_dates:
        additions.append(placeholder(pad_date))
    if additions:
        window = pd.concat([window, pd.DataFrame(additions)], ignore_index=True)
    if window.date.max() != pad_date:
        raise AssertionError("window panel does not end at the one-week calendar pad")
    if (pd.to_datetime(window.date) > pad_date).any():
        raise AssertionError("future actual rows leaked into fold panel")
    return window.sort_values(["series_id", "date"]).reset_index(drop=True)


def _row_membership_hash(frame: pd.DataFrame) -> str:
    ordered = frame[["series_id", "week"]].sort_values(
        ["series_id", "week"], kind="mergesort"
    )
    payload = "\n".join(
        f"{row.series_id}\t{int(row.week)}" for row in ordered.itertuples(index=False)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partition_rows(frame: pd.DataFrame, fold_id: str, labels: dict[str, str]) -> list[dict]:
    rows = []
    for split, partition in labels.items():
        subset = frame[frame.split == split]
        for row in subset[["series_id", "week"]].itertuples(index=False):
            rows.append({
                "fold_id": fold_id,
                "series_id": str(row.series_id),
                "week": int(row.week),
                "partition": partition,
            })
    return rows


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
    fold_index: int,
    phase: str,
) -> tuple[list[dict], list[dict]]:
    """Run policy configurations with common random numbers and retain weekly evidence."""
    forecast, _ = _fit_forecasts(d)
    item = _item_parameters(panel, d, forecast)
    target_cache = {}
    rows, weekly_rows = [], []
    for dist_name, probabilities in LEAD_DISTRIBUTIONS.items():
        target_cache[dist_name], _ = _fit_stochastic_targets(d, forecast, probabilities)
    for scenario_index, scenario in enumerate(scenarios):
        target = target_cache[scenario["lead_distribution"]]
        probabilities = scenario["lead_probabilities"]
        n_series = len(item["series_ids"])
        n_periods = target.week.nunique()
        for config in CONFIGS:
            for replicate in range(replications):
                seed = _replication_seed(phase, fold_index, scenario_index, replicate)
                rng = np.random.default_rng(seed)
                draws = rng.choice(
                    np.array([1, 2, 3, 4], dtype=int),
                    size=(n_series, n_periods),
                    p=np.array([probabilities[k] for k in (1, 2, 3, 4)], dtype=float),
                )
                summaries, weekly, _ = _simulate_v49(
                    target, item, probabilities, scenario, config, draws, False
                )
                for week_row in weekly:
                    if week_row["split"] not in ("selection", "test"):
                        continue
                    output_split = (
                        "forward_evaluation" if phase == "fold" and week_row["split"] == "test"
                        else "final_test" if phase == "final" and week_row["split"] == "test"
                        else "selection"
                    )
                    weekly_rows.append({
                        "phase": phase,
                        "scenario_id": scenario["scenario_id"],
                        "config_id": config["config_id"],
                        "replicate": replicate,
                        "week": int(week_row["week"]),
                        "split": output_split,
                        "mean_total_cost": float(week_row["mean_total_cost"]),
                        "fill_rate": float(week_row["fill_rate"]),
                        "lost_units": int(week_row["lost_units"]),
                    })
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
    return rows, weekly_rows


def _make_selector_rows(
    phase: str,
    fold_id: str,
    scenario_id: str,
    selection_metrics: pd.DataFrame,
    evaluation_metrics: pd.DataFrame,
    history: list[dict],
    weekly_metrics: pd.DataFrame,
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
            "history_folds": 0,
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
    evaluation_split = "forward_evaluation" if phase == "fold" else "final_test"
    scenario_weekly = weekly_metrics[
        (weekly_metrics.scenario_id == scenario_id)
        & (weekly_metrics.split == evaluation_split)
    ]
    baseline_weekly = (
        scenario_weekly[scenario_weekly.config_id == baseline_id]
        .groupby("week").agg(
            mean_total_cost=("mean_total_cost", "mean"),
            fill_rate=("fill_rate", "mean"),
        )
    )
    rows = []
    for selector, choice in choices.items():
        selected_id = choice["candidate_config_id"]
        selected_eval = metrics.loc[selected_id]
        selected_weekly = (
            scenario_weekly[scenario_weekly.config_id == selected_id]
            .groupby("week").agg(
                mean_total_cost=("mean_total_cost", "mean"),
                fill_rate=("fill_rate", "mean"),
            )
        )
        paired = selected_weekly.join(
            baseline_weekly, how="inner", lsuffix="_selected", rsuffix="_baseline"
        )
        if len(paired) < 4:
            raise AssertionError("insufficient weekly observations for block bootstrap")
        cost_delta = (
            paired.mean_total_cost_selected - paired.mean_total_cost_baseline
        ).to_numpy(float)
        service_delta = (paired.fill_rate_selected - paired.fill_rate_baseline).to_numpy(float)
        token = f"{phase}|{fold_id}|{scenario_id}|{selector}".encode("utf-8")
        bootstrap_seed = SEED_V50 + int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
        cost_ci = _block_bootstrap_ci(
            cost_delta, bootstrap_seed, reps=BOOTSTRAP_REPS, block=4
        )
        service_ci = _block_bootstrap_ci(
            service_delta, bootstrap_seed + 1, reps=BOOTSTRAP_REPS, block=4
        )
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
            "cost_delta_block_bootstrap_95_ci": json.dumps(cost_ci),
            "worst_week_cost_delta": float(cost_delta.max()),
            "selected_evaluation_fill_rate": float(selected_eval.mean_fill_rate),
            "baseline_evaluation_fill_rate": float(baseline_eval.mean_fill_rate),
            "service_delta_vs_baseline": float(selected_eval.mean_fill_rate - baseline_eval.mean_fill_rate),
            "service_delta_block_bootstrap_95_ci": json.dumps(service_ci),
            "worst_week_service_delta": float(service_delta.min()),
        })
    return rows


def _selector_summary(fold_decisions: pd.DataFrame, final_decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, frame in (("walkforward", fold_decisions), ("final_test", final_decisions)):
        for selector, group in frame.groupby("selector", sort=True):
            rows.append({
                "phase": phase,
                "selector": selector,
                "scenario_fold_count": int(len(group)),
                "abstention_rate": float(group.abstained.mean()),
                "mean_cost_delta_vs_baseline": float(group.cost_delta_vs_baseline.mean()),
                "worst_cost_regret_vs_baseline": float(group.cost_delta_vs_baseline.max()),
                "cost_win_count": int((group.cost_delta_vs_baseline < 0).sum()),
                "mean_service_delta_vs_baseline": float(group.service_delta_vs_baseline.mean()),
                "worst_service_degradation": float(group.service_delta_vs_baseline.min()),
                "service_noninferiority_rate": float(
                    (group.service_delta_vs_baseline >= -SERVICE_TOLERANCE).mean()
                ),
            })
    return pd.DataFrame(rows)


def run(panel: pd.DataFrame, quality: dict, out: Path, replications: int = 20) -> dict:
    if replications < 2:
        raise ValueError("at least two replications are required")
    out = Path(out)
    if out.exists():
        raise ValueError("choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=False)

    source = panel.copy()
    source_dates = pd.to_datetime(source.date, errors="raise")
    if (source_dates.dt.dayofweek != 6).any():
        raise ValueError("prepared panel dates must be Sundays")
    global_dates = pd.date_range(source_dates.min(), source_dates.max(), freq="W-SUN")
    if global_dates[0] != source_dates.min() or global_dates[-1] != source_dates.max():
        raise ValueError("prepared panel endpoints do not align to the weekly calendar")
    # retail_v46.features maps original calendar index j to feature week j-1,
    # removes the first/last partial weeks, and then drops the first 13 lags.
    weeks = list(range(13, len(global_dates) - 2))
    if len(weeks) != 91:
        raise ValueError(f"V0.50 frozen protocol requires 91 feature weeks, got {len(weeks)}")
    date_map = {week: pd.Timestamp(global_dates[week + 1]) for week in weeks}
    calendar_frame = pd.DataFrame({
        "week": weeks,
        "date": [date_map[week] for week in weeks],
    })
    schedule = _build_fold_schedule(weeks)
    scenarios = _scenario_grid_v49()
    if len(scenarios) != 32:
        raise AssertionError("unexpected V0.49 scenario grid")

    # Prepare every fold using only series known at its training cutoff and actual
    # records no later than its forward-evaluation horizon. The sole later row is
    # a synthetic zero-demand calendar pad needed by the shared feature builder.
    fold_data = []
    fold_partition_rows = []
    preflight_fold_partitions = {}
    for fold in schedule:
        train_end_date = date_map[fold["train_weeks"][-1]]
        forward_end_date = date_map[fold["forward_weeks"][-1]]
        fold_panel = _window_panel(
            panel, train_end_date, forward_end_date, global_dates
        )
        fold_features, _ = features(fold_panel)
        fold_d = _assign_fold_splits(fold_features, fold)
        if set(fold_d.week.unique()) != set(
            fold["train_weeks"] + fold["calibration_weeks"]
            + fold["selection_weeks"] + fold["forward_weeks"]
        ):
            raise AssertionError(f"{fold['fold_id']} feature calendar does not match its windows")
        fold_data.append((fold, fold_panel, fold_d))
        labels = {
            "train": "train",
            "calibration": "calibration",
            "selection": "selection",
            "test": "forward_evaluation",
        }
        partition_stats = {}
        for split, partition in labels.items():
            subset = fold_d[fold_d.split == split]
            partition_stats[partition] = {
                "rows": int(len(subset)),
                "sha256": _row_membership_hash(subset),
            }
        excluded = pd.DataFrame([
            {"series_id": sid, "week": week}
            for sid in sorted(fold_d.series_id.unique())
            for week in fold["final_test_weeks"]
        ])
        partition_stats["excluded_final_test"] = {
            "rows": int(len(excluded)),
            "sha256": _row_membership_hash(excluded),
        }
        preflight_fold_partitions[fold["fold_id"]] = {
            "series_count_known_at_train_cutoff": int(fold_d.series_id.nunique()),
            "train_end_date": str(train_end_date.date()),
            "forward_end_date": str(forward_end_date.date()),
            "partitions": partition_stats,
        }
        fold_partition_rows.extend(_partition_rows(fold_d, fold["fold_id"], labels))
        for row in excluded.itertuples(index=False):
            fold_partition_rows.append({
                "fold_id": fold["fold_id"],
                "series_id": str(row.series_id),
                "week": int(row.week),
                "partition": "excluded_final_test",
            })

    # The final cohort is also frozen using series known by the final training
    # cutoff. Final-test outcomes are retained solely for the last evaluation.
    final_train_end_week = weeks[53]
    final_panel = _window_panel(
        panel,
        date_map[final_train_end_week],
        date_map[weeks[-1]],
        global_dates,
    )
    final_d, _ = features(final_panel)
    expected_counts = final_d.groupby("split").week.nunique().to_dict()
    if expected_counts != {"calibration": 11, "selection": 10, "test": 16, "train": 54}:
        raise ValueError(f"unexpected final frozen split counts: {expected_counts}")
    final_partition_labels = {
        "train": "train",
        "calibration": "calibration",
        "selection": "selection",
        "test": "final_test",
    }
    final_partition_rows = _partition_rows(final_d, "FINAL", final_partition_labels)
    final_partition_stats = {}
    for split, partition in final_partition_labels.items():
        subset = final_d[final_d.split == split]
        final_partition_stats[partition] = {
            "rows": int(len(subset)),
            "sha256": _row_membership_hash(subset),
        }

    pd.DataFrame(_calendar_rows(schedule, calendar_frame)).to_csv(
        out / "fold_calendar.csv", index=False
    )
    pd.DataFrame(fold_partition_rows).to_csv(
        out / "fold_row_partitions.csv", index=False
    )
    pd.DataFrame(final_partition_rows).to_csv(
        out / "final_row_partitions.csv", index=False
    )
    preflight = {
        "version": VERSION,
        "status": "preflight_passed_before_benchmark",
        "feature_week_ids": weeks,
        "final_test_week_ids": schedule[0]["final_test_weeks"],
        "final_test_date_start": str(date_map[weeks[-16]].date()),
        "final_test_date_end": str(date_map[weeks[-1]].date()),
        "source_hashes": {
            "prepared_panel_sha256": "1d32ccc76a7bb6df0127d0a2d32368a22b79558e33603dfb82afd65eb0293831",
            "quality_manifest_sha256": "5a564a92ac78c8de95511862ea576aaf9aa5784572b8c732d38be78606c19d6f",
        },
        "fold_schedule": [
            {
                "fold_id": x["fold_id"],
                "train_weeks": x["train_weeks"],
                "calibration_weeks": x["calibration_weeks"],
                "selection_weeks": x["selection_weeks"],
                "forward_weeks": x["forward_weeks"],
            } for x in schedule
        ],
        "fold_row_partitions": preflight_fold_partitions,
        "final_row_partitions": final_partition_stats,
        "cohort_rule": "Only series with at least one source-panel row on or before that fold's training cutoff are included. Actual records after the evaluation horizon are removed before feature construction.",
        "final_test_firewall": "The final 16 weeks are not present in any fold feature panel, training, calibration, selection or forward evaluation input. They are present only in the separate final evaluation panel and are never used to choose a selector.",
    }
    preflight_path = out / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    preflight_sha256 = hashlib.sha256(preflight_path.read_bytes()).hexdigest()

    fold_replicate_rows = []
    fold_weekly_frames = []
    fold_policy_frames = []
    fold_selector_rows = []
    history_by_scenario = {x["scenario_id"]: [] for x in scenarios}
    for fold, fold_panel, fold_d in fold_data:
        fold_rows, fold_weekly_rows = _run_policy_grid(
            fold_d, fold_panel, scenarios, replications,
            fold_index=fold["fold_index"],
            phase="fold",
        )
        fold_replicate_rows.extend(
            [{"fold_id": fold["fold_id"], **row} for row in fold_rows]
        )
        fold_weekly_frame = pd.DataFrame(fold_weekly_rows)
        fold_weekly_frame.insert(0, "fold_id", fold["fold_id"])
        fold_weekly_frames.append(fold_weekly_frame)
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
                "fold", fold["fold_id"], scenario_id, selection, forward, history,
                fold_weekly_frame,
            ))
            history.append({
                "fold_id": fold["fold_id"],
                "baseline_id": baseline,
                "metrics": forward.copy(),
            })

    fold_replicate_frame = pd.DataFrame(fold_replicate_rows)
    fold_weekly_frame = pd.concat(fold_weekly_frames, ignore_index=True)
    fold_policy_frame = pd.concat(fold_policy_frames, ignore_index=True)

    final_rows, final_weekly_rows = _run_policy_grid(
        final_d, final_panel, scenarios, replications,
        fold_index=0,
        phase="final",
    )
    final_replicate_frame = pd.DataFrame(final_rows)
    final_weekly_frame = pd.DataFrame(final_weekly_rows)
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
            "final", "FINAL_TEST", scenario_id, selection, final_test, history,
            final_weekly_frame,
        ))

    fold_decision_frame = pd.DataFrame(fold_selector_rows)
    final_decision_frame = pd.DataFrame(final_selector_rows)
    selector_summary = _selector_summary(fold_decision_frame, final_decision_frame)

    fold_replicate_frame.to_csv(out / "fold_replicate_metrics.csv", index=False)
    fold_weekly_frame.to_csv(out / "fold_weekly_metrics.csv", index=False)
    fold_policy_frame.to_csv(out / "fold_policy_metrics.csv", index=False)
    fold_decision_frame.to_csv(out / "fold_selector_decisions.csv", index=False)
    final_replicate_frame.to_csv(out / "final_replicate_metrics.csv", index=False)
    final_weekly_frame.to_csv(out / "final_weekly_metrics.csv", index=False)
    final_policy_frame.to_csv(out / "final_policy_metrics.csv", index=False)
    final_decision_frame.to_csv(out / "final_selector_results.csv", index=False)
    selector_summary.to_csv(out / "selector_summary.csv", index=False)

    final_test_weeks = schedule[0]["final_test_weeks"]
    manifest = {
        "version": VERSION,
        "status": "research_benchmark",
        "dataset": "UCI Online Retail II prepared product-country weekly recorded-sales panel",
        "panel_rows": int(len(panel)),
        "series_count": int(final_d.series_id.nunique()),
        "full_source_series_count": int(panel.series_id.nunique()),
        "final_training_cohort_coverage": float(final_d.series_id.nunique() / panel.series_id.nunique()),
        "feature_week_count": len(weeks),
        "chronological_split_counts_final_holdout": expected_counts,
        "fold_count": len(schedule),
        "fold_fit_lengths": list(FOLD_FIT_LENGTHS),
        "calibration_length": CALIBRATION_LENGTH,
        "selection_length": SELECTION_LENGTH,
        "forward_evaluation_length": FORWARD_LENGTH,
        "final_test_length": FINAL_TEST_LENGTH,
        "final_test_week_ids": final_test_weeks,
        "final_test_date_start": str(date_map[final_test_weeks[0]].date()),
        "final_test_date_end": str(date_map[final_test_weeks[-1]].date()),
        "fold_schedule": preflight["fold_schedule"],
        "preflight_manifest_sha256": preflight_sha256,
        "preflight_written_before_benchmark": True,
        "row_level_partition_audit": True,
        "replications": replications,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_block_length_weeks": 4,
        "scenario_count": len(scenarios),
        "policy_config_count": len(CONFIGS),
        "selectors": list(SELECTORS),
        "common_random_numbers": "All eight policies share identical lead-time draws within each scenario, fold and replication.",
        "selector_rules": {
            "v49_single_window": "V0.49 selector applied to the current fold selection window; final test selector uses the original final selection window.",
            "walkforward_robust_80pct": "Select the lowest historical mean-cost non-proportional policy that was within 0.5 percentage points of that fold's proportional baseline in at least ceil(0.8*n) prior forward folds; final holdout uses four of five prior folds or more; otherwise fall back.",
            "walkforward_conservative_lcb": "Requires at least three prior folds; select the lowest historical mean-cost non-proportional policy whose one-sided 90% t lower confidence bound for paired fill-rate difference is at least -0.5 percentage points and whose across-fold service-delta SD is at most 2 percentage points; otherwise fall back.",
            "baseline_only": "Always use the lower selection-cost q50/q75 proportional baseline.",
        },
        "final_test_firewall": "The final 16 weeks are excluded from all walk-forward feature panels, model fitting, residual calibration, selector construction and threshold selection. They are evaluated only after the selectors are frozen from the five prior folds.",
        "cohort_rule": "For each fold, only series observed by that fold's training cutoff are included; actual records after the fold forward-evaluation horizon are removed before feature construction. Final-test cohort is frozen at the final training cutoff.",
        "fit_protocol": "Each fold refits demand models on its train weeks only, calibrates residual/lead-time distributions on its calibration weeks only, and evaluates policy selection on its own selection and subsequent forward windows.",
        "uncertainty": "Descriptive circular four-week block bootstrap over weekly paired policy-minus-baseline deltas, with 2,000 resamples; intervals are not guarantees and have limited temporal scope.",
        "source_hashes": preflight["source_hashes"],
        "source_quality": quality,
        "limitations": "Recorded sales may be censored by stockouts. No ground truth for on-hand inventory, latent demand, purchase orders, supplier receipts, acquisition cost, gross margin or financing expense is available. Scenario parameters are assumptions; results do not establish realized profit or cross-retailer generalization. Series entering after a fold's training cutoff are excluded from that fold's cohort.",
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
