"""Independent audit of V0.50 walk-forward selection and holdout firewall."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

BASELINE_CONFIGS = ("q50_proportional", "q75_proportional")
CANDIDATE_CONFIGS = (
    "q50_avg_greedy", "q75_avg_greedy", "q90_avg_greedy",
    "q50_marginal_lot", "q75_marginal_lot", "q90_marginal_lot",
)
TOLERANCE = 0.005
ROBUST_FRACTION = 0.80
ROBUST_MIN_HISTORY_FOLDS = 3
CONFIDENCE = 0.90
MAX_SERVICE_SD = 0.02
SELECTORS = (
    "v49_single_window",
    "walkforward_robust_80pct",
    "walkforward_conservative_lcb",
    "baseline_only",
)


def _baseline(selection: pd.DataFrame) -> tuple[str, str, int]:
    base = selection[selection.config_id.isin(BASELINE_CONFIGS)].sort_values(
        ["mean_cost", "config_id"]
    )
    if len(base) != 2:
        raise AssertionError("missing proportional baseline policies")
    baseline_id = str(base.iloc[0].config_id)
    floor = max(0.0, float(base.iloc[0].mean_fill_rate) - TOLERANCE)
    candidates = selection[~selection.config_id.isin(BASELINE_CONFIGS)]
    feasible = candidates[candidates.mean_fill_rate >= floor].sort_values(
        ["mean_cost", "config_id"]
    )
    selected = str(feasible.iloc[0].config_id) if len(feasible) else baseline_id
    return baseline_id, selected, int(len(feasible))


def _historical_choice(
    history: list[tuple[str, str, pd.DataFrame]],
    baseline_id: str,
    mode: str,
) -> tuple[str, bool, int]:
    n = len(history)
    required = max(ROBUST_MIN_HISTORY_FOLDS, int(math.ceil(ROBUST_FRACTION * n))) if n else ROBUST_MIN_HISTORY_FOLDS
    if mode == "robust" and n < ROBUST_MIN_HISTORY_FOLDS:
        return baseline_id, True, 0
    if mode == "conservative" and n < 3:
        return baseline_id, True, 0
    candidates = []
    for config_id in CANDIDATE_CONFIGS:
        costs, deltas = [], []
        for fold_id, fold_baseline, frame in history:
            m = frame.set_index("config_id")
            if config_id not in m.index or fold_baseline not in m.index:
                break
            costs.append(float(m.loc[config_id, "mean_cost"]))
            deltas.append(
                float(m.loc[config_id, "mean_fill_rate"] - m.loc[fold_baseline, "mean_fill_rate"])
            )
        if len(deltas) != n:
            continue
        values = np.asarray(deltas, dtype=float)
        sd = float(values.std(ddof=1)) if n > 1 else 0.0
        if mode == "robust":
            passes = int(np.sum(values >= -TOLERANCE))
            feasible = passes >= required
        else:
            critical = float(student_t.ppf(CONFIDENCE, df=n - 1))
            lcb = float(values.mean() - critical * sd / math.sqrt(n))
            feasible = lcb >= -TOLERANCE and sd <= MAX_SERVICE_SD
        if feasible:
            candidates.append((float(np.mean(costs)), config_id))
    if not candidates:
        return baseline_id, True, 0
    return sorted(candidates, key=lambda x: (x[0], x[1]))[0][1], False, len(candidates)


def _history_for_scenario(
    fold_metrics: pd.DataFrame,
    scenario_id: str,
    before_fold_index: int,
) -> list[tuple[str, str, pd.DataFrame]]:
    history = []
    for fold_index in range(1, before_fold_index):
        fold_id = f"F{fold_index}"
        current = fold_metrics[
            (fold_metrics.fold_id == fold_id)
            & (fold_metrics.scenario_id == scenario_id)
        ]
        selection = current[current.split == "selection"]
        baseline_id, _, _ = _baseline(selection)
        forward = current[current.split == "forward_evaluation"]
        history.append((fold_id, baseline_id, forward))
    return history


def _expected_selector_rows(
    metrics: pd.DataFrame,
    history: list[tuple[str, str, pd.DataFrame]],
) -> dict[str, tuple[str, bool, int]]:
    baseline_id, single_id, single_feasible = _baseline(
        metrics[metrics.split == "selection"]
    )
    robust_id, robust_abstained, robust_count = _historical_choice(
        history, baseline_id, "robust"
    )
    conservative_id, conservative_abstained, conservative_count = _historical_choice(
        history, baseline_id, "conservative"
    )
    return {
        "v49_single_window": (single_id, single_id == baseline_id and single_feasible == 0, single_feasible),
        "walkforward_robust_80pct": (robust_id, robust_abstained, robust_count),
        "walkforward_conservative_lcb": (conservative_id, conservative_abstained, conservative_count),
        "baseline_only": (baseline_id, False, 0),
    }


def _assert_aggregate(replicates: pd.DataFrame, metrics: pd.DataFrame, keys: list[str], phase: str) -> None:
    grouped = replicates.groupby(keys).agg(
        expected_cost=("mean_cost", "mean"),
        expected_fill=("fill_rate", "mean"),
    ).reset_index()
    observed = metrics.merge(grouped, on=keys, how="left", validate="one_to_one")
    if observed[["expected_cost", "expected_fill"]].isna().any().any():
        raise AssertionError(f"{phase} replicate aggregation has missing groups")
    if not np.allclose(observed.mean_cost, observed.expected_cost, atol=1e-9):
        raise AssertionError(f"{phase} mean cost does not reproduce from replications")
    if not np.allclose(observed.mean_fill_rate, observed.expected_fill, atol=1e-10):
        raise AssertionError(f"{phase} fill rate does not reproduce from replications")


def _membership_hash(frame: pd.DataFrame) -> str:
    ordered = frame[["series_id", "week"]].sort_values(
        ["series_id", "week"], kind="mergesort"
    )
    payload = "\n".join(
        f"{row.series_id}\t{int(row.week)}" for row in ordered.itertuples(index=False)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _block_bootstrap_ci(values: np.ndarray, seed: int, reps: int, block: int = 4) -> list[float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < block or not np.isfinite(values).all():
        raise AssertionError("invalid weekly values for block bootstrap audit")
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, size=(reps, int(math.ceil(n / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]
    return [float(x) for x in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def _audit_membership_rows(
    preflight_stats: dict,
    observed: pd.DataFrame,
    group_columns: list[str],
    partition_name: str,
) -> None:
    subset = observed[observed.partition == partition_name]
    expected_rows = int(preflight_stats["rows"])
    if len(subset) != expected_rows:
        raise AssertionError(f"row partition count mismatch: {partition_name}")
    if _membership_hash(subset) != preflight_stats["sha256"]:
        raise AssertionError(f"row partition hash mismatch: {partition_name}")


def audit(directory: str | Path) -> dict:
    path = Path(directory)
    manifest_path = path / "manifest.json"
    preflight_path = path / "preflight.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    calendar = pd.read_csv(path / "fold_calendar.csv")
    fold_partitions = pd.read_csv(path / "fold_row_partitions.csv")
    final_partitions = pd.read_csv(path / "final_row_partitions.csv")
    fold_reps = pd.read_csv(path / "fold_replicate_metrics.csv")
    fold_weekly = pd.read_csv(path / "fold_weekly_metrics.csv")
    fold_metrics = pd.read_csv(path / "fold_policy_metrics.csv")
    fold_decisions = pd.read_csv(path / "fold_selector_decisions.csv")
    final_reps = pd.read_csv(path / "final_replicate_metrics.csv")
    final_weekly = pd.read_csv(path / "final_weekly_metrics.csv")
    final_metrics = pd.read_csv(path / "final_policy_metrics.csv")
    final_decisions = pd.read_csv(path / "final_selector_results.csv")
    selector_summary = pd.read_csv(path / "selector_summary.csv")

    if manifest.get("version") != "0.50" or preflight.get("version") != "0.50":
        raise AssertionError("wrong benchmark/preflight version")
    if preflight.get("status") != "preflight_passed_before_benchmark":
        raise AssertionError("preflight was not recorded before the benchmark")
    preflight_sha = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    if manifest.get("preflight_manifest_sha256") != preflight_sha:
        raise AssertionError("preflight manifest hash mismatch")
    if manifest.get("source_hashes") != preflight.get("source_hashes"):
        raise AssertionError("source hashes changed after preflight")
    replications = int(manifest["replications"])
    scenario_count = int(manifest["scenario_count"])
    config_count = int(manifest["policy_config_count"])
    bootstrap_reps = int(manifest["bootstrap_reps"])
    bootstrap_block = int(manifest["bootstrap_block_length_weeks"])
    if len(calendar) != 5 or manifest.get("fold_count") != 5:
        raise AssertionError("expected five walk-forward folds")
    if scenario_count != 32 or config_count != 8:
        raise AssertionError("unexpected scenario/configuration grid")
    if manifest.get("selectors") != list(SELECTORS):
        raise AssertionError("selector registry mismatch")
    if int(manifest.get("robust_min_history_folds", -1)) != ROBUST_MIN_HISTORY_FOLDS:
        raise AssertionError("robust selector minimum history length mismatch")
    final_test = set(manifest["final_test_week_ids"])
    if len(final_test) != 16 or final_test != set(preflight["final_test_week_ids"]):
        raise AssertionError("final holdout must contain the same 16 preflighted weeks")

    schedule_by_id = {x["fold_id"]: x for x in preflight["fold_schedule"]}
    for row in calendar.itertuples(index=False):
        windows = {
            "train": json.loads(row.train_weeks),
            "calibration": json.loads(row.calibration_weeks),
            "selection": json.loads(row.selection_weeks),
            "forward": json.loads(row.forward_weeks),
        }
        if row.fold_id not in schedule_by_id:
            raise AssertionError("fold calendar and preflight schedule differ")
        expected = schedule_by_id[row.fold_id]
        for key in ("train", "calibration", "selection", "forward"):
            if windows[key] != expected[f"{key}_weeks"]:
                raise AssertionError(f"{row.fold_id} calendar differs from preflight")
        flattened = [w for values in windows.values() for w in values]
        if len(flattened) != len(set(flattened)):
            raise AssertionError(f"{row.fold_id} has overlapping windows")
        if set(flattened).intersection(final_test):
            raise AssertionError(f"{row.fold_id} overlaps frozen final test")
        if [len(windows[k]) for k in ("train", "calibration", "selection", "forward")] != [
            30 + 6 * (int(row.fold_index) - 1), 8, 6, 6
        ]:
            raise AssertionError(f"{row.fold_id} window lengths differ from protocol")
        for key, values in windows.items():
            if any(int(b) - int(a) != 1 for a, b in zip(values, values[1:])):
                raise AssertionError(f"{row.fold_id} {key} contains missing calendar weeks")
        if max(windows["train"]) >= min(windows["calibration"]):
            raise AssertionError(f"{row.fold_id} train/calibration ordering invalid")
        if max(windows["calibration"]) >= min(windows["selection"]):
            raise AssertionError(f"{row.fold_id} calibration/selection ordering invalid")
        if max(windows["selection"]) >= min(windows["forward"]):
            raise AssertionError(f"{row.fold_id} selection/forward ordering invalid")

        # Verify exact row-level partition membership hashes written before execution.
        stats = preflight["fold_row_partitions"][row.fold_id]["partitions"]
        for partition in ("train", "calibration", "selection", "forward_evaluation", "excluded_final_test"):
            subset = fold_partitions[
                (fold_partitions.fold_id == row.fold_id)
                & (fold_partitions.partition == partition)
            ]
            if len(subset) != int(stats[partition]["rows"]):
                raise AssertionError(f"{row.fold_id} row count mismatch for {partition}")
            if _membership_hash(subset) != stats[partition]["sha256"]:
                raise AssertionError(f"{row.fold_id} row hash mismatch for {partition}")
            if partition != "excluded_final_test" and set(subset.week.unique()).intersection(final_test):
                raise AssertionError(f"final test row entered {row.fold_id} {partition}")
        excluded = fold_partitions[
            (fold_partitions.fold_id == row.fold_id)
            & (fold_partitions.partition == "excluded_final_test")
        ]
        if set(excluded.week.unique()) != final_test:
            raise AssertionError(f"{row.fold_id} final-test exclusion manifest incomplete")

    for partition in ("train", "calibration", "selection", "final_test"):
        subset = final_partitions[final_partitions.partition == partition]
        stats = preflight["final_row_partitions"][partition]
        if len(subset) != int(stats["rows"]) or _membership_hash(subset) != stats["sha256"]:
            raise AssertionError(f"final row partition mismatch: {partition}")
    if set(final_partitions.loc[final_partitions.partition == "final_test", "week"].unique()) != final_test:
        raise AssertionError("final evaluation rows do not match frozen holdout weeks")
    if fold_partitions.duplicated(["fold_id", "series_id", "week", "partition"]).any():
        raise AssertionError("duplicate fold row-partition records")
    if final_partitions.duplicated(["series_id", "week", "partition"]).any():
        raise AssertionError("duplicate final row-partition records")

    expected_fold_reps = 5 * scenario_count * config_count * replications * 2
    expected_fold_metrics = 5 * scenario_count * config_count * 2
    expected_final_reps = scenario_count * config_count * replications * 2
    expected_final_metrics = scenario_count * config_count * 2
    expected_fold_weekly = 5 * scenario_count * config_count * replications * 12
    expected_final_weekly = scenario_count * config_count * replications * 26
    if len(fold_reps) != expected_fold_reps or len(fold_metrics) != expected_fold_metrics:
        raise AssertionError("walk-forward metric row counts do not match protocol")
    if len(final_reps) != expected_final_reps or len(final_metrics) != expected_final_metrics:
        raise AssertionError("final metric row counts do not match protocol")
    if len(fold_weekly) != expected_fold_weekly or len(final_weekly) != expected_final_weekly:
        raise AssertionError("weekly evidence row counts do not match protocol")
    if len(fold_decisions) != 5 * scenario_count * len(SELECTORS):
        raise AssertionError("fold selector decision row count mismatch")
    if len(final_decisions) != scenario_count * len(SELECTORS):
        raise AssertionError("final selector decision row count mismatch")
    if len(selector_summary) != 2 * len(SELECTORS):
        raise AssertionError("selector summary row count mismatch")

    if fold_reps.duplicated(["fold_id", "scenario_id", "config_id", "replicate", "split"]).any():
        raise AssertionError("duplicate walk-forward replicate rows")
    if final_reps.duplicated(["scenario_id", "config_id", "replicate", "split"]).any():
        raise AssertionError("duplicate final replicate rows")
    if fold_weekly.duplicated(["fold_id", "scenario_id", "config_id", "replicate", "week"]).any():
        raise AssertionError("duplicate walk-forward weekly rows")
    if final_weekly.duplicated(["scenario_id", "config_id", "replicate", "week"]).any():
        raise AssertionError("duplicate final weekly rows")
    if set(fold_reps.replicate.unique()) != set(range(replications)):
        raise AssertionError("walk-forward replicate indices incomplete")
    if set(final_reps.replicate.unique()) != set(range(replications)):
        raise AssertionError("final replicate indices incomplete")

    _assert_aggregate(
        fold_reps, fold_metrics,
        ["fold_id", "scenario_id", "config_id", "split"], "walk-forward"
    )
    _assert_aggregate(
        final_reps, final_metrics,
        ["scenario_id", "config_id", "split"], "final"
    )

    # Recompute selector decisions and weekly block-bootstrap intervals.
    for frame, policy_metrics, weekly_metrics, phase in (
        (fold_decisions, fold_metrics, fold_weekly, "fold"),
        (final_decisions, final_metrics, final_weekly, "final"),
    ):
        for row in frame.itertuples(index=False):
            if phase == "fold":
                fold_index = int(str(row.fold_id)[1:])
                current = policy_metrics[
                    (policy_metrics.fold_id == row.fold_id)
                    & (policy_metrics.scenario_id == row.scenario_id)
                ]
                history = _history_for_scenario(policy_metrics, row.scenario_id, fold_index)
                evaluation_split = "forward_evaluation"
                expected_history_count = len(history) if row.selector in (
                    "walkforward_robust_80pct", "walkforward_conservative_lcb"
                ) else 0
            else:
                current = policy_metrics[policy_metrics.scenario_id == row.scenario_id]
                history = _history_for_scenario(fold_metrics, row.scenario_id, 6)
                evaluation_split = "final_test"
                expected_history_count = 5 if row.selector in (
                    "walkforward_robust_80pct", "walkforward_conservative_lcb"
                ) else 0

            expected = _expected_selector_rows(current, history)[row.selector]
            if str(row.selected_config_id) != expected[0]:
                raise AssertionError(f"selector choice mismatch: {phase}/{row.scenario_id}/{row.selector}")
            if bool(row.abstained) != expected[1]:
                raise AssertionError(f"selector abstention mismatch: {phase}/{row.scenario_id}/{row.selector}")
            if int(row.history_folds) != expected_history_count:
                raise AssertionError("selector history count mismatch")
            metric = current[
                (current.config_id == row.selected_config_id)
                & (current.split == evaluation_split)
            ].iloc[0]
            baseline_metric = current[
                (current.config_id == row.baseline_config_id)
                & (current.split == evaluation_split)
            ].iloc[0]
            if not np.isclose(float(row.selected_evaluation_cost), float(metric.mean_cost), atol=1e-9):
                raise AssertionError("selector evaluation cost mismatch")
            if not np.isclose(float(row.selected_evaluation_fill_rate), float(metric.mean_fill_rate), atol=1e-10):
                raise AssertionError("selector evaluation service mismatch")
            if not np.isclose(float(row.baseline_evaluation_cost), float(baseline_metric.mean_cost), atol=1e-9):
                raise AssertionError("baseline evaluation cost mismatch")
            if not np.isclose(float(row.baseline_evaluation_fill_rate), float(baseline_metric.mean_fill_rate), atol=1e-10):
                raise AssertionError("baseline evaluation service mismatch")

            wk = weekly_metrics[
                (weekly_metrics.scenario_id == row.scenario_id)
                & (weekly_metrics.split == evaluation_split)
            ]
            selected_week = wk[wk.config_id == row.selected_config_id].groupby("week").agg(
                cost=("mean_total_cost", "mean"), fill=("fill_rate", "mean")
            )
            baseline_week = wk[wk.config_id == row.baseline_config_id].groupby("week").agg(
                cost=("mean_total_cost", "mean"), fill=("fill_rate", "mean")
            )
            paired = selected_week.join(baseline_week, lsuffix="_selected", rsuffix="_baseline", how="inner")
            cost_delta = (paired.cost_selected - paired.cost_baseline).to_numpy(float)
            service_delta = (paired.fill_selected - paired.fill_baseline).to_numpy(float)
            token = f"{phase}|{row.fold_id}|{row.scenario_id}|{row.selector}".encode("utf-8")
            seed = 50050 + int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
            expected_cost_ci = _block_bootstrap_ci(
                cost_delta, seed, bootstrap_reps, bootstrap_block
            )
            expected_service_ci = _block_bootstrap_ci(
                service_delta, seed + 1, bootstrap_reps, bootstrap_block
            )
            observed_cost_ci = json.loads(row.cost_delta_block_bootstrap_95_ci)
            observed_service_ci = json.loads(row.service_delta_block_bootstrap_95_ci)
            if not np.allclose(observed_cost_ci, expected_cost_ci, atol=1e-12):
                raise AssertionError("weekly cost block-bootstrap interval mismatch")
            if not np.allclose(observed_service_ci, expected_service_ci, atol=1e-12):
                raise AssertionError("weekly service block-bootstrap interval mismatch")
            if not np.isclose(float(row.worst_week_cost_delta), float(cost_delta.max()), atol=1e-10):
                raise AssertionError("worst-week cost regret mismatch")
            if not np.isclose(float(row.worst_week_service_delta), float(service_delta.min()), atol=1e-10):
                raise AssertionError("worst-week service degradation mismatch")

    # Recompute the compact selector scorecard from the full scenario-level rows.
    expected_summary = []
    for phase, frame in (("walkforward", fold_decisions), ("final_test", final_decisions)):
        for selector, group in frame.groupby("selector", sort=True):
            expected_summary.append({
                "phase": phase,
                "selector": selector,
                "scenario_fold_count": int(len(group)),
                "abstention_rate": float(group.abstained.mean()),
                "mean_cost_delta_vs_baseline": float(group.cost_delta_vs_baseline.mean()),
                "worst_cost_regret_vs_baseline": float(group.cost_delta_vs_baseline.max()),
                "cost_win_count": int((group.cost_delta_vs_baseline < 0).sum()),
                "mean_service_delta_vs_baseline": float(group.service_delta_vs_baseline.mean()),
                "worst_service_degradation": float(group.service_delta_vs_baseline.min()),
                "service_noninferiority_rate": float((group.service_delta_vs_baseline >= -TOLERANCE).mean()),
            })
    expected_summary_frame = pd.DataFrame(expected_summary)
    merged_summary = selector_summary.merge(
        expected_summary_frame, on=["phase", "selector"], suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    for column in (
        "scenario_fold_count", "abstention_rate", "mean_cost_delta_vs_baseline",
        "worst_cost_regret_vs_baseline", "cost_win_count",
        "mean_service_delta_vs_baseline", "worst_service_degradation",
        "service_noninferiority_rate",
    ):
        if not np.allclose(
            merged_summary[f"{column}_observed"],
            merged_summary[f"{column}_expected"],
            atol=1e-10,
        ):
            raise AssertionError(f"selector summary mismatch: {column}")

    result = {
        "status": "passed",
        "version": "0.50",
        "folds": 5,
        "scenarios": scenario_count,
        "policy_configurations": config_count,
        "replications_per_scenario_config_fold": replications,
        "walkforward_replicate_rows": len(fold_reps),
        "walkforward_weekly_rows": len(fold_weekly),
        "walkforward_policy_metric_rows": len(fold_metrics),
        "walkforward_selector_decisions": len(fold_decisions),
        "final_replicate_rows": len(final_reps),
        "final_weekly_rows": len(final_weekly),
        "final_policy_metric_rows": len(final_metrics),
        "final_selector_decisions": len(final_decisions),
        "preflight_manifest_sha256": preflight_sha,
        "checks": [
            "preflight written before candidate benchmark",
            "row-level series/week partition hashes",
            "five-fold expanding-origin chronology and missing-week detection",
            "future series and actual post-horizon records excluded from fold panels",
            "final 16-week holdout exclusion from all walk-forward inputs",
            "replication completeness and duplicate detection",
            "replicate-to-aggregate metric recomputation",
            "V0.49 single-window selector recomputation",
            "80-percent historical feasibility selector recomputation with three-fold minimum",
            "one-sided 90-percent service LCB selector recomputation",
            "baseline-only fallback recomputation",
            "weekly paired block-bootstrap interval recomputation",
            "worst-week cost regret and service degradation recomputation",
            "selected policy evaluation metric linkage",
            "selector summary recomputation",
        ],
        "scope_warning": manifest["limitations"],
    }
    (path / "audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
