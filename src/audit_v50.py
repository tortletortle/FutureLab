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
    required = int(math.ceil(ROBUST_FRACTION * n)) if n else 1
    if mode == "robust" and n == 0:
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
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    calendar = pd.read_csv(path / "fold_calendar.csv")
    fold_reps = pd.read_csv(path / "fold_replicate_metrics.csv")
    fold_metrics = pd.read_csv(path / "fold_policy_metrics.csv")
    fold_decisions = pd.read_csv(path / "fold_selector_decisions.csv")
    final_reps = pd.read_csv(path / "final_replicate_metrics.csv")
    final_metrics = pd.read_csv(path / "final_policy_metrics.csv")
    final_decisions = pd.read_csv(path / "final_selector_results.csv")

    if manifest.get("version") != "0.50":
        raise AssertionError("wrong benchmark version")
    replications = int(manifest["replications"])
    scenario_count = int(manifest["scenario_count"])
    config_count = int(manifest["policy_config_count"])
    if len(calendar) != 5 or manifest.get("fold_count") != 5:
        raise AssertionError("expected five walk-forward folds")
    if scenario_count != 32 or config_count != 8:
        raise AssertionError("unexpected scenario/configuration grid")
    if manifest.get("selectors") != list(SELECTORS):
        raise AssertionError("selector registry mismatch")
    if len(manifest["final_test_week_ids"]) != 16:
        raise AssertionError("final holdout must contain 16 weeks")

    final_test = set(manifest["final_test_week_ids"])
    for row in calendar.itertuples(index=False):
        windows = {
            "train": json.loads(row.train_weeks),
            "calibration": json.loads(row.calibration_weeks),
            "selection": json.loads(row.selection_weeks),
            "forward": json.loads(row.forward_weeks),
        }
        flattened = [w for values in windows.values() for w in values]
        if len(flattened) != len(set(flattened)):
            raise AssertionError(f"{row.fold_id} has overlapping windows")
        if set(flattened).intersection(final_test):
            raise AssertionError(f"{row.fold_id} overlaps frozen final test")
        if [len(windows[k]) for k in ("train", "calibration", "selection", "forward")] != [
            30 + 6 * (int(row.fold_index) - 1), 8, 6, 6
        ]:
            raise AssertionError(f"{row.fold_id} window lengths differ from protocol")
        if windows["train"] != sorted(windows["train"]):
            raise AssertionError(f"{row.fold_id} training weeks are not chronological")
        if max(windows["train"]) >= min(windows["calibration"]):
            raise AssertionError(f"{row.fold_id} train/calibration ordering invalid")
        if max(windows["calibration"]) >= min(windows["selection"]):
            raise AssertionError(f"{row.fold_id} calibration/selection ordering invalid")
        if max(windows["selection"]) >= min(windows["forward"]):
            raise AssertionError(f"{row.fold_id} selection/forward ordering invalid")

    expected_fold_reps = 5 * scenario_count * config_count * replications * 2
    expected_fold_metrics = 5 * scenario_count * config_count * 2
    expected_final_reps = scenario_count * config_count * replications * 2
    expected_final_metrics = scenario_count * config_count * 2
    if len(fold_reps) != expected_fold_reps or len(fold_metrics) != expected_fold_metrics:
        raise AssertionError("walk-forward metric row counts do not match protocol")
    if len(final_reps) != expected_final_reps or len(final_metrics) != expected_final_metrics:
        raise AssertionError("final metric row counts do not match protocol")
    if len(fold_decisions) != 5 * scenario_count * len(SELECTORS):
        raise AssertionError("fold selector decision row count mismatch")
    if len(final_decisions) != scenario_count * len(SELECTORS):
        raise AssertionError("final selector decision row count mismatch")

    if fold_reps.duplicated(["fold_id", "scenario_id", "config_id", "replicate", "split"]).any():
        raise AssertionError("duplicate walk-forward replicate rows")
    if final_reps.duplicated(["scenario_id", "config_id", "replicate", "split"]).any():
        raise AssertionError("duplicate final replicate rows")
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

    # Recompute each selector decision from the archived fold-level metrics.
    for row in fold_decisions.itertuples(index=False):
        fold_index = int(str(row.fold_id)[1:])
        current = fold_metrics[
            (fold_metrics.fold_id == row.fold_id)
            & (fold_metrics.scenario_id == row.scenario_id)
        ]
        history = _history_for_scenario(fold_metrics, row.scenario_id, fold_index)
        expected = _expected_selector_rows(current, history)[row.selector]
        if str(row.selected_config_id) != expected[0]:
            raise AssertionError(f"selector choice mismatch: {row.fold_id}/{row.scenario_id}/{row.selector}")
        if bool(row.abstained) != expected[1]:
            raise AssertionError(f"selector abstention mismatch: {row.fold_id}/{row.scenario_id}/{row.selector}")
        expected_history_count = len(history) if row.selector in ("walkforward_robust_80pct", "walkforward_conservative_lcb") else 0
        if int(row.history_folds) != expected_history_count:
            raise AssertionError("selector history count mismatch")
        metric = current[
            (current.config_id == row.selected_config_id)
            & (current.split == "forward_evaluation")
        ].iloc[0]
        if not np.isclose(float(row.selected_evaluation_cost), float(metric.mean_cost), atol=1e-9):
            raise AssertionError("fold selector evaluation cost mismatch")
        if not np.isclose(float(row.selected_evaluation_fill_rate), float(metric.mean_fill_rate), atol=1e-10):
            raise AssertionError("fold selector evaluation service mismatch")

    # The final holdout is evaluated only after all five fold outcomes are available.
    for row in final_decisions.itertuples(index=False):
        current = final_metrics[final_metrics.scenario_id == row.scenario_id]
        history = _history_for_scenario(fold_metrics, row.scenario_id, 6)
        expected = _expected_selector_rows(current, history)[row.selector]
        if str(row.selected_config_id) != expected[0]:
            raise AssertionError(f"final selector choice mismatch: {row.scenario_id}/{row.selector}")
        if bool(row.abstained) != expected[1]:
            raise AssertionError(f"final selector abstention mismatch: {row.scenario_id}/{row.selector}")
        expected_history_count = 5 if row.selector in ("walkforward_robust_80pct", "walkforward_conservative_lcb") else 0
        if int(row.history_folds) != expected_history_count:
            raise AssertionError("final selector history count mismatch")
        metric = current[
            (current.config_id == row.selected_config_id)
            & (current.split == "final_test")
        ].iloc[0]
        if not np.isclose(float(row.selected_evaluation_cost), float(metric.mean_cost), atol=1e-9):
            raise AssertionError("final selector evaluation cost mismatch")
        if not np.isclose(float(row.selected_evaluation_fill_rate), float(metric.mean_fill_rate), atol=1e-10):
            raise AssertionError("final selector evaluation service mismatch")

    result = {
        "status": "passed",
        "version": "0.50",
        "folds": 5,
        "scenarios": scenario_count,
        "policy_configurations": config_count,
        "replications_per_scenario_config_fold": replications,
        "walkforward_replicate_rows": len(fold_reps),
        "walkforward_policy_metric_rows": len(fold_metrics),
        "walkforward_selector_decisions": len(fold_decisions),
        "final_replicate_rows": len(final_reps),
        "final_policy_metric_rows": len(final_metrics),
        "final_selector_decisions": len(final_decisions),
        "checks": [
            "five-fold expanding-origin chronology",
            "train/calibration/selection/forward window disjointness",
            "final 16-week holdout exclusion from all walk-forward windows",
            "replication completeness and duplicate detection",
            "replicate-to-aggregate metric recomputation",
            "V0.49 single-window selector recomputation",
            "80-percent historical feasibility selector recomputation",
            "one-sided 90-percent service LCB selector recomputation",
            "baseline-only fallback recomputation",
            "selected policy evaluation metric linkage",
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
