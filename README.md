# FutureLab V0.37 — Active Regime Verification

Research extension of V0.36. Implements a separate controlled archive-retrieval episode API for open-set regime discrimination. The original V0.36 ZIP is retained verbatim in the downloadable release's `previous_version/` directory. Git history preserves the V0.36 source, report, tests and results (nested archives are intentionally excluded from Git).

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python src/benchmark_v37.py
python src/budget_study_v37.py
```

Outputs: `results/summary_v37.csv`, `scenes_v37.csv`, `paired_v37.csv`, `curves_v37.csv`, `trials_v37.csv`, independent development rows and configurations; `budget_*.csv` and `budget_config.json` contain a separate budget study. `REPORT.md` describes protocol, results and failures.

## API

`src/v37.py`: `build_belief`, `update_all`, `binary_gain`, `choose`, `update_probe`, `run_episode`. Predictions and probe choices use only prior archive estimates and observed randomized contrasts. `run_episode` uses an evaluator-owned bank of potential independent probe outcomes; only selected values enter updates. Equal probe counts are a synthetic cost proxy, not guaranteed equal real-world cost.

V0.36 archive is retained unchanged. V0.37 is not yet connected to its streaming archive or a production causal policy service. No real experiments, deployment, commercial profit claims or guaranteed false-recall control.
