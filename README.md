# FutureLab V0.38 — Decision Equivalence Learning

Research prototype for dynamic, group-level Gaussian randomized-contrast simulation. It extends the V0.36 bounded memory architecture and V0.37 verification direction. It preserves the original V0.37 archive in `previous_version/` (distribution ZIP only); the Git history preserves the previous source snapshot.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python src/benchmark_v38.py
python src/confirm_v38.py
```

- `src/v38.py`: working belief, relative risk, Gaussian-surrogate EVSI, bounded online archive with selected-probe feedback.
- `results/summary_v38.csv`, `scenes_v38.csv`, `curves_v38.csv`, `paired_v38.csv`: first held-out evaluation.
- `results/confirmation_summary.csv`, `confirmation_scenes.csv`: fresh-seed confirmation at two fixed thresholds; same scenario families.
- `results/*periods*.csv.gz`, `development_v38.csv.gz`, `trajectories_v38.csv`: raw simulation and trajectory summaries.
- `REPORT.md`: scope, mathematical definitions, observed results and failures.

This is not production business software. All data are synthetic; the risk threshold is conditional on a misspecified working model, not a safety guarantee. Reuse is assessed relative to recomputed policy, not a known oracle. The recomputed policy is actually computed for gating; operational savings from reuse have not been demonstrated. GitHub remote upload is not included.
