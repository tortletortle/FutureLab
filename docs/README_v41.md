# FutureLab V0.41 — Trustworthy Causal Evidence Engine

The V0.40 dynamic causal archive remains unchanged. This increment adds an independent finite-horizon evidence gate for frozen candidates and recomputed reference actions: unknown-variance Student-t bounds, matched negative-control contrasts, pooled sources, and robust median-of-bounds fusion. A matched negative control and honest independent Gaussian blocks are assumptions, not properties verified by this software. No production data or validated commercial gain.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 python src/benchmark_v41.py
```

- `src/v41.py`: reusable evidence gate, per-attempt stopping and costs.
- `src/benchmark_v41.py`: controlled boundary/power experiments and dynamic frozen-policy stress tests.
- `results/v41/`: config, boundary CSV with exact binomial intervals, trajectory aggregates, dynamic scene/summary tables, paired bootstrap and test logs.
- `REPORT.md`: derivations, results, failure cases and interpretation.
- `docs/REPORT_v40.md`: previous technical report; historical source and results remain in Git.
- Release ZIP includes an unchanged V0.40 archive under `previous_version/` (excluded from Git).

Run with Python 3 and dependencies listed in `requirements.txt`. All experiments are synthetic; the gate makes no guarantee for unmatched systematic bias, multiple compromised sources, serially dependent blocks, or across repeated deployment periods. Dynamic audit data are not fed back to the archive or used to train a new policy in this version.
