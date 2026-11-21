# FutureLab V0.42 — Adaptive Evidence Routing

V0.42 extends the frozen-candidate evidence audit of V0.41 with declared failure domains, residual-bias sensitivity bounds, and adaptive finite-horizon routing under a sample-equivalent budget. Source metadata and fault budgets are externally supplied assumptions, not certificates automatically inferred from data.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 python src/benchmark_v42.py
```

- `src/v42.py`: domain-aware audit router; abstains when provenance assumptions are unavailable.
- `src/benchmark_v42.py`: fresh-seed boundary/power experiments and frozen-policy dynamic stress tests; supports `boundary` and `dynamic` arguments.
- `results/v42/`: configuration, exact binomial pointwise intervals, per-trajectory aggregates, summaries, paired bootstrap, test log.
- `REPORT.md`: proof scope, results, failures and limitations.
- `docs/REPORT_v41.md`: previous report; prior code/results remain unchanged.
- ZIP contains the unchanged V0.41 ZIP under `previous_version/` (excluded from Git).

All data are synthetic. No automatic discovery of failure domains, no verified real-world source independence, no policy reoptimization using audit samples, no production deployment. The gate does not guarantee safe performance under misdeclared fault budgets, residual bias or serial dependence. Rejecting reuse is not a guarantee that the recomputed reference itself is safe.
