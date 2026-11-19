# FutureLab V0.39 — Adaptive Risk Calibration

Research extension of V0.38. Adds a prequential residual stress envelope, optional gated archive writes, and an independent Gaussian randomized-contrast audit. No real commercial data, live experiment execution or CATE production integration. The audit assumptions are essential; this is not a universal safety guarantee.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 python src/benchmark_v39.py
```

- `src/v39.py`: new algorithm; prior V0.38 implementation unchanged.
- `src/benchmark_v39.py`: 21 scenarios, 100 seeds each, 7 policies, paired trajectory-level analysis.
- `results/config.json`: fixed protocol, seeds and costs.
- `results/trajectories.csv`, `summary.csv`, `scenes.csv`, `paired.csv`: raw per-trajectory aggregates, summaries and stratified paired bootstrap intervals.
- `REPORT.md`: methods, genuine results and negative findings.
- `previous_version/`: exact V0.38 ZIP. Not included in Git tree to avoid nested binary archives; previous commits retained instead.

The independent-audit variant (without hard write quarantine) is the more promising approach under this synthetic benchmark, but always recomputing or a stricter original model gate remains a strong cost baseline. The original discovery/probe evidence is shared; audit observations are optional extra samples and explicitly billed. No measured reuse-operation savings.
