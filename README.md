# FutureLab V0.40 — Cost-aware Sequential Causal Auditing

An extension of V0.39's synthetic group-level causal archive. Frozen candidate/reference policy audits with finite-horizon alpha spending or a Gaussian mixture likelihood-ratio e-process, economic screening and optional soft archive writes. No real commercial data or production deployment.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 python src/benchmark_v40.py
```

`src/v40.py` is the independent audit module; `src/v39.py` receives two optional extensions with unchanged defaults. `src/benchmark_v40.py` runs 23 scene/stress types, 100 independent trajectories each, 10 policies. `results/` includes raw trajectory aggregates, summary, paired bootstrap intervals and fixed-policy cost sensitivity; `REPORT.md` documents assumptions and negative results. Exact V0.39 release preserved in `previous_version/` (excluded from Git history). `peek3_invalid` is an intentionally invalid negative control, never a deployment recommendation.
