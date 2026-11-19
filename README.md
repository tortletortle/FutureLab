# FutureLab V0.36 — Reusable Regime Memory

Independent synthetic research extension of V0.35, not a production CATE service integration. The original V0.35 ZIP is retained verbatim in `previous_version/`.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python src/benchmark_v36.py dev
python src/benchmark_v36.py hold
```

`dev` selects a configuration on separate development seeds; `hold` evaluates that frozen configuration on 13 synthetic scenario families. No true labels or effect vectors enter the policies. `results/` contains configuration, per-seed results, summary, paired bootstrap, birth/recall logs, and time-series traces. `REPORT.md` discusses limitations and negative results. `src/v35.py`, `src/v34.py`, and `src/benchmark_v35.py` are retained imports/reference code; to reproduce the full historical benchmark, extract the original V0.35 ZIP into a separate directory.
