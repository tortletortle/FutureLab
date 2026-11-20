# FutureLab V0.44 — Verified Reality Benchmark

**Status: real dataset acquisition blocked in the current environment.** No real uplift or commercial profit demonstrated. Restored V0.43 Git history and retained prior algorithm and results. V0.44 adds verified multi-source acquisition, fail-closed validation, provenance reports and manual GitHub Actions execution.

```bash
python -m pip install -r requirements-v44-lock.txt
python -m pytest -q
python src/reality_v44.py --download
# or supply a lawfully obtained exact dataset (raw CSV or pinned gzip):
python src/reality_v44.py --data data/hillstrom.csv
```

- Read `docs/V43_PROTOCOL.md` and `docs/V44_PROTOCOL_ADDENDUM.md` before evaluating real outcomes.
- `results/v44/acquisition.json` records the actual download failures.
- `src/reality_v43.py` is unchanged; its historical synthetic results are not real data.
- `.github/workflows/reality-benchmark.yml` is a manual-dispatch workflow to use after source synchronization; it has not been run remotely.
- `REPORT.md` summarizes the current delivery; older report is `docs/REPORT_v43.md`.
