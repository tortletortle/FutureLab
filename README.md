# FutureLab V0.43 — Reality Benchmark

**Status: benchmark implemented; real-data execution blocked by network.** No real-world uplift or profit demonstrated. Prior V0.42 engine retained unchanged. New static adapter tests risk-aware DR ranking, not the full dynamic audit engine.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python src/reality_v43.py --fixture  # engineering-only synthetic smoke; new output directory if already exists
python src/reality_v43.py --download # fetch verified compressed Hillstrom mirror and run locked real benchmark
# or after downloading mirror manually to data/hillstrom.csv.gz:
python src/reality_v43.py --data data/hillstrom.csv.gz
python src/economics_v43.py
```

- `docs/V43_PROTOCOL.md`: fixed protocol, primary endpoint and failure criterion.
- `src/reality_v43.py`: source checksum, pre-treatment feature whitelist, isolated train/validation/test, selected baseline comparison, AIPW/IPW sensitivity, cost and budget frontier.
- `src/economics_v43.py`: break-even reanalysis of original V0.42 simulation trajectories.
- `results/v43/synthetic_fixture/`: clearly labeled synthetic smoke outputs; not commercial evidence.
- `results/v43/economics.csv`: historical simulation derived results.
- `REPORT.md`: scope, actual results, blockers and next steps.
- `docs/REPORT_v42.md`: prior report. Raw Hillstrom data not distributed. CausalForestDML optional (`pip install econml` in compatible environment); not run this round.
