# FutureLab V0.45 — Economic evidence readout

Incremental development from V0.44. Raw Hillstrom dataset acquisition remains blocked. No real individual-level FutureLab benchmark or commercial superiority demonstrated. V0.43–44 protocol and benchmark are retained; this release adds a separate economic evaluation layer, not a new trained model.

```bash
python -m pip install -r requirements-v44-lock.txt
python -m pytest -q
python src/value_v45.py --locked results/v43/synthetic_fixture --output /tmp/futurelab_fixture_value.json
python src/value_v45.py --external docs/V45_EXTERNAL_AGGREGATE.json --output /tmp/futurelab_external_value.json
# After legally acquiring verified raw data:
python src/reality_v44.py --data data/hillstrom.csv.gz --output results/real_hillstrom
python src/value_v45.py --locked results/real_hillstrom --output results/real_value.json
```
Outputs must not exist already. See REPORT.md for assumptions, negative results, evidence and next actions. `results/v45` stores actual local execution outputs. External aggregate scenarios, synthetic results and raw real-data validation are kept separate. The manual workflow has not been executed on GitHub.
