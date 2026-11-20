# FutureLab V0.46 — Recorded-sales inventory decision benchmark

Incremental release over V0.45. The locked Hillstrom causal benchmark and economic evidence gate are unchanged. V0.46 adds a separate retrospective retail experiment using observed positive transactions. No inventory, availability, shortage, or real cost data are available; the loss function is hypothetical and does not measure real profit.

```bash
python -m pip install -r requirements-v46.txt
python -m pytest -q
python src/retail_v46.py --data /path/to/online+retail+ii.zip --output results/new-retail-run
```

Provide the original UCI Online Retail II ZIP (SHA-256 pinned in module). The original input file is not redistributed. `docs/V46_PROTOCOL.md` specifies the chronological experiment; `results/v46/retail/` contains the evaluated outputs if the run completed. Previous release documents are preserved under `docs/`.

Hillstrom commands (unchanged):
```
python src/reality_v44.py --download --output results/new-hillstrom
python src/value_v45.py --locked results/new-hillstrom --output results/new-value.json
```
