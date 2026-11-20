# FutureLab V0.48

Commercial decision research prototype: stochastic lead-time replenishment under portfolio-wide budget, normalized capacity and MOQ/pack constraints.

## Current release

V0.48 reuses the audited V0.47/V0.46 retail-sales pipeline and adds:
- 1–4 week stochastic lead-time mixture calibrated from train/calibration partitions (stress distributions are assumed);
- budget and normalized portfolio capacity limits;
- per-series MOQ/pack multiples;
- proportional vs marginal-value greedy allocation;
- exact one-period integer-lot MILP snapshots via SciPy;
- supplier purchase-order import/lead-time distribution adapter for future real supplier exports.

**Evidence boundary:** this release uses UCI Online Retail II recorded sales. It does not have actual inventory, stockout, supplier, purchase-order, lead-time, acquisition-cost or gross-margin records. Budget, capacity, lead-time and pack-size settings are scenario assumptions. The output is a historical recorded-sales replay, not measured enterprise profit.

## Run locally

```bash
python -m pip install -r requirements-v48.txt
python -m pytest -q
python src/inventory_v48.py \
  --panel data/retail_panel.csv \
  --quality data/retail_quality.json \
  --output results/v48/inventory \
  --replications 20
python src/audit_v48.py --results results/v48/inventory
```

## Main files

- `src/inventory_v48.py` — stochastic lead-time mixture, constrained replenishment, policies and benchmark runner.
- `src/audit_v48.py` — independent state-transition, constraint and selection audit.
- `src/supplier_leadtime_adapter.py` — import/validation for real supplier PO exports.
- `tests/test_v48.py` — V0.48 inventory tests.
- `tests/test_supplier_leadtime_adapter.py` — supplier adapter tests.
- `docs/V48_PROTOCOL.md` — locked evaluation protocol and assumptions.
- `docs/REPORT_v48.md` — results, negative findings and evidence limitations.
- `docs/OPEN_SOURCE_REVIEW_V48.md` — reuse and data-source review.
- `results/v48/inventory/` — reproducible metrics, audit samples, MILP snapshots and manifest.

## Prior versions

The V0.47 source and results are retained. V0.48 does not overwrite V0.43–V0.47 history or the separate Hillstrom marketing-uplift protocol.
