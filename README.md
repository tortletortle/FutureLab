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

## Experimental research branches

`research/v0.50-walkforward-selection` is an isolated V0.50 research branch. It implements five expanding-origin folds, refits demand/lead-time models per fold, compares the V0.49 single-window selector with an 80%-historical-feasibility selector and a conservative one-sided service-LCB selector, and reserves the final 16 weeks for one-time holdout evaluation. No V0.50 benchmark result is claimed until the remote workflow and independent audit pass. See `docs/V50_PROTOCOL.md`.

Run V0.50 from this branch with:

```bash
python -m pip install -r requirements-v50.txt
python -m pytest -q
python src/inventory_v50.py --panel data/retail_panel.csv --quality data/retail_quality.json --output results/v50/walkforward --replications 20
python src/audit_v50.py --results results/v50/walkforward
```

`research/v0.49-capital-service` is an isolated, unreleased research branch. It tests sequential per-lot marginal valuation, a working-capital exposure proxy and service-floor-constrained policy selection. V0.48 remains the current release and its code/results are unchanged. The local V0.49 20-replication benchmark and independent audit passed, but results are mixed and do not establish a general algorithmic improvement. The GitHub Actions reproduction passed (run 44874903283). See `docs/REPORT_v49.md` and `docs/V49_PROTOCOL.md`.