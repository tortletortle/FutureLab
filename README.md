# FutureLab V0.47 — multi-period inventory policy optimization

V0.47 extends V0.46's recorded-sales forecasting benchmark with a deterministic lead-time inventory simulator. It compares global and SKU-volume-group residual calibration, rolling baselines, HGB/XGBoost forecasts, base-stock and `(s,S)` policies across 1-, 2- and 4-week lead times and four assumed cost scenarios.

**Important evidence limit:** the UCI Online Retail II dataset contains transactions, not stock-on-hand or stockout records. The experiment treats recorded positive sales as a demand proxy. Inventory, lead times and costs are simulated assumptions; results are not the retailer's actual profit, service level or replenishment performance.

## Run

```bash
python -m pip install -r requirements-v47.txt
python -m pytest -q

# Optional fast path: use the hash-pinned V46 panel if you already have it locally.
python src/inventory_v47.py \
  --panel data/retail_panel.csv \
  --quality data/retail_quality.json \
  --output results/v47/inventory-new

# Standard path: re-parse the original UCI ZIP.
python src/inventory_v47.py \
  --data /path/to/online+retail+ii.zip \
  --output results/v47/inventory-from-raw

python src/audit_v47.py results/v47/inventory-new
```

The optional fast path checks exact SHA-256 hashes of both the prepared panel and its quality manifest, and verifies the original UCI source hashes recorded in that manifest. The standard path re-parses both workbook sheets using the existing V0.25 streaming adapter.

## Results

- `results/v47/inventory/manifest.json`: data provenance, frozen splits, calibration, scenarios and paired results.
- `policy_grid_metrics.csv`: all 480 policy configurations across 12 scenarios.
- `selected_policies.csv`: selection-period winners for baseline and candidate families.
- `selected_policy_decisions.csv`: period-by-period inventory state for the 12 paired comparisons.
- `audit.json`: independent state-transition, policy-selection, cost and paired-interval audit.

See `docs/V47_PROTOCOL.md` for the complete experiment protocol and `docs/OPEN_SOURCE_REVIEW_V47.md` for the Stockpyl reuse assessment.

## Data citation

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Licensed CC BY 4.0. The original 43.5 MB dataset is not redistributed in this repository.

## Other research tracks

The locked Hillstrom marketing-uplift benchmark and V0.43–V0.45 economic evaluation are preserved. They remain separate from this inventory proxy experiment.
