# V0.48 open-source and data-source review

## Reused components

- **V0.47/V0.46 internal pipeline**: reused the already audited chronological feature builder, HGB/XGBoost forecasts, panel hash checks and inventory event semantics. No second Excel parser was written.
- **SciPy `optimize.milp`**: used as a one-period integer allocation reference under linearized average lot benefit, weekly budget and normalized capacity constraints. This avoids reimplementing a general MILP solver.
- **Stockpyl 1.0.2 (MIT)**: the V0.47 review found existing EOQ, newsvendor, Wagner–Whitin, `(r,Q)`, `(s,S)`, base-stock and multi-echelon inventory functions. V0.48 does not copy Stockpyl. Its multi-period event order and the custom cross-SKU budget/capacity/pack allocation protocol differ from this benchmark; Stockpyl remains a candidate for future cross-checking, not an asserted dependency.

## Data-source screening

- UCI Online Retail II remains the only empirical transaction source used in the V0.48 replay. It has no supplier, inventory or stockout ground truth.
- A public sample advertised as purchase orders, quoted/actual lead times and stockout days (840 orders, CC0) explicitly states that its suppliers, items and figures are invented. It was rejected as real-world evidence. It may be suitable only for parser/demo testing.
- Other inventory demo datasets found during screening were also described as synthetic or generated. None was substituted for actual supplier records.

## New import adapter

`src/supplier_leadtime_adapter.py` defines a fail-closed schema for real PO exports:
- required `supplier_id`, `order_date`, `receipt_date`;
- optional `order_id`, `quantity`, `unit_cost`, `sku_id`, `warehouse`, `promised_date`;
- rejects duplicate order IDs, invalid dates, receipt-before-order and invalid supplied quantities/costs;
- uses only completed orders placed before `as_of`;
- shrinks sparse supplier distributions toward the global distribution;
- flags any lead time over 28 days as incompatible with the current 1–4 week V0.48 simulator.

The adapter is tested with synthetic fixtures, not empirical supplier data.
