# FutureLab V0.48 Protocol — stochastic lead time and constrained replenishment

## 1. Research question

Under uncertain supplier lead times and portfolio-wide procurement/storage constraints, does a value-density allocation heuristic improve a dynamic inventory replay over proportional allocation?

This is a separate inventory research track. It does not change the V0.43–V0.45 Hillstrom marketing-uplift protocol or the V0.47 deterministic inventory results.

## 2. Data and provenance

- Source: UCI Online Retail II, DOI `10.24432/C5CG6D`, CC BY 4.0.
- Prepared panel: 8,979 product-country-week records across 100 series.
- Source workbook: 1,067,371 transaction rows; 1,041,670 valid rows after the previously audited cleaning pipeline.
- The panel and quality manifest are hash-pinned to the V0.47/V0.46 provenance.
- Chronological splits: train 54 weeks / 5,400 SKU-weeks; calibration 11 / 1,100; selection 10 / 1,000; test 16 / 1,600.
- No stock-on-hand, stockout, purchase-order, supplier, replenishment or acquisition-cost fields exist in this dataset.

The benchmark therefore evaluates *recorded-sales replay under assumed operating conditions*. It is not a reconstruction of the retailer's actual inventory or supplier network.

## 3. Forecast and stochastic lead-time mixture

V0.47's chronological feature builder and HGB/XGBoost forecast pipeline are reused. The V0.48 mixture calibration fits an HGB point model on train only and collects cumulative residual windows on calibration only.

For each contiguous calibration window of length `L ∈ {1,2,3,4}`:

`r_L = sum(y - mu) / max(sum(scale), 1)`.

Two lead-time distributions are stress scenarios, not empirical estimates:

- `moderate`: P(L=1,2,3,4) = (0.15, 0.55, 0.20, 0.10), E[L]=2.25.
- `volatile`: P(L=1,2,3,4) = (0.10, 0.35, 0.35, 0.20), E[L]=2.65.

For each target row, the scenario pool is formed from `L * hgb_mean + L * scale * r_L`, with each horizon's residual windows weighted by its declared lead-time probability. The q50/q75/q90 weighted mixture quantiles become order-up-to targets. Current one-step mean and scale are held constant over each horizon; this remains an approximation, not a direct multi-horizon forecast.

## 4. Portfolio constraints and economic proxies

The benchmark crosses:

- Budget cap: 55% or 80% of total training-period mean weekly recorded-sales value.
- Capacity cap: 300 or 400 normalized SKU-week demand-equivalent units.
- Economic assumptions:
  - `balanced`: shortage coefficient 0.25, holding coefficient 0.01, fixed order cost 2.
  - `capital_pressure`: shortage coefficient 0.25, holding coefficient 0.03, fixed order cost 2.
- Lead-time distributions: `moderate` and `volatile`.

This yields 16 stress scenarios.

Important definitions:

- Unit value proxy: training-only median recorded selling price per series. It is **not** acquisition cost, supplier quote or gross margin.
- Budget proxy: weekly limit expressed in the same selling-price-value proxy. It is **not** a verified cash budget.
- Capacity: sum of on-hand plus pipeline divided by each series' training mean weekly recorded sales. It is a normalized portfolio coverage measure, not cubic or physical warehouse capacity.
- MOQ/pack multiple: `max(1, ceil(10% × training mean weekly recorded sales))` units per series. It is a controlled assumption, not an observed supplier pack size.
- Costs: holding and lost-sales penalties are price-weighted proxy units, not realized money or profit.

## 5. Allocation policies

Five configurations are evaluated:

1. `q50_proportional`
2. `q75_proportional`
3. `q50_greedy`
4. `q75_greedy`
5. `q90_greedy`

Proportional allocation scales all requested replenishments by the tightest budget/capacity ratio, then rounds down to pack multiples.

The priority allocator estimates an average value benefit per requested lot from the expected shortage reduction, MOQ overage holding penalty and amortized fixed order cost. It ranks SKUs by that benefit divided by normalized budget and capacity burden, then allocates feasible lots in descending score order.

For each scenario, the proportional-family champion (q50/q75) and priority-family champion (q50/q75/q90) are selected on the 10-week selection period, averaged over replications, then frozen before the 16-week test. The fixed-q75 greedy-vs-proportional ablation holds the forecast quantile constant to isolate the allocation rule.

## 6. Simulation semantics and randomization

Each simulation period follows:

1. Receive shipments due at the start of the week.
2. Compute inventory position = on-hand + pipeline.
3. Compute order-up-to shortfall and round the request to MOQ/pack multiples.
4. Allocate orders under the weekly portfolio budget and capacity caps.
5. Schedule each order using its sampled 1–4 week lead time.
6. Observe that week's recorded sales; fulfill from on-hand and treat unmet recorded sales as lost sales.
7. Charge price-weighted holding, lost-sales and fixed-order proxy costs.

Twenty replications use common random numbers: within a scenario/replication, all policies receive the same per-series/per-week lead-time draw matrix.

No new orders are allowed during the final four test weeks so every order placed during the evaluation horizon can arrive before the test closes. The final four weeks remain in demand/cost evaluation.

## 7. MILP subproblem reference

`scipy.optimize.milp` solves 128 sampled one-period integer-lot allocation subproblems using the same linearized average-benefit objective and budget/capacity constraints. This provides an exact optimum for that *single-period linearized allocation subproblem*. It is not a globally optimal multi-period inventory policy.

The greedy-to-MILP objective gap is reported on sampled selected-policy states.

## 8. Metrics and uncertainty

Primary: average price-weighted proxy cost per SKU-week.

Secondary: fill rate against recorded sales, lost recorded units, budget utilization, capacity utilization and order value proxy.

For each scenario, report:
- selection-frozen family champions and test cost difference;
- circular four-week block-bootstrap 95% descriptive interval across the 16 test weeks (2,000 draws);
- standard deviation of paired test cost differences across 20 simulation replications;
- fixed-q75 allocation ablation.

The 16 scenarios share the same retailer, time window and demand panel. They are stress cases, not 16 independent real-world replications.

## 9. Supplier lead-time import contract

`src/supplier_leadtime_adapter.py` accepts real purchase-order exports with:

- required: `supplier_id`, `order_date`, `receipt_date`;
- optional: `order_id`, `quantity`, `unit_cost`, `sku_id`, `warehouse`, `promised_date`.

It rejects duplicate order IDs, invalid dates, receipts before orders and invalid supplied quantities/costs. It estimates empirical 1–4 week buckets with supplier-to-global shrinkage using orders placed strictly before `as_of`. Four-week values are checked for any actual lead time above 28 days; if present, the output is marked incompatible with V0.48's four-week cap. The adapter is tested with synthetic fixtures only; no supplier purchase-order dataset was used in this benchmark.

## 10. Evidence boundaries

This benchmark does not establish:
- actual customer latent demand or stockout rates;
- actual supplier lead-time distributions or reliability;
- actual acquisition cost, cash budget or warehouse capacity;
- real company profit or ROI;
- cross-retailer generalization.

Missing calendar weeks are inherited as zero *recorded sales*, not proven zero latent demand. All commercial claims must remain bounded by these limitations.
