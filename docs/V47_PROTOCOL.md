# V0.47 protocol — multi-period inventory policy replay

## Research question

Does lead-time-aware residual calibration, stratified by pre-period SKU scale, improve a periodic-review replenishment policy over a validation-selected baseline when inventory state and replenishment pipelines are simulated explicitly?

This is a separate research track from the locked Hillstrom marketing uplift benchmark. It does not modify V43–V45 protocols.

## Data and provenance

- Dataset: UCI Online Retail II, Chen (2012), DOI `10.24432/C5CG6D`, CC BY 4.0.
- The experiment uses the previously ingested 100 product-country weekly recorded-sales series, not a new external holdout.
- Original ZIP SHA-256: `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`.
- Prepared panel SHA-256: `1d32ccc76a7bb6df0127d0a2d32368a22b79558e33603dfb82afd65eb0293831`.
- Quality manifest SHA-256: `5a564a92ac78c8de95511862ea576aaf9aa5784572b8c732d38be78606c19d6f`.
- The source contains 1,067,371 transaction rows and covers a UK online retailer from December 2009 to December 2011. The source data does not contain inventory-on-hand, stockout, replenishment, lead-time or supplier records.

## Chronological design

The V46 feature builder is reused unchanged. It selects the 100 product-country series using only early-history data, reconstructs the complete Sunday calendar, and makes four chronological partitions:

- Train: 54 weeks / 5,400 SKU-weeks.
- Calibration: 11 weeks / 1,100 SKU-weeks.
- Selection: 10 weeks / 1,000 SKU-weeks.
- Test: 16 weeks / 1,600 SKU-weeks.

The demand models are fitted on train only. Cumulative lead-time residual windows are calibrated on calibration only. Policy family and trigger configuration are selected on selection only. Test data is used once for final comparison. Lagged features may use actual prior-week recorded sales as they would be available in a sequential weekly workflow; no current-week or future sales enter a decision.

## Lead-time forecast

A one-week HGB point forecast `mu_t` is multiplied by deterministic lead time `L`. For each calibration series and each contiguous `L`-week window, compute:

`r_L = sum(actual - mu) / max(sum(trailing_13_week_mean), 1)`.

Estimate the 0.50, 0.75 and 0.90 quantiles of `r_L` globally and within three training-only SKU-volume terciles. Groups with fewer than 50 calibration windows fall back to the pooled quantile. The order-up-to target is:

`S_t(L,q) = max(0, L * mu_t + L * scale_t * Q_q(r_L))`.

This assumes the current one-step forecast and scale remain representative across the protection horizon. It is an approximation, not a direct multi-horizon model.

## Inventory simulator

The simulator uses this explicit order-before-demand event sequence:

1. Receive all shipments due at the start of the period.
2. Calculate inventory position = on-hand + outstanding pipeline.
3. Apply a periodic-review base-stock policy or an `(s,S)` trigger policy.
4. Place replenishment orders; an order placed at week `t` arrives at the start of `t+L`.
5. Observe that week's recorded sales and fulfill from on-hand stock; unmet recorded sales are modeled as lost sales.
6. Charge end-of-period holding cost, lost-sales penalty, and optional fixed order cost.

Initial on-hand inventory is `ceil(L * trailing_13_week_mean)` and is shared across all methods for each series. It is a common assumed initial condition, not an observed stock count. Orders and recorded weekly quantities are rounded upward to integer units.

Policies searched per lead time and cost scenario:

- Forecast families: rolling 13-week quantile, HGB mean, HGB one-week q75, XGBoost one-week q75, global horizon-calibrated q50/q75/q90, and group horizon-calibrated q50/q75/q90.
- Policy forms: base-stock, and `(s,S)` with trigger ratios 0.50, 0.75 and 0.90 of the dynamic order-up-to target.
- 40 configurations per scenario.
- Lead times: 1, 2 and 4 weeks.
- Cost scenarios: balanced (shortage 3, holding 1), service priority (6,1), capital pressure (3,2), and order setup (3,1 plus fixed order cost 10). These are dimensionless assumptions, not observed money costs.

For every scenario, the lowest selection-period cost configuration within the group-calibrated candidate family is compared with the lowest selection-period cost configuration among all other baseline families. Neither policy is reselected on test.

## Primary metrics and uncertainty

Primary metric: average simulated total cost per SKU-week, consisting of holding + lost-recorded-sales + fixed-order costs. Secondary metrics: fill rate against recorded sales, lost units, mean on-hand, mean order quantity and order frequency.

The paired candidate-minus-baseline test difference is reported with a circular four-week block bootstrap interval (2,000 draws, fixed seed). There are only 16 test weeks; intervals are descriptive and do not address cross-retailer transport, latent demand censoring, model-selection uncertainty or all forms of serial/cross-sectional dependence.

## Evidence limits

- Recorded transactions are not latent customer demand. Missing sales may reflect no demand, no availability, or incomplete records.
- The data has no actual inventory or stockout labels; the simulator cannot recover the retailer's historical service level.
- Costs are scenario weights, not currency, gross margin or actual operating cost.
- The lead time is deterministic. Supplier delays, order minimums, capacity, shelf life, cash constraints and multi-echelon dependencies are not modeled.
- This is a new decision endpoint on previously used data, not independent external replication.
- A lower simulated proxy cost is not proof of realized business profit or deployment readiness.
