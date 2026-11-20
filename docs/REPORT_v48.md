# FutureLab V0.48 — stochastic lead time and constrained replenishment

## 1. Executive summary

V0.48 adds a stochastic lead-time mixture, portfolio-wide procurement budget and normalized capacity constraints, SKU-level MOQ/pack multiples, value-density priority allocation, a one-period integer allocation MILP reference, and an adapter for future real supplier purchase-order exports.

The result is mixed, not a general algorithm win:

- The selection-frozen priority family had lower test proxy cost in 8 of 16 stress scenarios.
- Three scenarios had a descriptive block-bootstrap interval wholly in favor of the priority family; one scenario had an interval wholly against it; 12 intervals crossed zero.
- Mean relative cost reduction across scenarios was 2.65%, but the median was only 0.07%. The mean is dominated by a small number of high-budget/balanced scenarios.
- All eight `balanced` scenarios had lower candidate point cost; all eight `capital_pressure` scenarios had higher candidate point cost.
- With the quantile held fixed at q75, greedy priority allocation had a mean cost reduction of **-1.24%** versus proportional allocation and won only 5/16 scenarios. The allocation heuristic itself is therefore not shown to be generally superior.
- Candidate fill rate averaged 54.05% versus 57.36% for the baseline (−3.30 percentage points). Lower proxy cost can coexist with lower service, so service and cost must be evaluated together.

No result is a measured enterprise saving. The benchmark uses recorded sales and explicitly assumed lead-time distributions, budgets, capacities, pack sizes, selling-price proxies and economic coefficients.

## 2. Data and design

Source: previously audited UCI Online Retail II prepared panel.

| Data item | Count |
|---|---:|
| Original transaction rows | 1,067,371 |
| Valid transaction rows | 1,041,670 |
| Prepared product-country-week rows | 8,979 |
| Series | 100 |
| Train | 5,400 SKU-weeks / 54 weeks |
| Calibration | 1,100 / 11 weeks |
| Selection | 1,000 / 10 weeks |
| Test | 1,600 / 16 weeks |

The benchmark crosses 2 lead-time distributions × 2 budget limits × 2 capacity limits × 2 economic cost scenarios = 16 stress scenarios. Five policy configurations are evaluated over 20 common-random-number replications each: 1,600 policy simulations, or 4.16 million SKU-week state updates.

The last four test weeks do not allow new orders; all previously placed orders can arrive within the observed test horizon.

## 3. Main results

| Result | Value |
|---|---:|
| Priority-family point wins | 8 / 16 |
| Scenarios with interval wholly favoring candidate | 3 |
| Scenarios with interval wholly favoring baseline | 1 |
| Intervals crossing zero | 12 |
| Mean relative proxy-cost reduction | +2.65% |
| Median relative proxy-cost reduction | +0.07% |
| Mean fixed-q75 greedy-vs-proportional reduction | −1.24% |
| Fixed-q75 wins | 5 / 16 |
| Mean candidate fill rate | 54.05% |
| Mean baseline fill rate | 57.36% |
| Mean fill-rate difference | −3.30 pp |

### By economic scenario

| Economic scenario | Point wins | Mean relative proxy-cost change |
|---|---:|---:|
| Balanced | 8 / 8 | +8.15% reduction |
| Capital pressure | 0 / 8 | −2.86% (candidate cost higher) |

The candidate family selected `q75_greedy` in 13 scenarios and `q90_greedy` in 3. It did not select q50 greedy. The q90 selections occur in some high-budget balanced scenarios and account for much of the aggregate mean improvement.

Interpretation:
- Under balanced holding/shortage weights, the candidate family can improve proxy cost, especially when the budget cap is 80% and q90 is selected.
- Under capital-pressure weights, the priority family is consistently worse on point estimates. The current priority score does not adequately manage the long-run capital occupancy and holding exposure created by replenishment.
- The fixed-q75 ablation is negative on average. The observed family-level gains should not be attributed to the greedy allocation algorithm alone; forecast quantile selection is an important contributor.
- Lower candidate fill rate indicates that lower proxy cost is partly obtained by accepting more unfilled recorded sales. This is a trade-off, not a free efficiency gain.

## 4. MILP and engineering audit

- 168/168 automated tests passed: 145 historical tests, 16 V0.48 inventory tests and 7 supplier-adapter tests.
- 83,200 saved selected-policy SKU-week decisions were independently audited.
- 128 sampled one-period integer-lot allocation MILPs all reached optimal status.
- Median greedy-to-MILP objective gap: 0.00%; maximum: 2.34%.
- Audit verified selection-only champion reproduction, budget, capacity, MOQ, order bounds, demand balance, inventory/pipeline conservation, shipment timing, cost decomposition, terminal pipeline clearance and terminal order cutoff.

The MILP result is exact only for the sampled one-period linearized allocation subproblem. It does not prove multi-period global optimality.

## 5. Supplier data readiness

Public data screening found a purchase-order/lead-time sample that advertises 840 orders, but its own description explicitly says suppliers, items and figures are invented. It was rejected as empirical supplier evidence. V0.48 therefore continues to use declared stress distributions.

A supplier-order adapter was added for real exports. It requires supplier ID, order date and receipt date, filters to orders placed before the evaluation cutoff, applies supplier-to-global shrinkage, and flags lead times over 28 days as incompatible with the current four-week simulation cap. Its tests use generated fixtures only; no supplier records were used in the benchmark.

## 6. Evidence limitations

1. Online Retail II does not contain inventory-on-hand, purchase orders, supplier IDs, lead times, stockout labels or acquisition costs.
2. Recorded sales may be censored by stockouts; missing calendar weeks are treated as zero recorded sales in the inherited panel.
3. Selling price is used only as a value proxy; it is not procurement cost or gross margin.
4. Budget and capacity are normalized scenario constraints, not measured corporate resources.
5. Lead-time distributions and MOQ/pack rules are assumed stress scenarios, not empirically estimated.
6. All scenarios share one retailer and one time window; they are not independent replications.
7. The HGB one-step mean and scale are held constant over the lead-time horizon.
8. The bootstrap intervals are descriptive, with only 16 test weeks; they do not cover model-selection, cross-retailer or latent-demand uncertainty.

## 7. V0.49 research priorities

1. **Capital-aware marginal benefit**: estimate holding duration and expected capital occupancy of each order, not only MOQ overage.
2. **Service-level constrained optimization**: compare minimum-fill-rate constraints with cost minimization; do not allow cost reduction to hide unacceptable service deterioration.
3. **Real supplier data**: ingest actual purchase-order/receipt records and replace assumed lead-time distributions only after sample-size and temporal validation.
4. **Demand censoring**: distinguish zero recorded sales from unavailable inventory; require stockout or inventory observations before claiming latent-demand accuracy.
5. **External replication**: seek a dataset with aligned sales, on-hand, replenishment, lead-time and unit-cost observations. Synthetic data may be used for software stress tests, never as empirical proof.
