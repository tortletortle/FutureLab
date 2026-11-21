# FutureLab V0.49 Protocol — capital-aware marginal lots and service floors

## Research question

Can sequential per-lot marginal valuation, an explicit working-capital exposure proxy, and service-floor-constrained policy selection improve the V0.48 recorded-sales replay without obtaining apparent cost gains by sacrificing fill rate?

V0.49 is an isolated research branch. V0.48 remains unchanged and is the baseline.

## Hypotheses

- **H1 — Marginal valuation:** re-ranking after each accepted lot will reduce the objective gap to an exact one-period lot-allocation MILP relative to a SKU-level average-benefit greedy score.
- **H2 — Capital sensitivity:** increasing the assumed weekly capital charge will reduce allocations with long lead/excess exposure, with a measurable change in proxy cost and capital exposure.
- **H3 — Service protection:** selecting a candidate only when its selection-period fill rate is within 0.5 percentage points of the proportional baseline will prevent selection-time service degradation. Test-period adherence is measured separately and is not guaranteed.

These are hypotheses, not expected outcomes.

## Data and temporal protocol

Reuse the V0.48 hash-pinned UCI Online Retail II product-country weekly recorded-sales panel and its existing chronological partitions:

- train: 54 weeks / 5,400 series-weeks;
- calibration: 11 weeks / 1,100 series-weeks;
- selection: 10 weeks / 1,000 series-weeks;
- test: 16 weeks / 1,600 series-weeks.

Forecasts are fitted on train only. Residual windows and lead-mixture calibration use calibration only. Policy selection uses selection only. Test remains frozen for evaluation. Twenty replications use common lead-time draws across all policies within each scenario/replication.

No stock-on-hand, stockout, purchase-order, supplier, lead-time, acquisition-cost or gross-margin ground truth is present. Demand is recorded sales, not latent demand.

## Scenario design

Cross the V0.48 16 stress scenarios with two weekly capital-charge assumptions:

- 0.0% per week: no capital charge, an ablation;
- 0.5% per week: capital-pressure stress assumption.

The existing lead-time distributions, budget fractions, normalized capacity limits, pack-size rule and balanced/capital-pressure holding-cost coefficients are retained. These are declared assumptions, not estimated enterprise parameters.

## Policies

Eight configurations are compared:

1. q50 proportional;
2. q75 proportional;
3. q50/q75/q90 V0.48 average-benefit greedy;
4. q50/q75/q90 V0.49 sequential marginal-lot greedy.

For each lot, V0.49 estimates shortage benefit, excess holding cost, first-lot setup cost and capital exposure. After each accepted lot, the next lot's marginal value is re-evaluated. The allocator ranks feasible lots by incremental value divided by normalized budget-plus-capacity burden.

The capital charge in simulation is applied weekly to the selling-price-value proxy of average opening/closing on-hand plus pipeline inventory. The action score estimates capital exposure as assumed lead time plus a within-period consumption and excess-stock dwell approximation. Both use a selling-price proxy, not acquisition cost or measured financing expense.

## Selection rule

First select the lower-cost q50/q75 proportional baseline on selection-period mean proxy cost. Set the service floor to that baseline's selection fill rate minus 0.5 percentage points. Among all non-proportional policies that meet the floor, select the lowest-cost policy. If none qualifies, retain the proportional baseline and mark the candidate family infeasible.

Report selection-floor feasibility and test-period service-floor adherence separately. No test metric is used to select a policy.

## Exact allocation reference

For two sampled test weeks per scenario, SciPy MILP solves the same one-period per-lot objective on a deterministic random sample of up to 10 active series, capped at two lots per series. Budget and capacity limits are scaled by the sample share of requested spend and coverage. Binary lot variables have budget/capacity constraints and prefix constraints (a later lot cannot be selected without earlier lots). The solution is exact only for this bounded sampled subproblem, not the full portfolio or multi-period inventory process.

## Primary and secondary metrics

- primary: test mean total proxy cost per series-week, including holding, lost recorded sales, setup and capital-charge proxies;
- service: fill rate against recorded sales;
- working capital: average selling-price-value proxy tied up in on-hand plus pipeline inventory;
- operational: budget utilization, normalized capacity utilization, lost recorded units;
- optimization: one-period marginal-greedy objective gap to exact MILP.

Use paired replications/common random numbers and a descriptive circular four-week block bootstrap over the 16 test weeks. Intervals are descriptive and do not establish generalization.

## Evidence boundaries

A positive result is evidence only for this frozen recorded-sales replay and the declared scenario assumptions. It is not evidence of latent-demand accuracy, actual supplier reliability, actual cash/warehouse constraints, realized profit, or cross-retailer generalization. External replication with aligned demand, stockout, inventory, PO, receipt and acquisition-cost records remains necessary.