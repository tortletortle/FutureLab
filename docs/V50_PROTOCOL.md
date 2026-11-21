# FutureLab V0.50 Protocol — walk-forward policy selection under temporal drift

## Status

Research protocol only. V0.49 remains an unreleased research branch and V0.48 remains the current release baseline. No V0.50 empirical result is claimed until implementation, tests, benchmark and independent audit complete.

## Motivation

V0.49 reproduced successfully in GitHub Actions: 32 scenarios × 20 common-random-number replications, 10,240 policy/split/replication records, 166,400 independently audited decision rows and 32/32 successful bounded MILP snapshots. The independent audit passed.

However, only 6/32 scenarios had any non-proportional candidate satisfying the selection-period service floor. The candidate therefore fell back to the proportional baseline in 26/32 scenarios. In the six selected-candidate scenarios there were three cost wins and three losses. The selection-period service floor was met in 0/32 frozen test scenarios, even though candidate service was non-inferior to the baseline in 32/32. These are evidence of temporal selection instability, not proof that a particular alternative selector will fix it.

## Research question

Can a policy selector calibrated on multiple chronological pseudo-out-of-sample windows improve policy-selection stability and cost/service trade-offs, without using the final frozen test period for selection?

## Hypotheses

- **H1 — Stability:** walk-forward selection will reduce the rate at which a policy selected on one short window materially degrades on the next unseen window, relative to single-window selection.
- **H2 — Service robustness:** a service constraint calibrated from prior walk-forward evaluation windows will have better out-of-window adherence than a single selection-window point estimate.
- **H3 — Cost:** stability constraints may reduce apparent cost gains; any benefit must be measured against the V0.48 proportional baseline and V0.49 selector.
- **H4 — No free lunch:** if rolling windows disagree materially, the method should abstain or fall back rather than manufacture a single confident winner.

These are hypotheses, not expected outcomes.

## Frozen data boundary

Reuse the hash-pinned V0.48 prepared product-country weekly recorded-sales panel and its quality manifest. Preserve the existing final chronological test block (weeks 75–90 under the 91-week panel indexing). The final test must not influence fold construction, selector thresholds, policy choice, or hyperparameters.

Recorded sales may be censored by stockouts. No ground truth for on-hand inventory, latent demand, purchase orders, supplier receipts, acquisition cost, gross margin or financing cost is available. Scenario costs, lead times, budget, capacity and capital rates remain declared assumptions.

## Walk-forward design

Construct five expanding-origin folds wholly before the frozen final test:

| Fold | Forecast fit history | Calibration window | Policy selection window | Forward evaluation window |
|---|---:|---:|---:|---:|
| 1 | 30 weeks | 8 weeks | 6 weeks | next 6 weeks |
| 2 | 36 weeks | 8 weeks | 6 weeks | next 6 weeks |
| 3 | 42 weeks | 8 weeks | 6 weeks | next 6 weeks |
| 4 | 48 weeks | 8 weeks | 6 weeks | next 6 weeks |
| 5 | 54 weeks | 8 weeks | 6 weeks | next 6 weeks |

Origins advance by six weeks. Each fold must refit forecasts on its fit history only, calibrate lead-time residuals on its calibration window only, select policies on its selection window only, and evaluate the selected policy on the immediately following six-week window. No future rows may enter feature construction, residual calibration, policy selection or inventory state initialization.

The fold schedule, exact calendar boundaries, row counts, source hashes and leakage checks must be written to a manifest before running candidate comparisons. If the panel calendar cannot support the declared schedule exactly, fail closed and revise the protocol before looking at any outcome metrics.

## Candidate selectors

Compare, at minimum:

1. V0.49 single-window selector (baseline reproduction).
2. Robust-feasibility selector: require at least three completed prior forward folds before selecting a candidate; then it must satisfy the service floor in at least max(3, ceil(0.8 × prior-fold-count)) folds. On the final holdout this means at least four of five folds; before three folds, abstain to the proportional baseline.
3. Conservative selector: rank only policies meeting a predeclared lower confidence bound on service across folds, then minimize mean proxy cost; report fold dispersion and worst-fold regret.
4. Baseline-only fallback.

All selectors use identical policy candidates, scenarios, random seeds and common random numbers. No selector may inspect the final test block.

## Service calibration

Do not silently reuse the V0.49 fixed 0.5 percentage-point tolerance as if it were empirically calibrated. Compare predeclared alternatives, including:

- absolute service tolerance versus the proportional baseline;
- a fold-derived lower service bound based only on prior forward-evaluation folds;
- a minimum of three completed forward folds before the robust selector may leave the baseline;
- abstention when fold service estimates are too dispersed.

Thresholds must be selected using walk-forward folds only, frozen, and then applied once to the final test. Report both selection-window and forward-evaluation adherence. Do not claim a service guarantee from a finite sample.

## Metrics

Primary:
- paired total proxy-cost difference versus proportional baseline;
- forward-evaluation service difference versus baseline;
- frequency of selector abstention/fallback;
- worst-fold cost regret and service degradation.

Secondary:
- policy-selection frequency and stability across folds;
- budget and capacity utilization;
- lost recorded units;
- working-capital proxy exposure;
- performance by lead-time, capital and resource-stress scenario;
- descriptive block-bootstrap intervals, with their limited temporal scope stated.

The final frozen test report must include every scenario, not only favorable subsets. Report point estimates, intervals, denominator counts, fold variation and failed/abstained cases.

## MILP scope

Retain the V0.49 prefix-constrained one-period MILP only as a bounded local allocator reference. It is not a global multi-period optimum and must not be presented as one.

## Acceptance gates

- Reproduce V0.49 benchmark outputs from the pinned source before changing the selector.
- Add unit tests for fold boundaries, leakage prevention, missing-week handling, selector abstention, service-bound calculations and deterministic replay.
- Add an independent audit that recomputes fold membership, selected policy decisions, service constraints and forward-evaluation metrics from raw evidence.
- Verify final test rows are absent from all selection inputs by hash and row-level audit.
- Run the full paired benchmark with at least 20 replications and common random numbers.
- Publish full fold-level and scenario-level evidence, including negative results.
- If the walk-forward selector does not improve stability without unacceptable cost/service regression, retain the V0.49/V0.48 fallback and document the negative result.

## Interpretation boundary

A positive result is evidence only for this frozen recorded-sales replay and declared assumptions. It is not evidence of latent-demand accuracy, supplier reliability, actual working-capital savings, realized profit, or cross-retailer generalization. External replication with aligned inventory, stockout, purchase-order, receipt and acquisition-cost records remains necessary.
