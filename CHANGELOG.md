# Changelog

## V0.50 — Walk-forward selector research (implementation; validation pending)
- Added five expanding-origin folds with fold-specific forecast refitting and lead-time residual calibration.
- Added causal historical-feasibility selector requiring at least 80% prior forward folds to meet the service tolerance.
- Added conservative selector using a one-sided 90% t lower confidence bound and a predeclared service-dispersion abstention rule.
- Added baseline-only fallback, fold-level replication exports, frozen final 16-week holdout evaluation, and independent audit.
- Remote 20-replication benchmark and independent audit are pending; no empirical improvement is claimed.

## V0.49 — Experimental research branch (validated; not released)
- Added sequential per-lot marginal value re-ranking and an explicit weekly working-capital exposure proxy.
- Added selection-period service-floor constrained policy choice with baseline fallback when no candidate qualifies.
- Added prefix-constrained one-period MILP comparison on bounded random SKU/lot samples.
- Added independent inventory/capital/service-floor audit and a dedicated benchmark workflow.
- Local validation: 177/177 tests passed; 32-scenario × 20-replication benchmark and independent audit passed. Remote GitHub Actions reproduction passed (run 44874903283).
- This remains an experimental branch, not a release; the new allocator did not show a general win over V0.48.

## V0.48 — Stochastic lead time and constrained replenishment
- Added 1–4 week stochastic lead-time mixture calibration using train/calibration-only residual windows.
- Added portfolio-wide budget and normalized capacity constraints.
- Added per-series MOQ/pack multiples and proportional / marginal-value greedy allocation.
- Added 128 exact one-period integer-lot MILP reference snapshots.
- Added supplier purchase-order import and lead-time distribution adapter with temporal cutoff and sparse-supplier shrinkage.
- Added 16 scenario × 20 replication historical recorded-sales replay and independent audit.
- 168 automated tests pass. Results remain scenario assumptions and recorded-sales proxy evidence, not measured company profit.

## V0.47 — Multi-period inventory policy replay
- Added deterministic lead-time inventory state machine, base-stock and (s,S) policies, holding/lost-sales/setup proxy costs.