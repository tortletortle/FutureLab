# Changelog

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
