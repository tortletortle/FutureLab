# FutureLab V0.49 — capital-aware marginal lots and service-floor selection

## Status

Research implementation and local validation are in progress. Do not treat this document as a completed empirical result until the reproducible benchmark and independent audit finish on the V0.49 branch.

## Motivation from V0.48

The V0.48 frozen test reported candidate-family point-cost wins in 8/16 scenarios, but only 3 descriptive intervals wholly favored the candidate and 1 wholly favored the baseline. The median relative proxy-cost reduction was 0.07%, while the mean was 2.65%. All eight capital-pressure scenarios had higher candidate point cost. The candidate fill rate averaged 54.05%, versus 57.36% for the baseline. Fixed-q75 greedy allocation increased proxy cost by 1.24% on average and won only 5/16 scenarios.

This evidence does not support simply expanding the V0.48 greedy allocator.

## V0.49 changes

- Re-score each additional order lot using its own incremental shortage benefit, excess holding exposure, first-lot setup cost and estimated capital exposure.
- Add a weekly capital-charge proxy to the simulation, calculated from average opening/closing on-hand plus pipeline inventory.
- Select a candidate policy only if its selection-period fill rate is within 0.5 percentage points of the proportional baseline; otherwise retain the baseline.
- Compare the sequential marginal-lot heuristic with a prefix-constrained exact one-period MILP.
- Preserve V0.48 code and results unchanged.

## Validation

Pending full 20-replication benchmark, independent state-transition audit and GitHub Actions run. See `V49_PROTOCOL.md` for the frozen protocol and evidence limitations.