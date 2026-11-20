# FutureLab V0.43 — Hillstrom real-data evaluation

**Result: primary acceptance criterion not met.** This is a one-shot static policy benchmark, not an end-to-end validation of the V0.42 dynamic evidence-routing system.

## Provenance and locked protocol
- Uploaded compressed dataset SHA-256: `bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f` (matches the hash frozen in the V0.43 protocol).
- 64,000 rows; No E-Mail 21,306; Mens E-Mail 21,307; Womens E-Mail 21,387.
- Evaluator and protocol are preserved at `migration/v0.48/src/reality_v43.py` and `migration/v0.48/docs/V43_PROTOCOL.md`.
- 60/20/20 stratified split, seed 431901; pre-treatment features only; visit/conversion excluded.
- Primary assumptions: 25% contact cap, 30% gross margin, $0.05 email cost. These are scenario assumptions, not measured enterprise costs or profit.

## Primary result: Mens E-Mail vs No E-Mail
Validation-selected baseline: `recent_rule`; FutureLab candidate: `futurelab_0.5`.
- Baseline net value: $0.013315 per eligible customer.
- FutureLab net value: $0.032576 per eligible customer.
- Paired difference: +$0.019262.
- Approximate 95% CI: [-$0.075788, +$0.114312].
- Relative point estimate: +144.67%, inflated by the small baseline estimate.
- **Fail:** CI lower bound is below zero, so positive advantage is not established.

At 10,000 eligible customers, the point difference is about $192.62, with an interval from about -$757.88 to +$1,143.12 under the stated margin/cost assumptions. This is estimated historical policy value, not realized profit.

## Secondary result: Womens E-Mail
- Baseline: -$0.045815; FutureLab: -$0.046474.
- Difference: -$0.000659; approximate 95% CI [-$0.082331, +$0.081013].
- Descriptive only; shares the No E-Mail control and is not an independent replication.

## Important limitations
- The test set was used once; do not tune against it. A new confirmatory claim requires a fresh independent randomized holdout or trial.
- Confidence intervals are approximate and conditional on fitted policies/test cohort; they do not cover model fitting/selection uncertainty, clustering, or transport to other businesses and periods.
- CausalForestDML was unavailable because `econml` was not installed. Thompson Sampling, UCB, dynamic drift, memory reuse and live deployment were not tested.
- Raw customer data is intentionally not committed to this repository.
