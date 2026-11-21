# FutureLab V0.49 — capital-aware marginal lots and service-floor selection

## 1. Executive summary

V0.49 was implemented as an isolated research branch, preserving V0.48. The full local benchmark used 32 scenarios (the 16 V0.48 stress cases crossed with two capital-charge assumptions), eight policies and 20 common-random-number replications: **5,120 policy simulations**. The independent audit passed.

The result is deliberately mixed:

- The service-floor rule found a non-proportional policy meeting the selection-period floor in only **6/32** scenarios. In the other 26, it correctly fell back to the proportional baseline.
- The selected candidate had lower test proxy cost in **3/32** scenarios. Mean relative cost reduction was **0.79%**, but the median was **0.00%** because most scenarios fell back to baseline.
- Only **1/32** descriptive block-bootstrap intervals wholly favored the selected candidate; 31 crossed zero and none wholly favored the baseline.
- Candidate test fill rate was non-inferior to the baseline within 0.5 percentage points in **32/32** scenarios. Mean fill-rate difference was +1.23 pp, driven by the six scenarios in which a higher-service candidate was selected; fallback scenarios are equal by construction.
- The absolute selection-period service floor was not reached in the test period by either the candidate or the baseline in any scenario. This is a selection-to-test level shift, not a candidate-only failure.
- At fixed q75, the new marginal-lot allocator did **not** beat the V0.48 average-greedy allocator: it won 8/32 scenarios, with mean proxy-cost reduction **−0.11%** and mean fill-rate difference **−0.94 pp**. No bootstrap interval favored the new allocator; one favored the old allocator and 31 crossed zero.

Therefore V0.49 does **not** establish a general algorithmic improvement. The clearest engineering gain is a service-aware selection guard that avoids choosing a lower-cost policy when its selection-period fill rate is materially worse than the proportional baseline.

## 2. Experimental design

The benchmark reused the hash-pinned UCI Online Retail II prepared panel and the V0.48 chronological train/calibration/selection/test protocol (54/11/10/16 weeks; 100 product-country series). Forecast fitting used train only, stochastic lead-time residual calibration used calibration only, policy choice used selection only, and test remained frozen.

The 16 existing lead-time, budget, capacity and economic stress scenarios were crossed with two weekly capital-charge assumptions: 0.0% and 0.5% of the selling-price-value proxy. All 8 policies received common lead-time draws within each scenario and replication.

V0.49 added:

1. Sequential per-lot marginal valuation, including first-lot setup cost, estimated excess holding and capital exposure.
2. A weekly capital charge on the selling-price-value proxy of average opening/closing on-hand plus pipeline inventory.
3. A selection rule that accepts a non-proportional candidate only when its selection-period fill rate is within 0.5 percentage points of the proportional baseline; otherwise the baseline is retained.
4. A prefix-constrained exact one-period MILP reference on a deterministic random sample of up to 10 active series and two lots per series, with resource limits scaled by the sample share.

All capital, lead-time, budget, capacity, lot-size and economic parameters remain scenario assumptions. The panel contains recorded sales, not latent demand or stockout labels.

## 3. Main results

### 3.1 Service-constrained policy selection

| Measure | Result |
|---|---:|
| Scenarios | 32 |
| Selection-floor feasible scenarios | 6 / 32 |
| Scenarios falling back to proportional baseline | 26 / 32 |
| Selected q90 marginal-lot policies | 5 |
| Selected q90 average-greedy policies | 1 |
| Candidate test-cost point wins | 3 / 32 |
| Mean / median relative test-cost reduction | +0.79% / 0.00% |
| Bootstrap intervals wholly favoring candidate | 1 / 32 |
| Bootstrap intervals crossing zero | 31 / 32 |
| Bootstrap intervals wholly favoring baseline | 0 / 32 |
| Mean candidate-minus-baseline test fill-rate difference | +1.23 pp |
| Test fill-rate non-inferiority within 0.5 pp | 32 / 32 |

Among the six scenarios where a non-proportional candidate passed the selection-period service floor, three had lower test proxy cost. Their mean relative cost reduction was 4.23%, with a median of 0.20%; the other three did not improve cost. The six selected candidates also had a mean test fill-rate advantage of 6.57 pp over the baseline.

The absolute selection-period fill floor was not reached in test by either baseline or candidate in any scenario. The relevant positive result is narrower: candidate service did not fall more than 0.5 pp below baseline in any test scenario. The floor itself did not transfer across the selection/test boundary, indicating temporal instability in the recorded-sales service proxy.

### 3.2 Fixed-q75 allocator ablation

| Contrast | Point wins | Mean cost reduction | Median cost reduction | Mean fill-rate difference |
|---|---:|---:|---:|---:|
| V0.49 marginal-lot vs V0.48 average-greedy | 8 / 32 | −0.11% | −0.085% | −0.94 pp |
| V0.49 marginal-lot vs proportional | 9 / 32 | −1.75% | −2.80% | −5.75 pp |

For marginal-lot vs average-greedy, the paired four-week block-bootstrap intervals wholly favored the new allocator in 0/32 scenarios, favored the old allocator in 1/32, and crossed zero in 31/32. This does not support the claim that sequential marginal valuation improves the allocation rule in the current replay.

### 3.3 Capital sensitivity

The selected-candidate mean relative cost reduction was +1.32% at a zero capital charge and +0.27% at the 0.5%-per-week capital stress rate. Across economic assumptions, balanced scenarios averaged +2.28%, while capital-pressure scenarios averaged −0.69%.

This is consistent with the V0.48 finding that the candidate policy family is sensitive to capital occupancy. It is not an estimate of a real firm's cost of capital: the charge is calculated from selling-price proxy values, not supplier acquisition costs or observed financing rates.

### 3.4 One-period MILP comparison

The independent audit verified 32/32 sampled MILP subproblems reached optimal status. The median marginal-greedy objective gap was **0.21%** and the maximum was **3.89%**.

Each subproblem was restricted to a deterministic random sample of at most 10 active series and two lots per series. It is exact only for that bounded one-period linearized subproblem. It does not establish full-portfolio or multi-period optimality.

## 4. Engineering validation

- **177/177 automated tests passed** locally (168 inherited tests plus 9 V0.49 tests).
- **Independent audit passed** across 32 scenarios and 64 selected policy-role records.
- **166,400** saved selected-policy SKU-week decision rows were checked.
- Recomputed budget and normalized capacity constraints, MOQ/pack multiples, request bounds, demand balance, on-hand/pipeline conservation, shipment arrival timing, capital exposure, capital charge and full cost decomposition all passed.
- Terminal pipeline clearance and the final four-week no-new-order horizon passed.
- All 32 bounded MILP snapshots solved and passed objective/feasibility checks.

The remote GitHub Actions reproduction also passed: 177 tests, data-hash checks, the full 32-scenario × 20-replication benchmark, independent audit and artifact upload. Run 44874903283: https://github.com/tortletortle/FutureLab/actions/runs/44874903283.

## 5. Evidence limitations

1. Online Retail II has no on-hand inventory, stockout, supplier, purchase-order, receipt, acquisition-cost or gross-margin ground truth.
2. Recorded sales may be censored by stockouts; zero recorded sales do not prove zero latent demand.
3. The capital charge uses selling price as a value proxy and is not actual working-capital expense.
4. Lead-time distributions, budgets, normalized capacity, MOQ/pack rules and cost coefficients are assumed stress cases.
5. The service floor is calibrated relative to a baseline on a 10-week selection split and did not transfer as an absolute floor to the 16-week test split.
6. The 32 scenarios share one retailer and one time window; they are not independent real-world replications.
7. The MILP audit is limited to small sampled one-period subproblems.

## 6. Conclusion and next work

V0.49 should remain an experimental research branch, not replace V0.48. The service-aware fallback is worth retaining as a safety mechanism, but the new marginal-lot allocator did not outperform the old average-greedy rule under a fixed q75 target. The cost results are weak and mostly statistically inconclusive.

The next research step should not be another score-weight tweak. It should address the evidence bottleneck: obtain a dataset with aligned demand, stockout/on-hand, replenishment, supplier receipt and acquisition-cost records, or design a clearly separated synthetic stress suite for software verification only. In parallel, test service-floor calibration under temporal drift and evaluate a rolling/robust service constraint against a fixed baseline. No enterprise savings or supplier-generalization claim is justified yet.