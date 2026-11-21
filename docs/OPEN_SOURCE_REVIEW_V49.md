# V0.49 open-source reuse review

- Reuses V0.48's prepared UCI panel, chronological feature pipeline, stochastic lead-time calibration, baseline proportional allocator, V0.48 average-greedy allocator and common-random-number evaluation structure.
- Reuses `scipy.optimize.milp` for an exact one-period reference, adding binary lot variables and precedence constraints so later lots cannot be selected before earlier lots.
- No new optimizer, data parser or forecasting library is introduced.
- This is not a claim that the V0.49 heuristic is novel relative to the inventory-optimization literature. The contribution under test is a controlled comparison within this project's declared recorded-sales replay and constraints.
- No public synthetic purchase-order sample is treated as real supplier evidence. The supplier adapter remains a schema/readiness component only.