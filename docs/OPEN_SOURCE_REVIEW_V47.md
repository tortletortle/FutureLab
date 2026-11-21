# Open-source reuse review — V0.47

## Stockpyl

- Repository: https://github.com/LarrySnyder/stockpyl
- Documentation: https://stockpyl.readthedocs.io/en/latest/
- Version reviewed: 1.0.2
- License: MIT; the upstream repository includes its license text.
- Scope: EOQ, newsvendor, Wagner–Whitin, `(r,Q)`, `(s,S)`, base-stock and multi-echelon inventory optimization; discrete-time simulation supports multiple products, custom demand sources, inventory policies, supply disruptions, costs and period-level state output.

## Reuse decision

Stockpyl is the established reference implementation for general inventory optimization and multi-echelon simulation. We did not copy or vendor its source. The V0.47 experiment needs a narrow deterministic replay adapter over the project's already audited product-country-week panel, with an explicit order-before-demand event sequence and a paired chronological policy-selection protocol. Stockpyl's documented simulator uses a different default sequence (customer orders are received before outbound orders are placed) and explains that lead times must be adjusted to compare against the more common order-first convention. A direct substitution without an event-order adapter would silently change the meaning of lead time.

The current runtime is offline and Stockpyl is not installed, so a package-level parity run was not possible here. The V0.47 simulator is intentionally limited to a single-echelon periodic-review replay; it does not claim to replace Stockpyl's EOQ, dynamic programming, stochastic-service or multi-echelon capabilities. A future integration should add a deterministic one-node parity test against Stockpyl before adopting its engine.

## Other implementation choices

- Reuse the project's V0.25 streaming UCI ingestion and V0.46 chronological feature builder; do not create a second workbook parser.
- Keep the historical V0.43–V0.45 Hillstrom causal-evaluation protocol untouched.
- Do not add SimPy or a second discrete-event framework: the current single-node deterministic lead-time state machine is periodic and small enough to test exhaustively. If the project expands to multiple suppliers, stochastic transit events or multi-echelon networks, revisit Stockpyl before extending this simulator.
