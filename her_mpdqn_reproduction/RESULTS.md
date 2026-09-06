# Bounded reproduction results

> **Paper-fidelity v2 warning (2026-09-05):** all tables on this page predate
> the algorithm-specific observation interfaces and CATCH-consistent relay HER.
> They are diagnostic history, not final reproduction evidence. In particular,
> the old P-DQN/MP-DQN runs received `desired_goal` and explicit phase values
> that the published non-HER baselines do not receive. New v2 full results have
> not yet been generated.

> **Historical warning:** the original 5,000-step tables below used terminal
> out-of-bounds handling with reward `-1`. That allowed an agent to improve
> return by crashing early instead of accumulating `-1` for 100 steps. Those
> numbers are retained for provenance but must not be used as benchmark
> evidence. The default environment now clips attempted boundary crossings and
> continues the episode.

## Boundary-fix validation (seed 0, 40,000 steps)

This validation covered Direct/Relay × all four algorithms, used 50
deterministic evaluation episodes per run, and completed in 35 minutes.

| Environment | Algorithm | Eval success | Boundary hit | Mean length |
|---|---:|---:|---:|---:|
| Direct | P-DQN | 96% | 6% | 42.1 |
| Direct | MP-DQN | 100% | 10% | 38.5 |
| Direct | HER-PDQN | 98% | 6% | 38.7 |
| Direct | HER-MPDQN | 98% | 12% | 37.8 |
| Relay | P-DQN | 0% | 4% | 100.0 |
| Relay | MP-DQN | 0% | 58% | 100.0 |
| Relay | HER-PDQN | 0% | 48% | 100.0 |
| Relay | HER-MPDQN | **2%** | 40% | 99.3 |

No evaluation episode terminated out of bounds. In the last 100 training
episodes, Relay HER-MPDQN reached the relay in 85% and completed the final goal
in 38%; the other three methods reached the relay in only about 6% and had zero
final successes. Phase-aware HER-MPDQN filtered 345,167 invalid cross-phase
future-goal candidates. The large training-versus-deterministic-evaluation gap
means more seeds and longer training remain necessary, but the corrected task
now exhibits the intended qualitative learning signal.

These are sanity-scale behavioral results, not the paper's full training
budget. Every run used exactly 5,000 environment steps, three seeds (0, 1, 2),
and 30 deterministic evaluation episodes per seed. Values are mean ± population
standard deviation across seeds.

## Direct and relay benchmark

| Environment | Algorithm | Success rate | Mean return |
|---|---:|---:|---:|
| Direct | P-DQN | 2.2% ± 1.6% | -27.0 |
| Direct | MP-DQN | 1.1% ± 1.6% | -26.0 |
| Direct | HER-PDQN | 3.3% ± 4.7% | -70.2 |
| Direct | HER-MPDQN | **24.4% ± 17.8%** | -46.1 |
| Relay | P-DQN | 0.0% ± 0.0% | -40.2 |
| Relay | MP-DQN | 0.0% ± 0.0% | -39.4 |
| Relay | HER-PDQN | 0.0% ± 0.0% | -100.0 |
| Relay | HER-MPDQN | 0.0% ± 0.0% | -61.5 |

Direct reproduces the main qualitative signal that HER-MPDQN is strongest,
but not the complete requested middle ordering: at this small budget P-DQN and
MP-DQN are statistically indistinguishable, and HER-PDQN is highly variable.
Relay confirms that sparse two-stage navigation is substantially harder, but
does **not** establish the paper's claimed HER-MPDQN advantage because no method
completed the full relay task in evaluation. Longer training is required; the
result is intentionally reported rather than tuned until it matches.

## Long-horizon extension (HER-MPDQN)

| Relays | Phases | Final success | Reached at least one relay | Mean return |
|---:|---:|---:|---:|---:|
| 0 | 1 | 5.6% ± 5.7% | 0.0% | -51.7 |
| 1 | 2 | 0.0% ± 0.0% | 5.6% | -79.7 |
| 2 | 3 | 0.0% ± 0.0% | 21.1% | -110.1 |
| 4 | 5 | 0.0% ± 0.0% | 2.2% | -224.6 |
| 8 | 9 | 0.0% ± 0.0% | 7.8% | -254.3 |

Only zero-relay episodes show final success at this budget. Mean return is not
directly normalized across horizons because maximum episode length scales with
the number of phases; final success rate is the primary comparison metric.

Raw runs, losses, HER counts and aggregate JSON/CSV are under
`outputs/reproduction_3seed_5000` and `outputs/horizon_3seed_5000`.
