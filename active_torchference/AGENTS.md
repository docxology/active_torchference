# AGENTS — active_torchference/active_torchference

Status: REAL (doc fleet, 2026-08-30)

Core package for the PyTorch Active Inference framework.

## Layout
- `config.py` — dataclass configuration (`Config`) for agents/environments.
- `beliefs.py` — `BeliefState` and `GenerativeModel`.
- `agent.py` — the action-perception loop agent tying beliefs, policy, free energy together.
- `free_energy.py` — `VariationalFreeEnergy` and expected-free-energy computations.
- `policy.py` — policy evaluation/selection (softmax over EFE).
- `environment.py` — abstract `Environment` base plus concrete implementations.
- `output_manager.py` — unified writer for all experiment artifacts (config/data/visualizations under `output/`).
- `utils.py` — tensor helpers, Gaussian log-likelihood, KL divergence.
- `orchestrators/` — runner, logger, animator, visualizer (experiment orchestration).

## Invariants
- All experiment artifacts flow through `output_manager.py` — do not write to `output/` ad hoc elsewhere.
