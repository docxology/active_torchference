# AGENTS — active_torchference/examples

Status: REAL (doc fleet, 2026-08-30)

Runnable demo scripts, each writing artifacts to `output/<name>/` via the orchestrators.

## Scripts
- [`simple_navigation.py`](simple_navigation.py) — continuous navigation.
- [`gridworld_exploration.py`](gridworld_exploration.py) — grid-world exploration.
- `epistemic_pragmatic_balance.py` — EFE epistemic vs pragmatic weight sweep (`output/epistemic_pragmatic_comparison/`).
- [`custom_environment.py`](custom_environment.py) — oscillator tracking with
  `OscillatorEnvironment`. The example writes to `output/oscillator_tracking/`.
- `unified_output_example.py` — demonstrates `output_manager.py` usage.
- `complete_analysis_demo.py`, `comprehensive_fixes_demo.py` — broader end-to-end demos.

## Gotchas
- Running a script creates/overwrites its `output/<name>/` directory — treat those trees as disposable.
