# AGENTS — active_torchference/examples

Status: REAL (doc fleet, 2026-08-30)

Runnable demo scripts, each writing artifacts to `output/<name>/` via the orchestrators.

## Scripts
- `simple_navigation.py`, `gridworld_exploration.py` — discrete navigation demos.
- `epistemic_pragmatic_balance.py` — EFE epistemic vs pragmatic weight sweep (`output/epistemic_pragmatic_comparison/`).
- `custom_environment.py` — how to subclass `Environment`.
- `unified_output_example.py` — demonstrates `output_manager.py` usage.
- `complete_analysis_demo.py`, `comprehensive_fixes_demo.py` — broader end-to-end demos.
- Oscillator tracking artifacts exist as `output/oscillator_tracking/` (script name unverified on disk).

## Gotchas
- Running a script creates/overwrites its `output/<name>/` directory — treat those trees as disposable.
