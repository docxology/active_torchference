# AGENTS — active_torchference/output

Status: REAL (doc fleet, 2026-08-30)

GENERATED experiment artifacts for active_torchference, written by `output_manager.py`
and the orchestrators when examples run. One subdirectory per demo run, plus summary
PNGs at this level (e.g. `epistemic_pragmatic_comparison/comparison_summary.png`).

## Layout
- `simple_navigation/`, `gridworld_exploration/`, `oscillator_tracking/` — per-demo runs.
- `epistemic_pragmatic_comparison/` — sweep across three balance settings.
- `unified_demo/` — output-manager demo run.

Each run directory has `config/config.json`, `data/episode_NNNN.json`, and (for demos
with plots) `visualizations/*.png`. Do not hand-edit; rerun the example to regenerate.
