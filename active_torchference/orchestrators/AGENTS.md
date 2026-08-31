# AGENTS — active_torchference/active_torchference/orchestrators

Status: REAL (doc fleet, 2026-08-30)

Experiment orchestration for active_torchference.

## Layout
- `runner.py` — runs an experiment loop (agent in environment), collecting per-episode data.
- `logger.py` — run logging.
- `visualizer.py` — generates summary/trajectory/belief/free-energy plots (PNGs under `output/<run>/visualizations/`).
- `animator.py` — animation rendering.
- `__init__.py` — exports.

## Gotchas
- Orchestrators own the `output/<run_name>/config/data/visualizations` directory shape; changing it breaks downstream readers of those artifacts.
