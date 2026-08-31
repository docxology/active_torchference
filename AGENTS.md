# AGENTS — active_torchference

Status: REAL (doc fleet, 2026-08-30)

`active_torchference` — a standalone PyTorch Active Inference framework (its own git
repo, nested inside the docxology workspace).

## Layout
- `active_torchference/` — the Python package (agent, beliefs, environment, free energy, policy, orchestrators).
- `examples/` — runnable demo scripts; write outputs under `output/<example_name>/`.
- `tests/` — pytest suite (unit + integration, incl. free-energy validation).
- `validation/` — explicit VFE/EFE validation scripts; artifacts in `validation/outputs/`.
- `output/` — generated experiment artifacts (per-example subdirs).
- `docs/manuscript/` — manuscript status for the publication line.
- Root: `pyproject.toml`, `requirements.txt`, `Makefile`, `pytest.ini`, `check_installation.py`.

## Commands (verify with pytest)
- `pytest` from repo root (config in `pytest.ini`).
- `python check_installation.py` — environment sanity check (unverified).

## Invariants
- Output artifacts under `output/` and `validation/outputs/` are GENERATED — rerun the examples/validators rather than editing.
