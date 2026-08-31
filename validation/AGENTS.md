# AGENTS — active_torchference/validation

Status: REAL (doc fleet, 2026-08-30)

Explicit numerical validation of free-energy computations.

## Layout
- `validate_vfe_computation.py` — variational free energy checks.
- `validate_efe_computation.py` — expected free energy checks.
- `outputs/` — GENERATED validation artifacts (timestamped JSON/TXT runs, PNG plots).

## Invariants
- `outputs/` is generated; rerun the validators to refresh.
