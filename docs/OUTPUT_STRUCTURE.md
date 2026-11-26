# Outputs and numerical validation

Bundled examples put per-run configuration, logs, episode data, and
visualizations into experiment subdirectories under `output/`.
The examples can also write animations and checkpoints according to their
configuration. Tracked artifacts describe earlier example runs.

From a disposable checkout, run the two native validators on CPU:

```sh
MPLBACKEND=Agg .venv/bin/python validation/validate_vfe_computation.py
MPLBACKEND=Agg .venv/bin/python validation/validate_efe_computation.py
```

They check the belief and policy computations and write
`validation/outputs/vfe_validation.png` and `validation/outputs/efe_validation.png`.
The [manuscript build](manuscript-build.md) stages those regenerated plots
for the renderer. The [methods note](manuscript/03_evidence.md) identifies
which results were recorded in the archived episode.
