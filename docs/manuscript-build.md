# Build the Active Torchference methods note

Install Git, uv, Pandoc 3.14 with pandoc-crossref 0.3.26b, and XeLaTeX with the
[required TeX packages](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/usage/output-formats.md).
Use the [CPU setup](QUICKSTART.md) in the clean project checkout, or provide a
compatible Python interpreter with the project dependencies and CPU Torch.

From the project root:

```sh
set -e
project_root="$(pwd -P)"
project_python="$project_root/.venv/bin/python"
renderer="$(mktemp -d)/template"
git clone https://github.com/docxology/template.git "$renderer"
git -C "$renderer" checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
mkdir -p "$renderer/projects/working"
project_copy="$renderer/projects/working/active_torchference"
git clone --no-hardlinks "$project_root" "$project_copy"
git -C "$project_copy" checkout --detach "$(git -C "$project_root" rev-parse HEAD)"

cd "$project_copy"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$project_python" -m pytest -q
"$project_python" validation/validate_vfe_computation.py
"$project_python" validation/validate_efe_computation.py
mkdir -p output/figures
cp validation/outputs/vfe_validation.png validation/outputs/efe_validation.png output/figures/
cp docs/manuscript/figure_registry.json output/figures/figure_registry.json

cd "$renderer"
uv sync --locked --no-dev
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/active_torchference
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/active_torchference
```

Inspect `output/pdf/active_torchference_combined.pdf`, `output/web/index.html`,
and `output/reports/` in the disposable copy. The render includes six
sections, the regenerated CPU plots, and data from an archived episode.
