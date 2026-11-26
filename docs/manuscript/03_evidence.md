## Numerical evidence

The native CPU VFE validator checks that the implementation returns
per-step diagnostics, that latent dimensions respond, and that the
likelihood and complexity terms sum to the displayed objective. It also
checks iterative minimization on a seeded synthetic example. The EFE
validator checks candidate and timestep axes, distinct candidate scores,
and the effects of changing the preference and uncertainty weights.
The regenerated plots visualize those diagnostics.

![VFE diagnostics from the seeded CPU validation run. Panels show values by timestep, within-step iterations, latent state evolution, and within-step reduction.](../../output/figures/vfe_validation.png){#fig:vfe_validation}

![EFE diagnostics from the seeded CPU validation run. Panels show per-policy cost, horizon costs, optimization history and the candidate cost distribution.](../../output/figures/efe_validation.png){#fig:efe_validation}

The stored file `output/simple_navigation/data/episode_0000.json` is a
historical navigation example [@torchference]. Its episode diagnostics
are shown below, with the means rounded to four decimal places. The
two validation plots come from separate CPU runs.

| Logged field | Archived value |
| --- | ---: |
| Steps | 100 |
| Mean VFE | 46.1019 |
| Mean EFE | 11.2762 |

The row values describe that recorded trajectory. In particular, an episode
termination flag alone does not establish why the run ended.
