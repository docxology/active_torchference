# CPU quick start

From a repository checkout, use Python 3.12 and uv to create a local
environment with the CPU build of PyTorch:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
uv pip install --python .venv/bin/python -e .
```

The package supports Python 3.11 or later and PyTorch 2.0 or later. The
commands select the CPU build used to validate the manuscript.

```python
from active_torchference import ActiveInferenceAgent, Config, ContinuousEnvironment

config = Config(seed=42, device="cpu")
environment = ContinuousEnvironment(config)
agent = ActiveInferenceAgent(config)
observation = environment.reset()
action, info = agent.step(
    observation, environment.get_preferred_observation(), deterministic=True
)
next_observation, transition_info = environment.step(action)
print(action.shape, info["efe_per_policy"].shape, next_observation.shape)
```

`agent.step()` chooses an action and records the inference results.
`environment.step()` advances the simulated environment.

Run the tests from the repository root:

```sh
.venv/bin/python -m pytest -q
```

The [bundled examples](../examples/) include continuous navigation and
grid-world simulations. They write under `output/`. Use a disposable
checkout to preserve the tracked example snapshot. See the
[output guide](OUTPUT_STRUCTURE.md) for the two numerical validators.
