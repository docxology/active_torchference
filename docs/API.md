# API reference

The following interfaces are available from `active_torchference`.

| Interface | Use |
| --- | --- |
| `Config(...)` | Set hidden, observation and action dimensions, device, dtype, seed, and inference settings. |
| `ActiveInferenceAgent(config)` | Use `perceive(observation)`, `plan(preferred_obs)`, and `act(deterministic=False)` separately, or call `step(observation, preferred_obs, deterministic=False)` for the sequence. |
| `Environment` | Implement `reset()`, `step(action)`, and `get_preferred_observation()`. The bundled environments are continuous, grid, and oscillator simulations. |
| `BeliefState(config)` | Hold the Gaussian belief mean, log standard deviation, and prior. |
| `VariationalFreeEnergy`, `ExpectedFreeEnergy` | Evaluate the objective and its component diagnostics. |
| `PolicyEvaluator` | Produce per-candidate and per-timestep scores through model rollouts. |
| `OutputManager` | Organize configuration, metrics, episode data, plots, animations, and checkpoints. |

[CPU quick start](QUICKSTART.md) shows a complete call sequence.
The [update guide](BELIEF_AND_POLICY_UPDATES.md) specifies the scoring shapes
and the meaning of the diagnostic minimum.
