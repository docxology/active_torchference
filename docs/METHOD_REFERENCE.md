# Method Reference

Quick alphabetical reference for all public methods in Active Torchference.

## Table of Contents

- [ActiveInferenceAgent](#activeinfererenceagent)
- [BeliefState](#beliefstate)
- [Config](#config)
- [Environment Classes](#environment-classes)
- [ExpectedFreeEnergy](#expectedfreeenergy)
- [ExperimentRunner](#experimentrunner)
- [GenerativeModel](#generativemodel)
- [Logger](#logger)
- [OutputManager](#outputmanager)
- [PolicyEvaluator](#policyevaluator)
- [TransitionModel](#transitionmodel)
- [VariationalFreeEnergy](#variationalfreeenergy)
- [Visualizer](#visualizer)
- [Animator](#animator)
- [Utility Functions](#utility-functions)

---

## ActiveInferenceAgent

Main agent class implementing the action-perception loop.

### `__init__(config: Config)`
Initialize Active Inference agent with configuration.

**Parameters**:
- `config`: Configuration object

**Example**:
```python
agent = ActiveInferenceAgent(config)
```

---

### `perceive(observation: torch.Tensor, num_iterations: Optional[int] = None) -> Dict`
Update beliefs by minimizing VFE over multiple iterations.

**Parameters**:
- `observation`: Observation from environment
- `num_iterations`: Number of VFE minimization steps (uses config default if None)

**Returns**:
- Dictionary with:
  - `vfe`: Final VFE value
  - `vfe_initial`: Initial VFE
  - `vfe_final`: Final VFE
  - `vfe_history`: List of VFE values per iteration
  - `vfe_likelihood`: Likelihood component
  - `vfe_complexity`: Complexity component
  - `predicted_obs`: Predicted observation
  - `belief_mean`: Updated belief mean
  - `belief_std`: Updated belief std

**Example**:
```python
perception_info = agent.perceive(observation)
print(f"VFE: {perception_info['vfe']:.4f}")
```

---

### `plan(preferred_obs: torch.Tensor) -> Dict`
Evaluate policies using EFE and optimize policy distribution.

**Parameters**:
- `preferred_obs`: Goal/preferred observations

**Returns**:
- Dictionary with:
  - `efe`: Mean EFE across policies
  - `efe_per_policy`: EFE for each policy [num_rollouts]
  - `efe_per_policy_per_timestep`: EFE per policy per timestep [horizon, num_rollouts]
  - `efe_components`: Dict with epistemic and pragmatic values
  - `efe_history`: List of mean EFE per iteration
  - `best_policy_idx`: Index of best policy
  - `min_efe`: Minimum EFE value
  - `max_efe`: Maximum EFE value
  - `action_mean`: Updated action mean
  - `action_std`: Updated action std
  - `predicted_obs`: Predicted observations
  - `action_samples`: Sampled actions

**Example**:
```python
planning_info = agent.plan(preferred_obs)
print(f"Best policy: {planning_info['best_policy_idx']}")
```

---

### `act(deterministic: bool = False) -> torch.Tensor`
Select action from policy posterior.

**Parameters**:
- `deterministic`: If True, return mean action; if False, sample

**Returns**:
- Action tensor [action_dim]

**Example**:
```python
action = agent.act(deterministic=False)
```

---

### `step(observation: torch.Tensor, preferred_obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict]`
Execute full action-perception loop: perceive → plan → act.

**Parameters**:
- `observation`: Observation from environment
- `preferred_obs`: Goal/preferred observations
- `deterministic`: If True, use deterministic action selection

**Returns**:
- Tuple of (action, info_dict)
  - `action`: Selected action
  - `info_dict`: Combined perception and planning info

**Example**:
```python
action, info = agent.step(observation, preferred_obs)
print(f"VFE: {info['vfe']:.4f}, EFE: {info['efe']:.4f}")
```

---

### `reset()`
Reset agent to initial state (beliefs and policy).

**Example**:
```python
agent.reset()
```

---

### `get_history() -> Dict[str, list]`
Get agent history (observations, actions, beliefs, VFE, EFE).

**Returns**:
- Dictionary with lists of:
  - `observations`
  - `actions`
  - `beliefs`
  - `vfe`
  - `efe`

**Example**:
```python
history = agent.get_history()
print(f"Total steps: {len(history['observations'])}")
```

---

### `save_state() -> Dict`
Save agent state to dictionary.

**Returns**:
- Dictionary with beliefs, policy, and config

**Example**:
```python
state = agent.save_state()
```

---

### `load_state(state_dict: Dict)`
Load agent state from dictionary.

**Parameters**:
- `state_dict`: Dictionary with agent state

**Example**:
```python
agent.load_state(state_dict)
```

---

## BeliefState

Manages approximate posterior q(s) over hidden states.

### `__init__(config: Config)`
Initialize belief state.

---

### `sample(num_samples: int = 1) -> torch.Tensor`
Sample from belief distribution.

**Parameters**:
- `num_samples`: Number of samples

**Returns**:
- Samples [num_samples, hidden_dim]

**Example**:
```python
samples = beliefs.sample(num_samples=10)
```

---

### `update(gradient: Tuple[torch.Tensor, torch.Tensor], learning_rate: Optional[float] = None)`
Update beliefs via gradient descent.

**Parameters**:
- `gradient`: Tuple of (grad_mean, grad_log_std)
- `learning_rate`: Learning rate (uses config default if None)

**Example**:
```python
beliefs.update((grad_mean, grad_log_std), learning_rate=0.1)
```

---

### `reset()`
Reset beliefs to prior.

**Example**:
```python
beliefs.reset()
```

---

### `set_prior(mean: torch.Tensor, std: torch.Tensor)`
Set prior distribution parameters.

**Parameters**:
- `mean`: Prior mean
- `std`: Prior standard deviation

**Example**:
```python
beliefs.set_prior(torch.zeros(4), torch.ones(4))
```

---

### `detach() -> Tuple[torch.Tensor, torch.Tensor]`
Get detached copy of belief parameters.

**Returns**:
- Tuple of (mean, std) without gradients

**Example**:
```python
mean, std = beliefs.detach()
```

---

### `to_dict() -> dict`
Convert belief state to dictionary.

**Returns**:
- Dictionary with mean, log_std, prior_mean, prior_log_std

**Example**:
```python
belief_dict = beliefs.to_dict()
```

---

## Config

Configuration dataclass for all parameters.

### `__init__(...)`
Create configuration with all parameters.

**Key Parameters**:
- `hidden_dim`: Hidden state dimensionality
- `obs_dim`: Observation dimensionality
- `action_dim`: Action dimensionality
- `learning_rate_beliefs`: Belief update learning rate
- `learning_rate_policy`: Policy update learning rate
- `num_belief_iterations`: VFE minimization iterations
- `num_policy_iterations`: Policy optimization iterations
- `num_rollouts`: Number of policy rollouts
- `horizon`: Planning horizon
- `epistemic_weight`: Exploration weight
- `pragmatic_weight`: Exploitation weight
- `precision_obs`, `precision_prior`, `precision_state`: Precision parameters
- `device`: 'cpu' or 'cuda'
- `dtype`: torch.float32 or torch.float64
- `seed`: Random seed

**Example**:
```python
config = Config(
    hidden_dim=8,
    obs_dim=2,
    action_dim=2,
    epistemic_weight=1.0,
    pragmatic_weight=1.0
)
```

---

### `to_dict() -> Dict[str, Any]`
Convert configuration to dictionary.

**Returns**:
- Dictionary with all config parameters

**Example**:
```python
config_dict = config.to_dict()
```

---

### `from_dict(config_dict: Dict[str, Any]) -> Config`
Create configuration from dictionary (class method).

**Parameters**:
- `config_dict`: Dictionary with config parameters

**Returns**:
- Config object

**Example**:
```python
config = Config.from_dict(config_dict)
```

---

## Environment Classes

### Environment (Abstract Base)

#### `reset() -> torch.Tensor`
Reset environment to initial state.

**Returns**:
- Initial observation

---

#### `step(action: torch.Tensor) -> Tuple[torch.Tensor, dict]`
Execute action and return next observation.

**Parameters**:
- `action`: Action to execute

**Returns**:
- Tuple of (observation, info_dict)

---

#### `get_preferred_observation() -> torch.Tensor`
Return goal/preferred observation.

**Returns**:
- Goal observation tensor

---

### ContinuousEnvironment

Continuous 2D navigation environment.

#### `__init__(config: Config, goal_position: Optional[torch.Tensor] = None, noise_std: float = 0.1)`
Initialize continuous environment.

**Parameters**:
- `config`: Configuration
- `goal_position`: Target position (default: [1.0, 1.0])
- `noise_std`: Observation noise

**Example**:
```python
env = ContinuousEnvironment(config, goal_position=torch.tensor([2.0, 2.0]))
```

---

### GridWorld

Discrete grid world environment.

#### `__init__(config: Config, grid_size: int = 5, goal_position: Optional[Tuple[int, int]] = None)`
Initialize grid world.

**Parameters**:
- `config`: Configuration
- `grid_size`: Size of square grid
- `goal_position`: Goal (x, y) coordinates

**Example**:
```python
env = GridWorld(config, grid_size=5, goal_position=(4, 4))
```

---

### OscillatorEnvironment

Oscillating goal environment.

#### `__init__(config: Config, frequency: float = 0.1, amplitude: float = 2.0, noise_std: float = 0.1)`
Initialize oscillator environment.

**Parameters**:
- `config`: Configuration
- `frequency`: Oscillation frequency
- `amplitude`: Oscillation amplitude
- `noise_std`: Observation noise

**Example**:
```python
env = OscillatorEnvironment(config, frequency=0.05, amplitude=2.0)
```

---

## ExpectedFreeEnergy

Computes expected free energy for policy evaluation.

### `__init__(epistemic_weight: float = 1.0, pragmatic_weight: float = 1.0, precision_obs: float = 1.0)`
Initialize EFE calculator.

**Parameters**:
- `epistemic_weight`: Weight for exploration
- `pragmatic_weight`: Weight for exploitation
- `precision_obs`: Observation precision

---

### `compute(predicted_obs: torch.Tensor, predicted_obs_std: torch.Tensor, preferred_obs: torch.Tensor, state_entropy: torch.Tensor) -> Tuple[torch.Tensor, dict]`
Compute expected free energy.

**Parameters**:
- `predicted_obs`: Predicted observations
- `predicted_obs_std`: Observation uncertainty
- `preferred_obs`: Goal observations
- `state_entropy`: State entropy

**Returns**:
- Tuple of (efe, components_dict)

**Example**:
```python
efe, components = efe_calculator.compute(predicted_obs, predicted_obs_std, preferred_obs, state_entropy)
```

---

## ExperimentRunner

Orchestrates experiment execution.

### `__init__(agent, environment, logger, config)`
Initialize experiment runner.

**Parameters**:
- `agent`: ActiveInferenceAgent
- `environment`: Environment
- `logger`: Logger instance
- `config`: Config object

**Example**:
```python
runner = ExperimentRunner(agent, env, logger, config)
```

---

### `run(num_episodes, max_steps_per_episode, deterministic, render, save_every, verbose) -> Dict`
Run multiple episodes.

**Parameters**:
- `num_episodes`: Number of episodes
- `max_steps_per_episode`: Max steps per episode
- `deterministic`: Use deterministic actions
- `render`: Render environment
- `save_every`: Save checkpoint frequency
- `verbose`: Show progress

**Returns**:
- Results dictionary

**Example**:
```python
results = runner.run(num_episodes=10, max_steps_per_episode=100, verbose=True)
```

---

### `run_episode(max_steps, deterministic, render) -> Dict`
Run single episode.

**Parameters**:
- `max_steps`: Maximum steps
- `deterministic`: Use deterministic actions
- `render`: Render environment

**Returns**:
- Episode data dictionary

---

### `evaluate(num_episodes, max_steps_per_episode, verbose) -> Dict`
Evaluate agent (deterministic mode).

**Parameters**:
- `num_episodes`: Number of evaluation episodes
- `max_steps_per_episode`: Max steps per episode
- `verbose`: Show progress

**Returns**:
- Evaluation results

---

### `add_step_callback(callback: Callable)`
Add callback function called after each step.

**Parameters**:
- `callback`: Function with signature `(step, agent_info, env_info)`

**Example**:
```python
def my_callback(step, agent_info, env_info):
    print(f"Step {step}")

runner.add_step_callback(my_callback)
```

---

### `add_episode_callback(callback: Callable)`
Add callback function called after each episode.

**Parameters**:
- `callback`: Function with signature `(episode, episode_data)`

---

## GenerativeModel

Neural network mapping hidden states to observations.

### `__init__(config: Config)`
Initialize generative model.

---

### `forward(hidden_state: torch.Tensor) -> torch.Tensor`
Generate predicted observation from hidden state.

**Parameters**:
- `hidden_state`: Hidden state tensor

**Returns**:
- Predicted observation

**Example**:
```python
predicted_obs = generative_model(hidden_state)
```

---

### `predict_with_uncertainty(belief_mean, belief_std, num_samples=10) -> Tuple[torch.Tensor, torch.Tensor]`
Predict observation with uncertainty propagation.

**Parameters**:
- `belief_mean`: Mean of belief distribution
- `belief_std`: Std of belief distribution
- `num_samples`: Number of Monte Carlo samples

**Returns**:
- Tuple of (predicted_obs_mean, predicted_obs_std)

**Example**:
```python
obs_mean, obs_std = generative_model.predict_with_uncertainty(belief_mean, belief_std)
```

---

## Logger

Comprehensive experiment logging.

### `__init__(output_manager=None, log_dir=None, experiment_name=None)`
Initialize logger.

**Parameters**:
- `output_manager`: OutputManager instance (recommended)
- `log_dir`: Legacy log directory
- `experiment_name`: Experiment name

**Example**:
```python
logger = Logger(output_manager=output_mgr)
```

---

### `log_step(timestep, agent_info, env_info)`
Log data from single step.

**Parameters**:
- `timestep`: Current timestep
- `agent_info`: Agent information dict
- `env_info`: Environment information dict

**Example**:
```python
logger.log_step(step, agent_info, env_info)
```

---

### `log_episode(episode_num, episode_data)`
Log episode summary.

**Parameters**:
- `episode_num`: Episode number
- `episode_data`: Episode data dict

---

### `log_custom(key, value)`
Log custom metric.

**Parameters**:
- `key`: Metric name
- `value`: Metric value

**Example**:
```python
logger.log_custom("my_metric", value)
```

---

### `save_agent_state(agent, checkpoint_name="final")`
Save agent checkpoint.

**Parameters**:
- `agent`: ActiveInferenceAgent
- `checkpoint_name`: Checkpoint name

**Example**:
```python
logger.save_agent_state(agent, "final")
```

---

### `save_metrics(filename="metrics.json")`
Save accumulated metrics.

**Parameters**:
- `filename`: Output filename

---

### `save_config(config)`
Save configuration.

**Parameters**:
- `config`: Config object

---

### `get_summary() -> Dict`
Get experiment summary statistics.

**Returns**:
- Summary dictionary

---

### `print_summary()`
Print formatted summary to console.

---

## OutputManager

Unified output directory management.

### `__init__(output_root="./output", experiment_name=None)`
Initialize output manager.

**Parameters**:
- `output_root`: Root output directory
- `experiment_name`: Experiment name (auto-generated if None)

**Example**:
```python
output_mgr = OutputManager(output_root="./output", experiment_name="my_exp")
```

---

### `get_path(category, filename) -> Path`
Get full path for file in category.

**Parameters**:
- `category`: One of 'config', 'logs', 'checkpoints', 'visualizations', 'animations', 'data', 'metadata'
- `filename`: Filename

**Returns**:
- Full path

**Example**:
```python
config_path = output_mgr.get_path('config', 'config.json')
```

---

### `get_experiment_summary() -> dict`
Get experiment directory summary.

**Returns**:
- Summary dictionary

---

### `print_structure()`
Print directory structure to console.

---

### `list_experiments(output_root="./output") -> list` (static method)
List all experiments in output root.

**Parameters**:
- `output_root`: Root directory

**Returns**:
- List of experiment names

**Example**:
```python
experiments = OutputManager.list_experiments("./output")
```

---

## PolicyEvaluator

Evaluates policies using EFE.

### `__init__(config, transition_model, generative_model, efe_calculator)`
Initialize policy evaluator.

---

### `sample_actions(num_samples) -> torch.Tensor`
Sample candidate actions.

**Parameters**:
- `num_samples`: Number of samples

**Returns**:
- Action samples [num_samples, action_dim]

---

### `rollout(initial_state, actions, horizon=None) -> Tuple[List, List]`
Perform policy rollout.

**Parameters**:
- `initial_state`: Starting state
- `actions`: Action sequence
- `horizon`: Planning horizon

**Returns**:
- Tuple of (states, observations)

---

### `evaluate_policy(current_belief_mean, current_belief_std, preferred_obs) -> Tuple[torch.Tensor, torch.Tensor, dict]`
Evaluate policies using EFE.

**Parameters**:
- `current_belief_mean`: Belief mean
- `current_belief_std`: Belief std
- `preferred_obs`: Goal observations

**Returns**:
- Tuple of (efe_per_policy, efe_per_timestep, info)

---

### `update_policy(gradient, learning_rate=None)`
Update policy via gradient descent.

**Parameters**:
- `gradient`: Tuple of (grad_mean, grad_log_std)
- `learning_rate`: Learning rate

---

### `select_action(deterministic=False) -> torch.Tensor`
Select action from policy.

**Parameters**:
- `deterministic`: If True, return mean

**Returns**:
- Action tensor

---

### `reset()`
Reset policy to default.

---

### `to_dict() -> dict`
Convert policy to dictionary.

**Returns**:
- Policy dictionary

---

## TransitionModel

Neural network predicting state transitions.

### `__init__(config: Config)`
Initialize transition model.

---

### `forward(state, action) -> torch.Tensor`
Predict next state.

**Parameters**:
- `state`: Current state
- `action`: Action

**Returns**:
- Next state prediction

---

## VariationalFreeEnergy

Computes variational free energy for belief updating.

### `__init__(precision_obs=1.0, precision_prior=1.0)`
Initialize VFE calculator.

**Parameters**:
- `precision_obs`: Observation precision
- `precision_prior`: Prior precision

---

### `compute(observation, belief_mean, belief_std, predicted_obs, prior_mean, prior_std) -> Tuple[torch.Tensor, dict]`
Compute variational free energy.

**Parameters**:
- `observation`: Actual observation
- `belief_mean`: Belief mean
- `belief_std`: Belief std
- `predicted_obs`: Predicted observation
- `prior_mean`: Prior mean
- `prior_std`: Prior std

**Returns**:
- Tuple of (vfe, components_dict)

**Example**:
```python
vfe, components = vfe_calculator.compute(observation, belief_mean, belief_std, predicted_obs, prior_mean, prior_std)
```

---

## Visualizer

Creates static visualizations.

### `__init__(output_manager=None, save_dir=None)`
Initialize visualizer.

**Parameters**:
- `output_manager`: OutputManager (recommended)
- `save_dir`: Legacy save directory

---

### `plot_free_energy(history, title="Free Energy Over Time") -> Figure`
Plot VFE and EFE.

**Parameters**:
- `history`: Agent history
- `title`: Plot title

**Returns**:
- Matplotlib figure

---

### `plot_beliefs(history, title="Belief States Over Time") -> Figure`
Plot belief evolution.

**Parameters**:
- `history`: Agent history
- `title`: Plot title

**Returns**:
- Matplotlib figure

---

### `plot_trajectory_2d(history, goal=None, title="Agent Trajectory") -> Figure`
Plot 2D trajectory.

**Parameters**:
- `history`: Agent history
- `goal`: Goal position
- `title`: Plot title

**Returns**:
- Matplotlib figure

---

### `plot_actions(history, title="Actions Over Time") -> Figure`
Plot action evolution.

---

### `plot_comprehensive_summary(history, goal=None) -> Figure`
Create multi-panel summary.

**Parameters**:
- `history`: Agent history
- `goal`: Goal position

**Returns**:
- Matplotlib figure

---

### `show()`
Display all plots.

---

### `close_all()`
Close all figures.

---

## Animator

Generates animations.

### `__init__(output_manager=None, save_dir=None)`
Initialize animator.

---

### `animate_trajectory_2d(history, goal=None, fps=10, filename="trajectory_animation.gif")`
Animate 2D trajectory.

**Parameters**:
- `history`: Agent history
- `goal`: Goal position
- `fps`: Frames per second
- `filename`: Output filename

---

### `animate_beliefs(history, fps=10, filename="beliefs_animation.gif")`
Animate belief evolution.

---

### `animate_comprehensive(history, goal=None, fps=10, filename="comprehensive_animation.gif")`
Create multi-panel animation.

---

## Utility Functions

### `ensure_tensor(x, device=None, dtype=None) -> torch.Tensor`
Ensure input is tensor with correct device/dtype.

**Parameters**:
- `x`: Input
- `device`: Target device
- `dtype`: Target dtype

**Returns**:
- Tensor

---

### `gaussian_log_likelihood(x, mu, precision) -> torch.Tensor`
Compute Gaussian log-likelihood.

**Parameters**:
- `x`: Observations
- `mu`: Mean predictions
- `precision`: Precision (inverse variance)

**Returns**:
- Log-likelihood values

---

### `softmax(x, temperature=1.0) -> torch.Tensor`
Temperature-scaled softmax.

**Parameters**:
- `x`: Input logits
- `temperature`: Temperature parameter

**Returns**:
- Probability distribution

---

### `kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p) -> torch.Tensor`
KL divergence between Gaussians.

**Parameters**:
- `mu_q`, `sigma_q`: q distribution parameters
- `mu_p`, `sigma_p`: p distribution parameters

**Returns**:
- KL divergence

---

### `entropy_categorical(probs, eps=1e-8) -> torch.Tensor`
Categorical entropy.

**Parameters**:
- `probs`: Probability distribution
- `eps`: Numerical stability constant

**Returns**:
- Entropy value

---

## Method Count Summary

- **ActiveInferenceAgent**: 10 methods
- **BeliefState**: 8 methods
- **Config**: 3 methods
- **Environment**: 3 abstract methods + 3 implementations
- **ExpectedFreeEnergy**: 2 methods
- **ExperimentRunner**: 6 methods
- **GenerativeModel**: 3 methods
- **Logger**: 10 methods
- **OutputManager**: 5 methods
- **PolicyEvaluator**: 8 methods
- **TransitionModel**: 2 methods
- **VariationalFreeEnergy**: 2 methods
- **Visualizer**: 8 methods
- **Animator**: 4 methods
- **Utility Functions**: 5 functions

**Total**: 82 public methods/functions documented

---

## Related Documentation

- **[API.md](API.md)** - Detailed API documentation
- **[AGENTS.md](AGENTS.md)** - Usage examples
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
- **[../active_torchference/README.md](../active_torchference/README.md)** - Package structure

