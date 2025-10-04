# API Documentation

## Core Classes

### ActiveInferenceAgent

Main agent class implementing the action-perception loop.

```python
from active_torchference import ActiveInferenceAgent, Config

config = Config()
agent = ActiveInferenceAgent(config)
```

**Methods:**

- `perceive(observation)`: Update beliefs via VFE minimization
- `plan(preferred_obs)`: Evaluate policies via EFE
- `act(deterministic=False)`: Select action from policy posterior
- `step(observation, preferred_obs)`: Execute full action-perception loop
- `reset()`: Reset agent state
- `get_history()`: Retrieve agent history
- `save_state()`: Save agent state to dict
- `load_state(state_dict)`: Load agent state from dict

### Config

Configuration dataclass for all parameters.

```python
config = Config(
    hidden_dim=8,
    obs_dim=2,
    action_dim=2,
    learning_rate_beliefs=0.1,
    learning_rate_policy=0.05,
    epistemic_weight=1.0,
    pragmatic_weight=1.0
)
```

**Key Parameters:**

- `hidden_dim`: Hidden state dimensionality
- `obs_dim`: Observation dimensionality
- `action_dim`: Action dimensionality
- `learning_rate_beliefs`: Learning rate for belief updates
- `learning_rate_policy`: Learning rate for policy updates
- `num_policy_iterations`: Policy optimization iterations
- `horizon`: Planning horizon
- `epistemic_weight`: Weight for exploration
- `pragmatic_weight`: Weight for exploitation

### Environment

Abstract base class for environments.

```python
from active_torchference.environment import Environment

class CustomEnvironment(Environment):
    def reset(self):
        # Return initial observation
        pass
    
    def step(self, action):
        # Return (observation, info)
        pass
    
    def get_preferred_observation(self):
        # Return goal observation
        pass
```

**Built-in Environments:**

- `ContinuousEnvironment`: Continuous 2D navigation
- `GridWorld`: Discrete grid navigation

## Free Energy Classes

### VariationalFreeEnergy

Computes VFE for belief updating.

```python
from active_torchference.free_energy import VariationalFreeEnergy

vfe_calc = VariationalFreeEnergy(
    precision_obs=1.0,
    precision_prior=1.0
)

vfe, components = vfe_calc.compute(
    observation, belief_mean, belief_std,
    predicted_obs, prior_mean, prior_std
)
```

**Returns:**

- `vfe`: Total variational free energy
- `components`: Dict with 'likelihood', 'complexity', 'vfe'

### ExpectedFreeEnergy

Computes EFE for policy evaluation.

```python
from active_torchference.free_energy import ExpectedFreeEnergy

efe_calc = ExpectedFreeEnergy(
    epistemic_weight=1.0,
    pragmatic_weight=1.0
)

efe, components = efe_calc.compute(
    predicted_obs, predicted_obs_std,
    preferred_obs, state_entropy
)
```

**Returns:**

- `efe`: Total expected free energy
- `components`: Dict with 'epistemic', 'pragmatic', 'efe'

## Belief Management

### BeliefState

Manages belief state distribution.

```python
from active_torchference.beliefs import BeliefState

beliefs = BeliefState(config)

# Sample from beliefs
samples = beliefs.sample(num_samples=10)

# Update beliefs
beliefs.update(gradient, learning_rate=0.1)

# Reset to prior
beliefs.reset()
```

### GenerativeModel

Neural network for observation prediction.

```python
from active_torchference.beliefs import GenerativeModel

gen_model = GenerativeModel(config)

# Predict observation
predicted_obs = gen_model(hidden_state)

# Predict with uncertainty
obs_mean, obs_std = gen_model.predict_with_uncertainty(
    belief_mean, belief_std, num_samples=100
)
```

## Policy Evaluation

### PolicyEvaluator

Evaluates and selects policies.

```python
from active_torchference.policy import PolicyEvaluator

evaluator = PolicyEvaluator(
    config, transition_model,
    generative_model, efe_calculator
)

# Evaluate policy
efe, info = evaluator.evaluate_policy(
    belief_mean, belief_std, preferred_obs
)

# Select action
action = evaluator.select_action(deterministic=False)
```

### TransitionModel

Predicts state transitions.

```python
from active_torchference.policy import TransitionModel

trans_model = TransitionModel(config)
next_state = trans_model(state, action)
```

## Orchestrators

### ExperimentRunner

Orchestrates experiments.

```python
from active_torchference.orchestrators import ExperimentRunner

runner = ExperimentRunner(agent, environment, logger, config)

# Run multiple episodes
results = runner.run(
    num_episodes=10,
    max_steps_per_episode=100,
    deterministic=False,
    verbose=True
)

# Evaluate agent
eval_results = runner.evaluate(num_episodes=5)
```

**Callbacks:**

```python
# Step callback
def my_step_callback(step, agent_info, env_info):
    print(f"Step {step}: VFE = {agent_info['vfe']}")

runner.add_step_callback(my_step_callback)

# Episode callback
def my_episode_callback(episode, episode_data):
    print(f"Episode {episode} complete")

runner.add_episode_callback(my_episode_callback)
```

### Logger

Comprehensive logging system.

```python
from active_torchference.orchestrators import Logger

logger = Logger(
    log_dir="./logs",
    experiment_name="my_experiment"
)

# Log step
logger.log_step(timestep, agent_info, env_info)

# Log episode
logger.log_episode(episode_num, episode_data)

# Log custom metrics
logger.log_custom("my_metric", value)

# Save artifacts
logger.save_agent_state(agent, checkpoint_name="final")
logger.save_metrics()
logger.save_config(config)

# Get summary
summary = logger.get_summary()
logger.print_summary()
```

### Visualizer

Visualization tools.

```python
from active_torchference.orchestrators import Visualizer

viz = Visualizer(save_dir="./results")

# Get agent history
history = agent.get_history()

# Create plots
viz.plot_free_energy(history)
viz.plot_beliefs(history)
viz.plot_trajectory_2d(history, goal=goal_position)
viz.plot_actions(history)
viz.plot_comprehensive_summary(history, goal=goal_position)

# Display
viz.show()

# Close
viz.close_all()
```

### Animator

Animation generation.

```python
from active_torchference.orchestrators import Animator

animator = Animator(save_dir="./results")

# Create animations
animator.animate_trajectory_2d(
    history, goal=goal_position,
    fps=10, filename="trajectory.gif"
)

animator.animate_beliefs(
    history, fps=10,
    filename="beliefs.gif"
)

animator.animate_comprehensive(
    history, goal=goal_position,
    fps=10, filename="comprehensive.gif"
)
```

## Utility Functions

### Tensor Operations

```python
from active_torchference.utils import (
    ensure_tensor,
    gaussian_log_likelihood,
    softmax,
    kl_divergence_gaussian,
    entropy_categorical
)

# Ensure tensor with correct device/dtype
tensor = ensure_tensor(data, device=device, dtype=dtype)

# Gaussian log-likelihood
log_like = gaussian_log_likelihood(x, mu, precision)

# Temperature-scaled softmax
probs = softmax(logits, temperature=1.0)

# KL divergence between Gaussians
kl = kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)

# Categorical entropy
entropy = entropy_categorical(probs)
```

## Common Patterns

### Basic Agent-Environment Loop

```python
# Initialize
config = Config()
agent = ActiveInferenceAgent(config)
env = ContinuousEnvironment(config)

# Reset
obs = env.reset()
agent.reset()

# Run episode
for step in range(100):
    # Get goal
    preferred_obs = env.get_preferred_observation()
    
    # Agent step
    action, info = agent.step(obs, preferred_obs)
    
    # Environment step
    obs, env_info = env.step(action)
    
    if env_info.get("done"):
        break
```

### Using Orchestrator

```python
# Setup
config = Config()
agent = ActiveInferenceAgent(config)
env = ContinuousEnvironment(config)
logger = Logger()
viz = Visualizer(save_dir="./results")

# Run experiment
runner = ExperimentRunner(agent, env, logger, config)
results = runner.run(num_episodes=10)

# Visualize
history = agent.get_history()
viz.plot_comprehensive_summary(history, goal=env.get_preferred_observation())
viz.show()
```

### Custom Environment

```python
class MyEnvironment(Environment):
    def __init__(self, config):
        super().__init__(config)
        self.goal = torch.tensor([1.0, 1.0])
    
    def reset(self):
        self.state = torch.zeros(self.config.obs_dim)
        return self.state
    
    def step(self, action):
        self.state += action * 0.1
        obs = self.state + torch.randn_like(self.state) * 0.1
        
        info = {
            "distance_to_goal": torch.norm(self.state - self.goal).item(),
            "done": torch.norm(self.state - self.goal) < 0.1
        }
        
        return obs, info
    
    def get_preferred_observation(self):
        return self.goal
```

### Tuning Epistemic-Pragmatic Balance

```python
# High exploration
config_explore = Config(
    epistemic_weight=2.0,  # High exploration
    pragmatic_weight=0.5   # Low exploitation
)

# Balanced
config_balanced = Config(
    epistemic_weight=1.0,
    pragmatic_weight=1.0
)

# High exploitation
config_exploit = Config(
    epistemic_weight=0.5,  # Low exploration
    pragmatic_weight=2.0   # High exploitation
)
```

## Best Practices

1. **Configuration**: Start with default config and adjust gradually
2. **Learning Rates**: Beliefs typically need higher LR than policy
3. **Horizon**: Longer horizons = more planning, but slower
4. **Rollouts**: More rollouts = better policy evaluation
5. **Precision**: Higher precision = more confident predictions
6. **Weights**: Balance epistemic/pragmatic based on task
7. **Testing**: Always test with reproducible seeds first
8. **Logging**: Use logger and visualizer for all experiments
9. **Modularity**: Subclass environments for custom dynamics
10. **TDD**: Write tests for custom components

