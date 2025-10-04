# Active Torchference Package

Core implementation of Active Inference framework in PyTorch.

## Package Structure

```
active_torchference/
├── __init__.py              # Package exports
├── agent.py                 # Active Inference agent
├── beliefs.py               # Belief state and generative model
├── config.py                # Configuration system
├── environment.py           # Environment implementations
├── free_energy.py           # VFE and EFE computations
├── policy.py                # Policy evaluation and selection
├── utils.py                 # Utility functions
├── output_manager.py        # Unified output management
├── orchestrators/           # Orchestration tools
│   ├── __init__.py
│   ├── animator.py          # Animation generation
│   ├── logger.py            # Experiment logging
│   ├── runner.py            # Experiment execution
│   └── visualizer.py        # Visualization tools
└── README.md                # This file
```

## Core Modules

### `agent.py`
Active Inference agent implementing the action-perception loop.

**Key Class**: `ActiveInferenceAgent`

**Methods**:
- `perceive(observation)`: Update beliefs via VFE minimization
- `plan(preferred_obs)`: Evaluate policies via EFE
- `act(deterministic)`: Select action from policy posterior
- `step(observation, preferred_obs)`: Execute full loop
- `reset()`: Reset agent state

**Action-Perception Loop**:
1. **Perceive**: Minimize VFE to update beliefs
2. **Plan**: Minimize EFE to optimize policy
3. **Act**: Sample action from policy posterior

---

### `beliefs.py`
Belief state management and generative model.

**Classes**:
- `BeliefState`: Manages approximate posterior q(s|o)
- `GenerativeModel`: Neural network for p(o|s)

**BeliefState Methods**:
- `sample(num_samples)`: Sample from belief distribution
- `update(gradient)`: Update beliefs via gradient descent
- `reset()`: Reset to prior
- `set_prior(mean, std)`: Set prior distribution

**GenerativeModel**:
- Forward pass: Hidden state → Predicted observation
- Uncertainty propagation via sampling

---

### `config.py`
Configuration system for all parameters.

**Class**: `Config` (dataclass)

**Key Parameters**:
- Dimensions: `hidden_dim`, `obs_dim`, `action_dim`
- Learning rates: `learning_rate_beliefs`, `learning_rate_policy`
- Policy: `num_policy_iterations`, `num_rollouts`, `horizon`
- Free energy: `epistemic_weight`, `pragmatic_weight`
- Precision: `precision_obs`, `precision_prior`, `precision_state`
- Device: `device`, `dtype`, `seed`

**Methods**:
- `to_dict()`: Serialize configuration
- `from_dict(config_dict)`: Deserialize configuration

---

### `environment.py`
Environment base class and implementations.

**Abstract Base**: `Environment`

**Required Methods**:
- `reset()`: Initialize environment
- `step(action)`: Process action, return observation
- `get_preferred_observation()`: Return goal

**Implementations**:
- `ContinuousEnvironment`: Continuous 2D navigation
- `GridWorld`: Discrete grid navigation
- `OscillatorEnvironment`: Oscillating goal tracking

---

### `free_energy.py`
Free energy computations for inference and planning.

**Classes**:
- `VariationalFreeEnergy`: VFE for belief updates
- `ExpectedFreeEnergy`: EFE for policy evaluation

**VariationalFreeEnergy**:
```python
VFE = -E_q[log p(o|s)] + KL[q(s)||p(s)]
    = Likelihood Term + Complexity Term
```
- Minimized to update beliefs
- Returns: (vfe, components)

**ExpectedFreeEnergy**:
```python
EFE = -(Epistemic Value + Pragmatic Value)
    = -(Information Gain + Goal Achievement)
```
- Minimized to select policies
- Returns: (efe, components)

---

### `policy.py`
Policy evaluation and selection.

**Classes**:
- `TransitionModel`: Predicts state transitions p(s'|s,a)
- `PolicyEvaluator`: Evaluates policies via rollouts

**PolicyEvaluator Methods**:
- `sample_actions(num_samples)`: Sample candidate actions
- `rollout(initial_state, actions, horizon)`: Perform rollout
- `evaluate_policy(belief, preferred_obs)`: Compute EFE per policy
- `update_policy(gradient)`: Update policy via gradient descent
- `select_action(deterministic)`: Select action from posterior

**Key Feature**: EFE computed per policy per timestep (not averaged).

---

### `utils.py`
Utility functions for computations.

**Functions**:
- `ensure_tensor(x, device, dtype)`: Tensor conversion
- `gaussian_log_likelihood(x, mu, precision)`: Log-likelihood
- `softmax(x, temperature)`: Temperature-scaled softmax
- `kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)`: KL divergence
- `entropy_categorical(probs)`: Categorical entropy

---

### `output_manager.py`
Unified output directory management.

**Class**: `OutputManager`

**Directory Structure**:
```
output/
└── experiment_name/
    ├── config/           # Configuration files
    ├── logs/             # Metrics and summaries
    ├── checkpoints/      # Agent checkpoints
    ├── visualizations/   # Static plots
    ├── animations/       # GIF animations
    ├── data/             # Episode records
    └── metadata/         # Experiment metadata
```

**Methods**:
- `get_path(category, filename)`: Get file path
- `get_experiment_summary()`: Get summary dict
- `print_structure()`: Print directory tree
- `list_experiments(output_root)`: List all experiments

---

## Orchestrators

See [orchestrators/README.md](orchestrators/README.md) for detailed documentation.

**Modules**:
- `runner.py`: Experiment execution
- `logger.py`: Comprehensive logging
- `visualizer.py`: Static plots
- `animator.py`: Animations

---

## Usage

### Basic Agent Creation

```python
from active_torchference import Config, ActiveInferenceAgent

config = Config(
    hidden_dim=8,
    obs_dim=2,
    action_dim=2,
    epistemic_weight=1.0,
    pragmatic_weight=1.0
)

agent = ActiveInferenceAgent(config)
```

### Manual Loop

```python
# Perceive
perception_info = agent.perceive(observation)

# Plan
planning_info = agent.plan(preferred_obs)

# Act
action = agent.act(deterministic=False)
```

### Complete Step

```python
action, info = agent.step(observation, preferred_obs)
```

### With Environment

```python
from active_torchference import ContinuousEnvironment

env = ContinuousEnvironment(config)
observation = env.reset()
preferred_obs = env.get_preferred_observation()

for step in range(100):
    action, info = agent.step(observation, preferred_obs)
    observation, env_info = env.step(action)
```

### With Orchestrators

```python
from active_torchference import OutputManager
from active_torchference.orchestrators import ExperimentRunner, Logger

output_mgr = OutputManager(experiment_name="my_experiment")
logger = Logger(output_manager=output_mgr)

runner = ExperimentRunner(agent, env, logger, config)
results = runner.run(num_episodes=10)
```

---

## Design Principles

1. **Modularity**: Each component has single responsibility
2. **Clarity**: Explicit action-perception loop
3. **Testability**: Comprehensive test coverage
4. **Extensibility**: Easy to extend with custom components
5. **Documentation**: Self-documenting code

---

## Key Concepts

### No Reward Function
Active Inference does NOT use reward signals. Instead:
- **VFE** drives perception (belief updates)
- **EFE** drives action (policy selection)
- **Epistemic value**: Information gain (exploration)
- **Pragmatic value**: Goal achievement (exploitation)

### Per-Policy EFE
EFE computed for each policy individually:
- Shape: `[num_rollouts]`
- NOT averaged during evaluation
- Individual policies differentiable for gradient descent

### Iterative Minimization
Both belief and policy updates use iterative gradient descent:
- Beliefs: Minimize VFE over iterations
- Policy: Minimize mean EFE over iterations

---

## Extending the Package

### Custom Environment

```python
from active_torchference.environment import Environment

class CustomEnvironment(Environment):
    def reset(self):
        # Initialize
        return observation
    
    def step(self, action):
        # Process action
        return observation, info
    
    def get_preferred_observation(self):
        return goal_obs
```

### Custom Generative Model

```python
from active_torchference.beliefs import GenerativeModel

class CustomGenerativeModel(GenerativeModel):
    def __init__(self, config):
        super().__init__(config)
        # Custom architecture
    
    def forward(self, hidden_state):
        # Custom forward pass
        return predicted_obs
```

---

## Dependencies

**Core**:
- `torch`: Neural networks and autodiff
- `numpy`: Numerical operations

**Optional** (for orchestrators):
- `matplotlib`: Plotting
- `pillow`: Animation export
- `tqdm`: Progress bars

---

## Related Documentation

- **[../docs/AGENTS.md](../docs/AGENTS.md)**: Comprehensive agent guide
- **[../docs/API.md](../docs/API.md)**: Complete API reference
- **[../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)**: System design
- **[../docs/QUICKSTART.md](../docs/QUICKSTART.md)**: Getting started tutorial
- **[../examples/README.md](../examples/README.md)**: Example scripts
- **[../tests/README.md](../tests/README.md)**: Test suite
- **[../validation/README.md](../validation/README.md)**: Validation scripts

---

## Import Structure

**Top-level imports**:
```python
from active_torchference import (
    ActiveInferenceAgent,
    Config,
    OutputManager,
    Environment,
    ContinuousEnvironment,
    GridWorld,
    OscillatorEnvironment,
    VariationalFreeEnergy,
    ExpectedFreeEnergy,
    BeliefState,
    PolicyEvaluator
)
```

**Orchestrators**:
```python
from active_torchference.orchestrators import (
    ExperimentRunner,
    Logger,
    Visualizer,
    Animator
)
```

**Utilities**:
```python
from active_torchference.utils import (
    gaussian_log_likelihood,
    kl_divergence_gaussian,
    softmax
)
```

