# Active Inference Agents

## Overview

This document provides a comprehensive guide to the **Active Inference Agent** implementation in **active_torchference**. Active Inference is a unified theory that explains perception, action, and learning as minimizing variational free energy (VFE) and expected free energy (EFE).

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Agent Architecture](#agent-architecture)
3. [Action-Perception Loop](#action-perception-loop)
4. [Belief State Management](#belief-state-management)
5. [Policy Evaluation](#policy-evaluation)
6. [Free Energy Computations](#free-energy-computations)
7. [Agent Configuration](#agent-configuration)
8. [Usage Examples](#usage-examples)
9. [Validation and Testing](#validation-and-testing)

---

## Core Concepts

### Active Inference Framework

Active Inference posits that intelligent agents operate by:

1. **Perceiving**: Using observations to update beliefs about hidden states (minimize VFE)
2. **Planning**: Evaluating potential action sequences (policies) using EFE
3. **Acting**: Selecting actions from the optimized policy posterior

**Key Distinction**: Unlike reinforcement learning, Active Inference does NOT use reward functions. Instead:
- **VFE** (Variational Free Energy) drives perception and belief updates
- **EFE** (Expected Free Energy) drives action selection through:
  - **Epistemic value**: Information gain (exploration)
  - **Pragmatic value**: Goal achievement (exploitation)

### Fundamental Components

1. **Generative Model**: \( p(o|s) \) - Predicts observations from hidden states
2. **Belief State**: \( q(s|o) \) - Approximate posterior over hidden states
3. **Transition Model**: \( p(s'|s,a) \) - Predicts state transitions
4. **Policy**: \( \pi(a|s) \) - Distribution over actions given states

---

## Agent Architecture

### `ActiveInferenceAgent`

The main agent class orchestrating the action-perception loop.

```python
from active_torchference import Config, ActiveInferenceAgent

config = Config(
    hidden_dim=8,
    obs_dim=4,
    action_dim=2,
    num_belief_iterations=5,
    num_policy_iterations=10,
    horizon=3
)

agent = ActiveInferenceAgent(config)
```

**Key Attributes:**

- `beliefs`: Current belief state (mean and std)
- `generative_model`: Maps hidden states to observations
- `policy_evaluator`: Evaluates policies using EFE
- `vfe_calculator`: Computes variational free energy
- `efe_calculator`: Computes expected free energy

**Methods:**

- `perceive(observation)`: Update beliefs using VFE minimization
- `plan(preferred_obs)`: Optimize policy using EFE minimization
- `act(deterministic)`: Select action from policy posterior
- `step(observation, preferred_obs)`: Execute full loop
- `reset()`: Reset agent to initial state

---

## Action-Perception Loop

The agent operates in a continuous cycle:

```
┌─────────────────────────────────────────────┐
│                                             │
│  1. PERCEIVE                                │
│     - Receive observation from environment  │
│     - Minimize VFE to update beliefs        │
│     - Iterate belief updates                │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│  2. PLAN                                    │
│     - Evaluate multiple policies            │
│     - Compute EFE per policy per timestep   │
│     - Minimize EFE to optimize policy       │
│     - Iterate policy updates                │
│                                             │
├─────────────────────────────────────────────┤
│                                             │
│  3. ACT                                     │
│     - Sample action from policy posterior   │
│     - Execute action in environment         │
│     - Receive next observation              │
│                                             │
└─────────────────────────────────────────────┘
```

### Implementation

```python
# Initialize
env = ContinuousEnvironment(config)
agent = ActiveInferenceAgent(config)
observation = env.reset()
preferred_obs = torch.tensor([1.0, 0.0, 0.0, 0.0])

# Action-perception loop
for step in range(100):
    # 1. Perceive: Update beliefs
    perception_info = agent.perceive(observation)
    
    # 2. Plan: Optimize policy
    planning_info = agent.plan(preferred_obs)
    
    # 3. Act: Select and execute action
    action = agent.act(deterministic=False)
    observation, _ = env.step(action)
```

---

## Belief State Management

### Belief Representation

Beliefs are represented as Gaussian distributions over hidden states:

```
q(s|o) = N(μ, σ²)
```

Where:
- `μ` (mean): Expected hidden state
- `σ` (std): Uncertainty in hidden state

### Belief Update Process

**Iterative VFE Minimization:**

```python
def perceive(self, observation: torch.Tensor, num_iterations: int = None) -> dict:
    """
    Update beliefs by minimizing VFE over multiple iterations.
    """
    if num_iterations is None:
        num_iterations = self.config.num_belief_iterations
    
    for iteration in range(num_iterations):
        # 1. Enable gradients for belief parameters
        self.beliefs.mean.requires_grad_(True)
        self.beliefs.log_std.requires_grad_(True)
        
        # 2. Compute predicted observation from beliefs
        predicted_obs = self.generative_model(self.beliefs.mean)
        
        # 3. Compute VFE
        vfe, components = self.vfe_calculator.compute(
            observation=observation,
            belief_mean=self.beliefs.mean,
            belief_std=self.beliefs.std,
            predicted_obs=predicted_obs,
            prior_mean=self.beliefs.prior_mean,
            prior_std=self.beliefs.prior_std
        )
        
        # 4. Backpropagate to get gradients
        vfe.backward()
        
        # 5. Update beliefs using gradient descent
        self.beliefs.update((
            self.beliefs.mean.grad,
            self.beliefs.log_std.grad
        ))
        
        # 6. Clear gradients
        self.beliefs.mean.grad.zero_()
        self.beliefs.log_std.grad.zero_()
    
    return info
```

**Key Properties:**

- **Per-Timestep**: Beliefs update at each timestep with new observations
- **Per-Dimension**: Each latent dimension is independently updated
- **Iterative**: Multiple gradient steps per timestep ensure convergence
- **VFE Minimization**: Driven by minimizing VFE = Likelihood + Complexity

### Validation

Belief updates are validated to ensure:
1. VFE computed at every timestep
2. All latent dimensions are active
3. VFE decreases with iterations
4. Beliefs respond to observations
5. Components (likelihood + complexity) sum correctly

See: `validation/validate_vfe_computation.py`

---

## Policy Evaluation

### Policy Representation

Policies are represented as Gaussian distributions over action sequences:

```
π(a|s) = N(μ_a, σ_a²)
```

### Policy Evaluation Process

**Computing EFE per Policy per Timestep:**

```python
def evaluate_policy(
    self,
    current_belief_mean: torch.Tensor,
    current_belief_std: torch.Tensor,
    preferred_obs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Evaluate multiple policies by computing EFE per policy per timestep.
    
    Returns:
        efe_per_policy: Total EFE for each policy [num_rollouts]
        efe_per_timestep: EFE for each policy at each timestep [horizon, num_rollouts]
        info: Additional information dictionary
    """
    # 1. Sample multiple action sequences (policies)
    action_samples = self.sample_actions(num_samples=self.config.num_rollouts)
    
    # 2. Rollout each policy from current belief
    states, observations = self.rollout(
        initial_state=current_belief_mean,
        actions=action_samples,
        horizon=self.config.horizon
    )
    
    # 3. Compute EFE per policy per timestep
    efe_per_timestep = []
    
    for t in range(self.config.horizon):
        predicted_obs = observations[t]  # [num_rollouts, obs_dim]
        
        # Compute EFE for all policies at timestep t
        efe_t, components = self.efe_calculator.compute(
            predicted_obs,
            predicted_obs_std,
            preferred_obs.expand_as(predicted_obs),
            state_entropy
        )
        
        efe_per_timestep.append(efe_t)  # [num_rollouts]
    
    # 4. Aggregate EFE across timesteps
    efe_per_policy_per_timestep = torch.stack(efe_per_timestep, dim=0)  # [horizon, num_rollouts]
    efe_per_policy = efe_per_policy_per_timestep.sum(dim=0)  # [num_rollouts]
    
    return efe_per_policy, efe_per_policy_per_timestep, info
```

**Key Properties:**

- **Per-Policy**: Each policy (action sequence) has its own EFE value
- **Per-Timestep**: EFE is computed at each step in the planning horizon
- **NOT Averaged**: Individual policy EFEs are preserved for selection
- **Differentiable**: Enables gradient-based policy optimization

### Policy Optimization

**Iterative EFE Minimization:**

```python
def plan(self, preferred_obs: torch.Tensor) -> dict:
    """
    Optimize policy by minimizing expected free energy.
    """
    for iteration in range(self.config.num_policy_iterations):
        # 1. Evaluate multiple policies
        efe_per_policy, efe_per_timestep, eval_info = \
            self.policy_evaluator.evaluate_policy(
                current_belief_mean=self.beliefs.mean,
                current_belief_std=self.beliefs.std,
                preferred_obs=preferred_obs
            )
        
        # 2. Use mean EFE for optimization
        mean_efe = efe_per_policy.mean()
        
        # 3. Backpropagate
        mean_efe.backward()
        
        # 4. Update policy parameters
        self.policy_evaluator.update_policy(
            (policy_mean.grad, policy_log_std.grad)
        )
    
    return planning_info
```

### Action Selection

After policy optimization, actions are selected from the posterior:

```python
# Sample from policy posterior
action = agent.act(deterministic=False)

# Or use mean action
action = agent.act(deterministic=True)
```

### Validation

Policy evaluation is validated to ensure:
1. EFE computed per policy (not averaged)
2. EFE computed per timestep in horizon
3. Epistemic and pragmatic components present
4. Different policies have different EFE values
5. Best policy is correctly identified
6. Weights affect EFE appropriately

See: `validation/validate_efe_computation.py`

---

## Free Energy Computations

### Variational Free Energy (VFE)

**Purpose**: Update beliefs to explain observations

**Formula**:
```
VFE = E_q[log q(s) - log p(o,s)]
    = -E_q[log p(o|s)] + KL[q(s) || p(s)]
    = Likelihood term + Complexity term
```

**Components**:
- **Likelihood**: How well beliefs predict observations
- **Complexity**: Deviation from prior beliefs

**Implementation**:

```python
class VariationalFreeEnergy:
    def compute(
        self,
        observation: torch.Tensor,
        belief_mean: torch.Tensor,
        belief_std: torch.Tensor,
        predicted_obs: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        # Likelihood term (prediction error)
        likelihood_term = -gaussian_log_likelihood(
            observation, 
            predicted_obs, 
            predicted_obs_std
        )
        
        # Complexity term (KL divergence from prior)
        complexity_term = kl_divergence_gaussian(
            belief_mean, belief_std,
            prior_mean, prior_std
        )
        
        vfe = likelihood_term + complexity_term
        
        return vfe, {"likelihood": likelihood_term, "complexity": complexity_term}
```

**Properties**:
- Always positive
- Minimizing VFE improves belief accuracy
- Balances fit to data with adherence to priors

### Expected Free Energy (EFE)

**Purpose**: Evaluate and select policies

**Formula**:
```
EFE = E_π[log p(o|s) - log q(s|π)]
    = -Epistemic value - Pragmatic value
```

**Components**:
- **Epistemic Value**: Information gain / exploration
- **Pragmatic Value**: Goal achievement / exploitation

**Implementation**:

```python
class ExpectedFreeEnergy:
    def compute(
        self,
        predicted_obs: torch.Tensor,
        predicted_obs_std: torch.Tensor,
        preferred_obs: torch.Tensor,
        state_entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        # Pragmatic value (goal-seeking)
        obs_log_likelihood = gaussian_log_likelihood(
            predicted_obs,
            preferred_obs,
            predicted_obs_std
        )
        pragmatic_value = obs_log_likelihood
        
        # Epistemic value (information-seeking)
        epistemic_value = state_entropy
        
        # Combine with weights
        efe = -(self.config.epistemic_weight * epistemic_value + 
                self.config.pragmatic_weight * pragmatic_value)
        
        return efe, {
            "epistemic": epistemic_value,
            "pragmatic": pragmatic_value
        }
```

**Properties**:
- Lower EFE = better policy
- Balances exploration (epistemic) and exploitation (pragmatic)
- Configurable via `epistemic_weight` and `pragmatic_weight`

---

## Agent Configuration

### Key Parameters

```python
config = Config(
    # Dimensions
    hidden_dim=8,           # Latent state dimensionality
    obs_dim=4,              # Observation dimensionality
    action_dim=2,           # Action dimensionality
    
    # Belief updates
    num_belief_iterations=5,        # Iterations for VFE minimization
    learning_rate_beliefs=0.1,      # Belief update step size
    
    # Policy optimization
    num_policy_iterations=10,       # Iterations for EFE minimization
    learning_rate_policy=0.01,      # Policy update step size
    num_rollouts=20,                # Number of policies to evaluate
    horizon=3,                      # Planning horizon
    
    # EFE components
    epistemic_weight=1.0,           # Weight for exploration
    pragmatic_weight=1.0,           # Weight for exploitation
    
    # Computation
    device='cpu',
    dtype=torch.float32,
    seed=42
)
```

### Configuration Profiles

**High Exploration (Epistemic)**:
```python
config = Config(
    epistemic_weight=2.0,
    pragmatic_weight=0.5,
    num_rollouts=30  # More policy diversity
)
```

**High Exploitation (Pragmatic)**:
```python
config = Config(
    epistemic_weight=0.5,
    pragmatic_weight=2.0,
    horizon=5  # Longer planning
)
```

**Precise Beliefs**:
```python
config = Config(
    num_belief_iterations=10,
    learning_rate_beliefs=0.2
)
```

---

## Usage Examples

### Basic Agent Loop

```python
from active_torchference import Config, ActiveInferenceAgent
from active_torchference.environment import ContinuousEnvironment

# Setup
config = Config(hidden_dim=4, obs_dim=2, action_dim=2)
env = ContinuousEnvironment(config)
agent = ActiveInferenceAgent(config)

observation = env.reset()
preferred_obs = torch.tensor([1.0, 0.0])

# Run episode
for step in range(100):
    # Execute full step
    action, info = agent.step(observation, preferred_obs)
    observation, env_info = env.step(action)
    
    print(f"Step {step}:")
    print(f"  VFE: {info['vfe']:.4f}")
    print(f"  EFE: {info['efe']:.4f}")
    print(f"  Action: {action.numpy()}")
```

### Manual Control

```python
# Manual perception-action loop
perception_info = agent.perceive(observation)
print(f"Belief mean: {perception_info['belief_mean']}")
print(f"VFE: {perception_info['vfe']}")
print(f"VFE history: {perception_info['vfe_history']}")

planning_info = agent.plan(preferred_obs)
print(f"EFE per policy: {planning_info['efe_per_policy']}")
print(f"Best policy: #{planning_info['best_policy_idx']}")

action = agent.act(deterministic=False)
```

### With Logging and Visualization

```python
from active_torchference.output_manager import OutputManager
from active_torchference.orchestrators import Logger, Visualizer, Animator

# Setup output management
output_mgr = OutputManager(experiment_name="my_experiment")
logger = Logger(config, output_manager=output_mgr)
visualizer = Visualizer(config, output_manager=output_mgr)
animator = Animator(config, output_manager=output_mgr)

# Run with logging
for episode in range(10):
    observation = env.reset()
    agent.reset()
    
    for step in range(100):
        action, info = agent.step(observation, preferred_obs)
        observation, env_info = env.step(action)
        
        # Log everything
        logger.log_step(
            step=step,
            observation=observation,
            action=action,
            belief_mean=info['belief_mean'],
            vfe=info['vfe'],
            efe=info['efe']
        )
    
    logger.save_episode(episode)

# Visualize results
visualizer.plot_beliefs(logger.history)
visualizer.plot_free_energy(logger.history)
animator.create_animation(logger.history)
```

---

## Validation and Testing

### Running Validation Scripts

**VFE Validation**:
```bash
python3 validation/validate_vfe_computation.py
```

Validates:
- VFE computed per timestep
- All latent dimensions active
- Iterative minimization working
- Components sum correctly
- Responsive to observations

**EFE Validation**:
```bash
python3 validation/validate_efe_computation.py
```

Validates:
- EFE computed per policy
- EFE computed per timestep
- Epistemic + pragmatic components
- Policy differentiation
- Best policy selection
- Weight effects

### Unit Tests

Run comprehensive test suite:

```bash
pytest tests/
pytest tests/test_belief_updates.py -v
pytest tests/test_efe_per_policy.py -v
pytest tests/test_agent.py -v
pytest tests/test_integration.py -v
```

### Output Validation

Check experiment outputs:

```bash
python3 examples/unified_output_example.py

# Inspect outputs
ls -R output/unified_demo/
cat output/unified_demo/config/config.json
cat output/unified_demo/logs/metrics.json
```

---

## Advanced Topics

### Custom Generative Models

```python
from active_torchference.beliefs import GenerativeModel

class CustomGenerativeModel(GenerativeModel):
    def __init__(self, config: Config):
        super().__init__(config)
        # Add custom layers
        self.custom_layer = nn.Linear(config.hidden_dim, 64)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.custom_layer(hidden_state)
        x = torch.relu(x)
        return self.layers(x)
```

### Custom Environments

```python
from active_torchference.environment import BaseEnvironment

class CustomEnvironment(BaseEnvironment):
    def reset(self) -> torch.Tensor:
        self.state = torch.randn(self.config.hidden_dim)
        return self.observe()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Custom dynamics
        self.state = self.state + 0.1 * action
        observation = self.observe()
        return observation, {}
```

### Batch Processing

```python
# Process multiple episodes in parallel
from active_torchference.orchestrators import run_batch_experiments

results = run_batch_experiments(
    config=config,
    num_experiments=10,
    episodes_per_experiment=100,
    output_dir="batch_outputs"
)
```

---

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
2. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*.
3. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. *Biological Cybernetics*.
4. Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: A synthesis. *Journal of Mathematical Psychology*.

---

## Troubleshooting

### Beliefs Not Updating

- Increase `num_belief_iterations`
- Increase `learning_rate_beliefs`
- Check observation scale
- Verify generative model convergence

### Poor Policy Selection

- Increase `num_rollouts`
- Adjust `epistemic_weight` and `pragmatic_weight`
- Increase `horizon`
- Increase `num_policy_iterations`

### Numerical Instability

- Decrease learning rates
- Add gradient clipping
- Check tensor dtypes
- Normalize observations

### Performance Issues

- Reduce `num_rollouts`
- Reduce `horizon`
- Use GPU (`device='cuda'`)
- Reduce `num_belief_iterations` and `num_policy_iterations`

---

## Support

For questions and issues:
- GitHub Issues: [active_torchference/issues](https://github.com/yourusername/active_torchference/issues)
- Documentation: `docs/`
- Examples: `examples/`
- Tests: `tests/`

