# Belief State and Policy Updates

## Overview

This document explains how Active Torchference correctly implements:
1. **Belief state updates via Variational Free Energy (VFE) minimization**
2. **Policy evaluation using Expected Free Energy (EFE) per policy per timestep**

## Belief State Updates

### Problem Solved
Previously, beliefs were not properly updating when receiving observations. Now they update correctly through iterative VFE minimization.

### Implementation

```python
def perceive(observation, num_iterations=5):
    """Update beliefs by iteratively minimizing VFE."""
    
    for iteration in range(num_iterations):
        # Predict observation from current beliefs
        predicted_obs = generative_model(beliefs.mean)
        
        # Compute VFE
        vfe = compute_vfe(observation, predicted_obs, beliefs, prior)
        
        # Minimize VFE via gradient descent
        vfe.backward()
        beliefs.update(gradients)
        
    return updated_beliefs
```

### Key Features

**Iterative Minimization:**
- Multiple gradient steps (default: 5 iterations)
- VFE decreases with each iteration
- Beliefs converge toward explaining observations

**VFE Components:**
```
VFE = Likelihood Term + Complexity Term
    = -E_q[log p(o|s)] + KL[q(s)||p(s)]
```

- **Likelihood**: How well beliefs predict observations
- **Complexity**: Divergence from prior beliefs

**Tracking:**
- VFE history per iteration
- Initial and final VFE values
- Belief mean and std evolution

### Example

```python
from active_torchference import Config, ActiveInferenceAgent

config = Config(num_belief_iterations=10, learning_rate_beliefs=0.1)
agent = ActiveInferenceAgent(config)

observation = torch.tensor([1.0, 2.0])
info = agent.perceive(observation)

print(f"VFE decreased: {info['vfe_initial']} → {info['vfe_final']}")
print(f"Belief mean: {info['belief_mean']}")
print(f"VFE history: {info['vfe_history']}")
```

### Configuration

```python
config = Config(
    num_belief_iterations=5,      # Number of VFE minimization steps
    learning_rate_beliefs=0.1,     # Belief update learning rate
    precision_obs=1.0,             # Observation precision
    precision_prior=1.0,           # Prior precision
)
```

## Policy Evaluation (EFE per Policy)

### Problem Solved
Previously, EFE was computed as a single averaged value. Now it's computed for each policy (action sequence) separately at each timestep.

### Implementation

```python
def evaluate_policy(belief, preferred_obs):
    """Evaluate multiple policies using EFE."""
    
    # Sample multiple candidate policies
    policies = sample_policies(num_rollouts=5)
    
    # Evaluate each policy
    efe_per_policy = []
    for policy in policies:
        # Rollout policy
        states, obs = rollout(policy, horizon=3)
        
        # Compute EFE at each timestep
        efe_timesteps = []
        for t in range(horizon):
            efe_t = compute_efe(
                predicted_obs=obs[t],
                preferred_obs=preferred_obs,
                state_entropy=entropy(states[t])
            )
            efe_timesteps.append(efe_t)
        
        # Sum across horizon
        total_efe = sum(efe_timesteps)
        efe_per_policy.append(total_efe)
    
    return efe_per_policy  # [num_rollouts]
```

### Key Features

**Per-Policy Evaluation:**
- Each policy gets its own EFE value
- Shape: `[num_rollouts]`
- Best policy has lowest EFE

**Per-Timestep Breakdown:**
- EFE computed at each planning step
- Shape: `[horizon, num_rollouts]`
- Sum over horizon = total policy EFE

**EFE Components:**
```
EFE = -(Epistemic Value + Pragmatic Value)
    = -(Information Gain + Goal Achievement)
```

- **Epistemic**: Exploration, information gain
- **Pragmatic**: Exploitation, goal achievement

**Policy Selection:**
- Best policy identified (minimum EFE)
- Policy statistics tracked (mean, min, max)
- All candidate policies available

### Example

```python
from active_torchference import Config, ActiveInferenceAgent

config = Config(
    num_rollouts=10,        # Evaluate 10 candidate policies
    horizon=3,              # Plan 3 steps ahead
    num_policy_iterations=5 # Optimize policy 5 times
)

agent = ActiveInferenceAgent(config)

observation = torch.randn(config.obs_dim)
preferred_obs = torch.tensor([1.0, 1.0])

# Perceive and plan
agent.perceive(observation)
planning_info = agent.plan(preferred_obs)

# Inspect EFE per policy
print(f"EFE per policy: {planning_info['efe_per_policy']}")
print(f"Best policy idx: {planning_info['best_policy_idx']}")
print(f"Min EFE: {planning_info['min_efe']}")
print(f"EFE per timestep:\n{planning_info['efe_per_policy_per_timestep']}")
```

### Output Structure

```python
{
    'efe_per_policy': tensor([11.2, 10.8, 11.5, 10.9, 11.3]),  # 5 policies
    'efe_per_policy_per_timestep': tensor([
        [3.8, 3.6, 3.9, 3.7, 3.8],  # timestep 0
        [3.7, 3.5, 3.8, 3.6, 3.7],  # timestep 1
        [3.7, 3.7, 3.8, 3.6, 3.8],  # timestep 2
    ]),  # shape: [horizon=3, num_rollouts=5]
    'best_policy_idx': 1,  # Policy with lowest EFE
    'min_efe': 10.8,
    'max_efe': 11.5,
    'mean_efe': 11.14,
}
```

## Integration

### Complete Action-Perception Loop

```python
from active_torchference import Config, ActiveInferenceAgent
from active_torchference.environment import ContinuousEnvironment

# Configure
config = Config(
    num_belief_iterations=10,    # Belief update iterations
    num_rollouts=10,             # Policies to evaluate
    horizon=3,                   # Planning horizon
    learning_rate_beliefs=0.1,   # Belief learning rate
    learning_rate_policy=0.05,   # Policy learning rate
)

# Initialize
agent = ActiveInferenceAgent(config)
env = ContinuousEnvironment(config)

# Reset
obs = env.reset()
preferred_obs = env.get_preferred_observation()

# Action-perception loop
for step in range(100):
    # Step 1: PERCEIVE - Update beliefs via VFE
    perception_info = agent.perceive(obs)
    print(f"VFE: {perception_info['vfe_initial']} → {perception_info['vfe_final']}")
    
    # Step 2: PLAN - Evaluate policies via EFE
    planning_info = agent.plan(preferred_obs)
    print(f"EFE per policy: {planning_info['efe_per_policy']}")
    print(f"Best policy: {planning_info['best_policy_idx']}")
    
    # Step 3: ACT - Select action
    action = agent.act()
    
    # Environment step
    obs, env_info = env.step(action)
    
    if env_info['done']:
        break
```

## Testing

### Belief Update Tests

```bash
# Test that beliefs update correctly
pytest tests/test_belief_updates.py -v
```

Tests verify:
- ✓ Beliefs change with observations
- ✓ VFE decreases over iterations
- ✓ Beliefs converge to explain data
- ✓ Learning rate affects update magnitude
- ✓ VFE history is tracked

### EFE per Policy Tests

```bash
# Test that EFE is computed per policy
pytest tests/test_efe_per_policy.py -v
```

Tests verify:
- ✓ EFE computed for each policy
- ✓ Different policies have different EFE
- ✓ Best policy identified correctly
- ✓ EFE per timestep has correct structure
- ✓ Policy optimization reduces mean EFE

## Configuration Parameters

### Belief Updates

```python
config = Config(
    # Iterations for VFE minimization
    num_belief_iterations=5,
    
    # Learning rate for belief updates
    learning_rate_beliefs=0.1,
    
    # Precision (inverse variance) parameters
    precision_obs=1.0,      # How much to trust observations
    precision_prior=1.0,    # How much to trust prior
)
```

**Guidelines:**
- More iterations → better convergence, slower
- Higher learning rate → faster updates, less stable
- Higher precision_obs → trust observations more
- Higher precision_prior → stay closer to prior

### Policy Evaluation

```python
config = Config(
    # Number of candidate policies to evaluate
    num_rollouts=5,
    
    # Planning horizon (timesteps ahead)
    horizon=3,
    
    # Iterations to optimize policy
    num_policy_iterations=10,
    
    # Learning rate for policy updates
    learning_rate_policy=0.05,
    
    # Epistemic vs pragmatic balance
    epistemic_weight=1.0,   # Exploration
    pragmatic_weight=1.0,   # Exploitation
)
```

**Guidelines:**
- More rollouts → better policy selection, slower
- Longer horizon → more foresight, computationally expensive
- Higher epistemic → more exploration
- Higher pragmatic → more goal-directed

## Visualization

### Belief Updates

```python
from active_torchference.orchestrators import Visualizer

visualizer = Visualizer()
history = agent.get_history()

# Plot belief evolution
visualizer.plot_beliefs(history)

# Plot VFE over time
visualizer.plot_free_energy(history)
```

### Policy Evaluation

Planning info contains:
- `efe_per_policy`: Visualize policy quality
- `efe_per_policy_per_timestep`: Heatmap of EFE over time and policies
- `best_policy_idx`: Highlight best policy

## Mathematical Details

### Variational Free Energy

```
VFE = -E_q[log p(o|s)] + KL[q(s)||p(s)]

where:
  q(s) = N(μ_q, σ_q²)  - Variational posterior (beliefs)
  p(s) = N(μ_p, σ_p²)  - Prior
  p(o|s) = N(g(s), σ_o²) - Likelihood (generative model)
```

**Minimizing VFE:**
- Maximize likelihood: beliefs explain observations
- Minimize complexity: beliefs stay close to prior
- Balance achieved through precision parameters

### Expected Free Energy

```
EFE = -(Epistemic + Pragmatic)

Epistemic = H[p(o)] - E[H[p(o|s)]]  
          ≈ -H[q(s)]  (information gain)

Pragmatic = E_q[log p(o|C)]
          ≈ -||o_pred - o_pref||² (goal achievement)
```

**Minimizing EFE:**
- Maximize epistemic → explore uncertain states
- Maximize pragmatic → achieve goals
- Trade-off controlled by weights

## Best Practices

1. **Start with defaults**, then tune
2. **Monitor VFE history** - should decrease
3. **Check belief convergence** - should stabilize
4. **Inspect EFE per policy** - variance indicates exploration
5. **Balance epistemic/pragmatic** based on task
6. **More iterations** if not converging
7. **Lower learning rates** if unstable

## References

- **Tests**: `tests/test_belief_updates.py`, `tests/test_efe_per_policy.py`
- **Implementation**: `active_torchference/agent.py`, `active_torchference/policy.py`
- **Config**: `active_torchference/config.py`

