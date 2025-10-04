# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/active_torchference.git
cd active_torchference

# Install dependencies
pip install -r requirements.txt

# Install package (editable mode for development)
pip install -e .
```

## 5-Minute Example

Create a simple navigation task:

```python
import torch
from active_torchference import Config, ActiveInferenceAgent
from active_torchference.environment import ContinuousEnvironment
from active_torchference.orchestrators import ExperimentRunner, Visualizer

# 1. Configure
config = Config(
    hidden_dim=8,
    obs_dim=2,
    action_dim=2,
    seed=42
)

# 2. Initialize
agent = ActiveInferenceAgent(config)
goal = torch.tensor([2.0, 2.0])
environment = ContinuousEnvironment(config, goal_position=goal)

# 3. Run
runner = ExperimentRunner(agent, environment)
results = runner.run(num_episodes=5, max_steps_per_episode=100)

# 4. Visualize
visualizer = Visualizer(save_dir="./results")
history = agent.get_history()
visualizer.plot_comprehensive_summary(history, goal=goal)
visualizer.show()
```

## Running Examples

### Example 1: Simple Navigation

```bash
python3 examples/simple_navigation.py
```

Demonstrates basic Active Inference navigation to a fixed goal.

**Key Concepts:**
- Action-perception loop
- VFE minimization for beliefs
- EFE evaluation for policies

### Example 2: Grid World Exploration

```bash
python3 examples/gridworld_exploration.py
```

Shows epistemic drive (exploration) in discrete environment.

**Key Concepts:**
- Epistemic value (information gain)
- Pragmatic value (goal achievement)
- Discrete state spaces

### Example 3: Custom Environment

```bash
python3 examples/custom_environment.py
```

Tracking oscillating goal with custom dynamics.

**Key Concepts:**
- Custom environment implementation
- Dynamic goals
- Prediction and tracking

### Example 4: Epistemic-Pragmatic Balance

```bash
python3 examples/epistemic_pragmatic_balance.py
```

Compares different exploration-exploitation trade-offs.

**Key Concepts:**
- Weight tuning
- Behavioral differences
- Comparative analysis

## Understanding the Framework

### The Action-Perception Loop

Active Torchference implements a clear 3-step loop:

```python
# Step 1: PERCEIVE - Update beliefs via VFE
perception_info = agent.perceive(observation)
# Agent minimizes: VFE = -E_q[log p(o|s)] + KL[q(s)||p(s)]

# Step 2: PLAN - Evaluate policies via EFE
planning_info = agent.plan(preferred_observation)
# Agent evaluates: EFE = -(Epistemic + Pragmatic)

# Step 3: ACT - Select action from policy posterior
action = agent.act(deterministic=False)
# No reward function - pure Active Inference
```

### Free Energy Components

**Variational Free Energy (VFE):**
- **Likelihood Term**: How well beliefs predict observations
- **Complexity Term**: Divergence from prior beliefs
- **Goal**: Minimize VFE → accurate beliefs

**Expected Free Energy (EFE):**
- **Epistemic Value**: Information gain (exploration)
- **Pragmatic Value**: Goal achievement (exploitation)
- **Goal**: Minimize EFE → optimal policies

### Configuration Tuning

Start with defaults and adjust:

```python
config = Config(
    # Architecture
    hidden_dim=8,           # Increase for complex tasks
    obs_dim=2,              # Match your observations
    action_dim=2,           # Match your actions
    
    # Learning
    learning_rate_beliefs=0.1,   # Higher = faster belief updates
    learning_rate_policy=0.05,   # Lower = more stable policies
    
    # Planning
    num_policy_iterations=10,    # More = better policies
    horizon=3,                   # Longer = more planning
    num_rollouts=5,              # More = better evaluation
    
    # Behavior
    epistemic_weight=1.0,   # Higher = more exploration
    pragmatic_weight=1.0,   # Higher = more goal-directed
    
    # Precision
    precision_obs=1.0,      # Higher = more confident in observations
    precision_prior=1.0,    # Higher = stronger prior beliefs
)
```

## Common Tasks

### Task 1: Goal-Directed Navigation

```python
from active_torchference import Config, ActiveInferenceAgent
from active_torchference.environment import ContinuousEnvironment

config = Config(
    epistemic_weight=0.5,  # Low exploration
    pragmatic_weight=2.0   # High goal-seeking
)

agent = ActiveInferenceAgent(config)
goal = torch.tensor([3.0, 3.0])
env = ContinuousEnvironment(config, goal_position=goal)

# Run and reach goal quickly
```

### Task 2: Exploration

```python
config = Config(
    epistemic_weight=2.0,  # High exploration
    pragmatic_weight=0.5,  # Low goal-seeking
    horizon=5              # Longer planning
)

# Agent will explore more before exploiting
```

### Task 3: Tracking Moving Target

```python
from active_torchference.environment import Environment

class MovingTargetEnv(Environment):
    def step(self, action):
        # Update moving goal
        self.goal = self.goal + self.velocity * 0.1
        # ... rest of implementation
```

## Debugging Tips

### Check Free Energy

```python
history = agent.get_history()

# VFE should generally decrease
import matplotlib.pyplot as plt
plt.plot(history["vfe"])
plt.title("VFE should decrease as beliefs improve")
plt.show()
```

### Verify Learning

```python
# Track distance to goal
def track_distance(step, agent_info, env_info):
    if "distance_to_goal" in env_info:
        print(f"Step {step}: Distance = {env_info['distance_to_goal']:.3f}")

runner.add_step_callback(track_distance)
```

### Inspect Beliefs

```python
# Visualize belief evolution
from active_torchference.orchestrators import Visualizer

viz = Visualizer()
history = agent.get_history()
viz.plot_beliefs(history)
viz.show()
```

## Testing Your Code

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=active_torchference tests/

# Run specific test
pytest tests/test_agent.py::test_agent_step
```

## Next Steps

1. **Explore Examples**: Run all example scripts
2. **Read API Docs**: Check `docs/API.md` for detailed API
3. **Customize**: Create your own environments
4. **Experiment**: Tune epistemic/pragmatic weights
5. **Contribute**: Add new features following TDD

## Common Issues

### Issue: Agent not learning

**Solution:** Increase learning rates
```python
config = Config(
    learning_rate_beliefs=0.2,  # Increase
    learning_rate_policy=0.1    # Increase
)
```

### Issue: Agent too exploratory

**Solution:** Reduce epistemic weight
```python
config = Config(
    epistemic_weight=0.3,   # Decrease
    pragmatic_weight=1.5    # Increase
)
```

### Issue: Unstable behavior

**Solution:** Reduce learning rates, increase precision
```python
config = Config(
    learning_rate_beliefs=0.05,  # Decrease
    learning_rate_policy=0.02,   # Decrease
    precision_obs=2.0            # Increase
)
```

## Resources

- **Web Search Results**: Active Inference theory and implementations
- **API Documentation**: `docs/API.md`
- **Examples Directory**: `examples/`
- **Test Suite**: `tests/` - shows usage patterns
- **README**: Project overview and architecture

## Getting Help

1. Check examples directory for similar use cases
2. Review test files for usage patterns
3. Read API documentation
4. Examine framework source code (well-commented)

Happy inferring! 🧠

