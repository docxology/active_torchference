# Orchestrators

High-level tools for running, logging, visualizing, and animating Active Inference experiments.

## Overview

Orchestrators provide a unified interface for experiment management, separating concerns between agent logic and experiment infrastructure.

## Modules

### `runner.py`
Orchestrates experiment execution with agent-environment interaction.

**Class**: `ExperimentRunner`

**Initialization**:
```python
from active_torchference.orchestrators import ExperimentRunner

runner = ExperimentRunner(
    agent=agent,
    environment=environment,
    logger=logger,
    config=config
)
```

**Methods**:

**`run(num_episodes, max_steps_per_episode, deterministic, render, save_every, verbose)`**
- Runs multiple episodes
- Handles episode management
- Saves checkpoints periodically
- Returns results dictionary

**`run_episode(max_steps, deterministic, render)`**
- Runs single episode
- Returns episode data

**`evaluate(num_episodes, max_steps_per_episode, verbose)`**
- Evaluation mode (deterministic)
- No checkpoint saving

**`add_step_callback(callback)`**
- Register callback: `callback(step, agent_info, env_info)`

**`add_episode_callback(callback)`**
- Register callback: `callback(episode, episode_data)`

**Example**:
```python
runner = ExperimentRunner(agent, env, logger, config)

# Add callbacks
def step_callback(step, agent_info, env_info):
    print(f"Step {step}: VFE = {agent_info['vfe']:.4f}")

runner.add_step_callback(step_callback)

# Run
results = runner.run(
    num_episodes=10,
    max_steps_per_episode=100,
    verbose=True
)
```

---

### `logger.py`
Comprehensive experiment logging system.

**Class**: `Logger`

**Initialization**:
```python
from active_torchference.orchestrators import Logger
from active_torchference import OutputManager

# With OutputManager (recommended)
output_mgr = OutputManager(experiment_name="my_experiment")
logger = Logger(output_manager=output_mgr)

# Legacy mode
logger = Logger(log_dir="./logs", experiment_name="my_experiment")
```

**Methods**:

**`log_step(timestep, agent_info, env_info)`**
- Log data from single step
- Tracks VFE, EFE, distance, custom metrics

**`log_episode(episode_num, episode_data)`**
- Log episode summary
- Saves to `data/episode_NNNN.json`

**`log_custom(key, value)`**
- Log custom metrics
- Stored in `metrics["custom"][key]`

**`save_agent_state(agent, checkpoint_name)`**
- Save agent checkpoint
- Saved to `checkpoints/{checkpoint_name}_checkpoint.pkl`

**`save_metrics(filename)`**
- Save accumulated metrics
- Saved to `logs/metrics.json`

**`save_config(config)`**
- Save configuration
- Saved to `config/config.json`

**`get_summary()`**
- Returns summary statistics dict

**`print_summary()`**
- Prints formatted summary to console

**Output Structure**:
```
experiment_name/
├── config/
│   └── config.json
├── logs/
│   └── metrics.json
├── checkpoints/
│   ├── episode_N_checkpoint.pkl
│   └── final_checkpoint.pkl
└── data/
    ├── episode_0000.json
    └── episode_0001.json
```

**Example**:
```python
logger = Logger(output_manager=output_mgr)

# Log during experiment
logger.log_step(step, agent_info, env_info)
logger.log_custom("my_metric", value)

# Save
logger.save_agent_state(agent, "final")
logger.save_metrics()
logger.save_config(config)

# Summary
logger.print_summary()
```

---

### `visualizer.py`
Creates static visualizations of experiment results.

**Class**: `Visualizer`

**Initialization**:
```python
from active_torchference.orchestrators import Visualizer

# With OutputManager (recommended)
visualizer = Visualizer(output_manager=output_mgr)

# Legacy mode
visualizer = Visualizer(save_dir="./visualizations")
```

**Methods**:

**`plot_free_energy(history, title)`**
- Plots VFE and EFE over time
- Two subplots (VFE, EFE)
- Saved to `visualizations/free_energy.png`

**`plot_beliefs(history, title)`**
- Plots belief state evolution
- One subplot per dimension (up to 4)
- Saved to `visualizations/beliefs.png`

**`plot_trajectory_2d(history, goal, title)`**
- Plots 2D trajectory
- Shows start, end, goal
- Saved to `visualizations/trajectory.png`

**`plot_actions(history, title)`**
- Plots action evolution
- One subplot per action dimension
- Saved to `visualizations/actions.png`

**`plot_comprehensive_summary(history, goal)`**
- Multi-panel summary figure
- 6 subplots: VFE, EFE, trajectory, beliefs, actions, statistics
- Saved to `visualizations/summary.png`

**`show()`**
- Display all plots

**`close_all()`**
- Close all figures

**Example**:
```python
visualizer = Visualizer(output_manager=output_mgr)

history = agent.get_history()
goal = env.get_preferred_observation()

# Create plots
visualizer.plot_free_energy(history)
visualizer.plot_beliefs(history)
visualizer.plot_trajectory_2d(history, goal=goal)
visualizer.plot_comprehensive_summary(history, goal=goal)

# Display or just save (already saved automatically)
visualizer.show()
```

---

### `animator.py`
Generates animated visualizations.

**Class**: `Animator`

**Initialization**:
```python
from active_torchference.orchestrators import Animator

# With OutputManager (recommended)
animator = Animator(output_manager=output_mgr)

# Legacy mode
animator = Animator(save_dir="./animations")
```

**Methods**:

**`animate_trajectory_2d(history, goal, fps, filename)`**
- Animates 2D trajectory
- Shows agent moving toward goal
- Saved to `animations/trajectory_animation.gif`

**`animate_beliefs(history, fps, filename)`**
- Animates belief evolution
- Plots belief dimensions over time
- Saved to `animations/beliefs_animation.gif`

**`animate_comprehensive(history, goal, fps, filename)`**
- Multi-panel comprehensive animation
- 4 subplots: trajectory, VFE, EFE, beliefs
- Saved to `animations/comprehensive_animation.gif`

**Parameters**:
- `fps`: Frames per second (default: 10)
- `filename`: Output filename

**Example**:
```python
animator = Animator(output_manager=output_mgr)

history = agent.get_history()
goal = env.get_preferred_observation()

# Create animations
animator.animate_trajectory_2d(history, goal=goal, fps=10)
animator.animate_beliefs(history, fps=10)
animator.animate_comprehensive(history, goal=goal, fps=5)

# Files automatically saved to animations/
```

---

## Unified Workflow

Recommended workflow using all orchestrators:

```python
from active_torchference import (
    Config, ActiveInferenceAgent, ContinuousEnvironment, OutputManager
)
from active_torchference.orchestrators import (
    ExperimentRunner, Logger, Visualizer, Animator
)

# 1. Setup output structure
output_mgr = OutputManager(
    output_root="./output",
    experiment_name="my_experiment"
)

# 2. Configure and initialize
config = Config(
    hidden_dim=8,
    obs_dim=2,
    action_dim=2
)
agent = ActiveInferenceAgent(config)
environment = ContinuousEnvironment(config)

# 3. Create orchestrators
logger = Logger(output_manager=output_mgr)
visualizer = Visualizer(output_manager=output_mgr)
animator = Animator(output_manager=output_mgr)

# 4. Run experiment
runner = ExperimentRunner(agent, environment, logger, config)
results = runner.run(num_episodes=10, max_steps_per_episode=100)

# 5. Visualize
history = agent.get_history()
goal = environment.get_preferred_observation()

visualizer.plot_comprehensive_summary(history, goal=goal)
animator.animate_comprehensive(history, goal=goal)

print(f"✓ Complete! Outputs in: {output_mgr.experiment_dir}")
```

---

## Output Organization

All orchestrators use `OutputManager` for unified structure:

```
output/
└── experiment_name/
    ├── config/
    │   └── config.json              # Logger
    ├── logs/
    │   └── metrics.json             # Logger
    ├── checkpoints/
    │   └── final_checkpoint.pkl     # Logger
    ├── visualizations/
    │   ├── free_energy.png          # Visualizer
    │   ├── beliefs.png              # Visualizer
    │   ├── trajectory.png           # Visualizer
    │   └── summary.png              # Visualizer
    ├── animations/
    │   ├── trajectory_animation.gif # Animator
    │   └── comprehensive_animation.gif # Animator
    └── data/
        └── episode_0000.json        # Logger
```

---

## Callbacks

Callbacks enable custom behavior during experiments.

### Step Callbacks

Called after each step:
```python
def custom_step_callback(step, agent_info, env_info):
    if step % 10 == 0:
        print(f"Step {step}: Distance = {env_info.get('distance_to_goal', 'N/A')}")
    
    # Log custom metrics
    if "custom_metric" in agent_info:
        logger.log_custom("my_metric", agent_info["custom_metric"])

runner.add_step_callback(custom_step_callback)
```

### Episode Callbacks

Called after each episode:
```python
def custom_episode_callback(episode, episode_data):
    print(f"Episode {episode} complete!")
    print(f"  Total steps: {episode_data['steps']}")
    print(f"  Average VFE: {episode_data['avg_vfe']:.4f}")

runner.add_episode_callback(custom_episode_callback)
```

---

## Integration with OutputManager

All orchestrators accept `OutputManager` for unified organization:

```python
output_mgr = OutputManager(experiment_name="my_experiment")

# Pass to all orchestrators
logger = Logger(output_manager=output_mgr)
visualizer = Visualizer(output_manager=output_mgr)
animator = Animator(output_manager=output_mgr)

# All outputs go to: output/my_experiment/
```

**Benefits**:
- Single top-level output directory
- Organized subdirectories by type
- Easy cleanup and archiving
- Consistent structure across experiments

---

## Related Documentation

- **[../README.md](../README.md)**: Package overview
- **[../../docs/OUTPUT_STRUCTURE.md](../../docs/OUTPUT_STRUCTURE.md)**: Output organization details
- **[../../examples/README.md](../../examples/README.md)**: Example usage
- **[../../docs/API.md](../../docs/API.md)**: Complete API reference

---

## Design Principles

1. **Separation of Concerns**: Agent logic separate from experiment management
2. **Composability**: Mix and match orchestrators as needed
3. **Unified Output**: All outputs organized consistently
4. **Extensibility**: Easy to add custom callbacks
5. **Clarity**: Self-documenting API

