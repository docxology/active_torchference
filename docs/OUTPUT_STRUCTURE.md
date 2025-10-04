# Unified Output Directory Structure

## Overview

**All outputs are organized into subdirectories of a single top-level output folder.**

This ensures clean organization, easy cleanup, and straightforward experiment management.

## Directory Structure

```
output/
└── experiment_name/
    ├── config/              # Configuration files
    │   └── config.json
    ├── logs/                # Metric logs and summaries
    │   └── metrics.json
    ├── checkpoints/         # Agent state checkpoints
    │   ├── episode_10_checkpoint.pkl
    │   └── final_checkpoint.pkl
    ├── visualizations/      # Static plots and figures
    │   ├── free_energy.png
    │   ├── beliefs.png
    │   ├── trajectory.png
    │   ├── actions.png
    │   └── summary.png
    ├── animations/          # GIF/video animations
    │   ├── trajectory_animation.gif
    │   ├── beliefs_animation.gif
    │   └── comprehensive_animation.gif
    ├── data/                # Raw episode data
    │   ├── episode_0000.json
    │   ├── episode_0001.json
    │   └── episode_0002.json
    └── metadata/            # Experiment metadata
        └── experiment_info.json
```

## Usage

### Basic Setup

```python
from active_torchference import OutputManager

# Create unified output structure
output_mgr = OutputManager(
    output_root="./output",         # Single top-level directory
    experiment_name="my_experiment"  # Experiment subdirectory
)

# Prints directory structure
output_mgr.print_structure()
```

### With Orchestrators

```python
from active_torchference import Config, ActiveInferenceAgent, OutputManager
from active_torchference.environment import ContinuousEnvironment
from active_torchference.orchestrators import (
    ExperimentRunner, Logger, Visualizer, Animator
)

# Step 1: Create OutputManager
output_mgr = OutputManager(
    output_root="./output",
    experiment_name="navigation_exp"
)

# Step 2: Pass to orchestrators
logger = Logger(output_manager=output_mgr)
visualizer = Visualizer(output_manager=output_mgr)
animator = Animator(output_manager=output_mgr)

# Step 3: Run experiment
config = Config()
agent = ActiveInferenceAgent(config)
env = ContinuousEnvironment(config)

runner = ExperimentRunner(agent, env, logger, config)
results = runner.run(num_episodes=10)

# All outputs automatically organized!
```

### Accessing Output Files

```python
# Get specific file paths
config_file = output_mgr.get_path('config', 'config.json')
metrics_file = output_mgr.get_path('logs', 'metrics.json')
checkpoint = output_mgr.get_path('checkpoints', 'final_checkpoint.pkl')
plot = output_mgr.get_path('visualizations', 'trajectory.png')
animation = output_mgr.get_path('animations', 'trajectory.gif')
episode_data = output_mgr.get_path('data', 'episode_0000.json')

# Access subdirectories directly
print(output_mgr.visualizations_dir)  # Path to visualizations/
print(output_mgr.logs_dir)            # Path to logs/
print(output_mgr.checkpoints_dir)     # Path to checkpoints/
```

## Output Categories

### 1. Config (`config/`)

**Contents**: Configuration files

- `config.json`: Agent and environment configuration
- Parameter settings
- Hyperparameters

**Purpose**: Reproducibility and parameter tracking

### 2. Logs (`logs/`)

**Contents**: Metric logs and summaries

- `metrics.json`: Time-series metrics (VFE, EFE, distance, etc.)
- Custom metrics
- Experiment summary statistics

**Purpose**: Quantitative analysis and comparison

### 3. Checkpoints (`checkpoints/`)

**Contents**: Agent state checkpoints

- `episode_N_checkpoint.pkl`: Periodic checkpoints
- `final_checkpoint.pkl`: Final agent state
- Belief and policy parameters

**Purpose**: Model saving, loading, and resuming

### 4. Visualizations (`visualizations/`)

**Contents**: Static plots and figures

- `free_energy.png`: VFE and EFE over time
- `beliefs.png`: Belief state evolution
- `trajectory.png`: Agent trajectory
- `actions.png`: Action evolution
- `summary.png`: Multi-panel summary

**Purpose**: Visual analysis and reporting

### 5. Animations (`animations/`)

**Contents**: Animated visualizations

- `trajectory_animation.gif`: Animated trajectory
- `beliefs_animation.gif`: Belief evolution animation
- `comprehensive_animation.gif`: Multi-panel animation

**Purpose**: Dynamic visualization and presentations

### 6. Data (`data/`)

**Contents**: Raw episode data

- `episode_NNNN.json`: Per-episode records
- Observations, actions, beliefs
- Environment information

**Purpose**: Detailed analysis and post-processing

### 7. Metadata (`metadata/`)

**Contents**: Experiment metadata

- Timestamps
- System information
- Git commit hashes
- Custom metadata

**Purpose**: Experiment tracking and provenance

## Managing Multiple Experiments

### Listing Experiments

```python
# List all experiments in output directory
experiments = OutputManager.list_experiments("./output")
print(f"Found {len(experiments)} experiments:")
for exp in experiments:
    print(f"  - {exp}")
```

### Automatic Naming

```python
# Auto-generate timestamped experiment name
output_mgr = OutputManager(output_root="./output")
# Creates: output/exp_20231003_143022/
```

### Explicit Naming

```python
# Use descriptive experiment name
output_mgr = OutputManager(
    output_root="./output",
    experiment_name="epistemic_exploration_v2"
)
# Creates: output/epistemic_exploration_v2/
```

## Benefits

### 1. Organization

✓ All experiment artifacts in one place  
✓ Clear categorization by type  
✓ Easy to navigate and find files

### 2. Cleanup

```python
import shutil

# Delete entire experiment
shutil.rmtree("output/experiment_name")

# Delete all experiments
shutil.rmtree("output/")
```

### 3. Portability

```python
# Archive experiment
import shutil
shutil.make_archive(
    "experiment_archive",
    'zip',
    "output/experiment_name"
)

# Share or backup complete experiment
```

### 4. Reproducibility

All files needed to reproduce experiment are in one directory:
- Configuration
- Checkpoints
- Metrics
- Data

### 5. Version Control

```gitignore
# .gitignore
output/       # Exclude all outputs
!output/.gitkeep  # Keep directory structure
```

## Example: Complete Workflow

```python
from active_torchference import *
from active_torchference.environment import ContinuousEnvironment
from active_torchference.orchestrators import *
import torch

# 1. Setup unified output
output_mgr = OutputManager(
    output_root="./output",
    experiment_name="nav_experiment_001"
)

# 2. Configure and initialize
config = Config(
    hidden_dim=8,
    obs_dim=2,
    action_dim=2,
    epistemic_weight=1.0,
    pragmatic_weight=1.0
)

agent = ActiveInferenceAgent(config)
env = ContinuousEnvironment(config, goal_position=torch.tensor([2.0, 2.0]))

# 3. Create orchestrators with unified output
logger = Logger(output_manager=output_mgr)
visualizer = Visualizer(output_manager=output_mgr)
animator = Animator(output_manager=output_mgr)

# 4. Run experiment
runner = ExperimentRunner(agent, env, logger, config)
results = runner.run(num_episodes=10, max_steps_per_episode=100)

# 5. Generate visualizations
history = agent.get_history()
visualizer.plot_comprehensive_summary(history, goal=env.get_preferred_observation())
animator.animate_comprehensive(history, goal=env.get_preferred_observation())

# 6. All outputs organized in: output/nav_experiment_001/
#    ├── config/config.json
#    ├── logs/metrics.json
#    ├── checkpoints/final_checkpoint.pkl
#    ├── visualizations/*.png
#    ├── animations/*.gif
#    └── data/episode_*.json
```

## Backward Compatibility

Old style (still works):

```python
# Legacy: separate directories
logger = Logger(log_dir="./logs", experiment_name="exp1")
visualizer = Visualizer(save_dir="./visualizations")
```

New style (recommended):

```python
# Unified: single output directory
output_mgr = OutputManager(output_root="./output", experiment_name="exp1")
logger = Logger(output_manager=output_mgr)
visualizer = Visualizer(output_manager=output_mgr)
```

Both approaches work, but the unified structure is recommended for better organization.

## Best Practices

### 1. Descriptive Names

```python
# Good
OutputManager(experiment_name="gridworld_epistemic_high")

# Less good
OutputManager(experiment_name="exp1")
```

### 2. Consistent Root

```python
# Use same output_root for all experiments
output_mgr = OutputManager(output_root="./output", ...)
```

### 3. Clean Up Old Experiments

```python
# Periodically review and archive/delete old experiments
experiments = OutputManager.list_experiments("./output")
print(f"You have {len(experiments)} experiments")
```

### 4. Archive Important Results

```python
import shutil

# Archive successful experiment
shutil.make_archive(
    "paper_results/experiment_3",
    'zip',
    "output/critical_experiment"
)
```

## Integration Example

See `examples/unified_output_example.py` for complete working example:

```bash
python3 examples/unified_output_example.py
```

This demonstrates:
- Creating unified output structure
- Running experiment with all orchestrators
- Verifying all outputs are correctly organized
- Accessing specific output files

