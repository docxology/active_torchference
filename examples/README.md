# Active Torchference Examples

Practical demonstrations of Active Inference agents in various tasks and configurations.

## Overview

All examples use the unified output directory structure, organizing results into:
- `output/experiment_name/config/` - Configuration files
- `output/experiment_name/logs/` - Metrics and summaries
- `output/experiment_name/checkpoints/` - Agent checkpoints
- `output/experiment_name/visualizations/` - Static plots
- `output/experiment_name/animations/` - GIF animations
- `output/experiment_name/data/` - Episode records

## Examples

### 1. Unified Output Example (★ Recommended First)

**File**: `unified_output_example.py`

Comprehensive demonstration of unified output directory structure and all orchestration tools.

**Run**:
```bash
python3 examples/unified_output_example.py
```

**Features**:
- Complete unified output structure
- All orchestrators (Logger, Visualizer, Animator)
- Output verification and validation
- Directory structure walkthrough

**Use Case**: Learn the framework's output organization system.

---

### 2. Simple Navigation

**File**: `simple_navigation.py`

Agent navigates to fixed goal in continuous 2D space.

**Run**:
```bash
python3 examples/simple_navigation.py
```

**Features**:
- Basic action-perception loop
- VFE minimization for beliefs
- EFE evaluation for policies
- Trajectory visualization
- Comprehensive animations

**Configuration**:
- Balanced epistemic/pragmatic weights (0.5/1.0)
- Horizon: 3 steps
- Episodes: 5

**Use Case**: Understand fundamental Active Inference navigation.

---

### 3. Grid World Exploration

**File**: `gridworld_exploration.py`

Discrete grid world navigation with epistemic drive.

**Run**:
```bash
python3 examples/gridworld_exploration.py
```

**Features**:
- Discrete state space (5×5 grid)
- High epistemic weight (1.5) for exploration
- Position tracking
- Coverage metrics

**Configuration**:
- Epistemic weight: 1.5 (high exploration)
- Pragmatic weight: 1.0
- Horizon: 4 steps
- Episodes: 3

**Use Case**: Explore epistemic drive and information-seeking behavior.

---

### 4. Custom Environment

**File**: `custom_environment.py`

Track oscillating goal using OscillatorEnvironment.

**Run**:
```bash
python3 examples/custom_environment.py
```

**Features**:
- Dynamic oscillating goal
- Sinusoidal target motion
- Tracking error metrics
- Prediction and adaptation

**Configuration**:
- Oscillation frequency: 0.05
- Amplitude: 2.0
- Longer episodes (200 steps)
- Higher learning rates

**Use Case**: Demonstrate custom environment implementation and dynamic goal tracking.

---

### 5. Epistemic-Pragmatic Balance

**File**: `epistemic_pragmatic_balance.py`

Comparative analysis of exploration vs exploitation.

**Run**:
```bash
python3 examples/epistemic_pragmatic_balance.py
```

**Features**:
- Three configurations compared:
  - **Pure Exploration**: epistemic=2.0, pragmatic=0.1
  - **Balanced**: epistemic=1.0, pragmatic=1.0
  - **Pure Exploitation**: epistemic=0.1, pragmatic=2.0
- Side-by-side trajectory comparison
- Comparative metrics visualization

**Use Case**: Understand epistemic-pragmatic trade-offs and tune behavior.

---

## Running All Examples

Use the unified runner script:

```bash
python3 run_all_examples.py
```

This runs all examples with comprehensive validation, generates a detailed report, and verifies all outputs.

## Output Structure

After running any example, outputs are organized as:

```
output/
└── experiment_name/
    ├── config/
    │   └── config.json
    ├── logs/
    │   └── metrics.json
    ├── checkpoints/
    │   ├── episode_N_checkpoint.pkl
    │   └── final_checkpoint.pkl
    ├── visualizations/
    │   ├── free_energy.png
    │   ├── beliefs.png
    │   ├── trajectory.png
    │   ├── actions.png
    │   └── summary.png
    ├── animations/
    │   ├── trajectory_animation.gif
    │   ├── beliefs_animation.gif
    │   └── comprehensive_animation.gif
    └── data/
        ├── episode_0000.json
        ├── episode_0001.json
        └── ...
```

## Creating Custom Examples

Template for new examples:

```python
import torch
from active_torchference import Config, ActiveInferenceAgent, OutputManager
from active_torchference.environment import ContinuousEnvironment
from active_torchference.orchestrators import ExperimentRunner, Logger, Visualizer

def main():
    # Step 1: Create output structure
    output_mgr = OutputManager(
        output_root="./output",
        experiment_name="my_custom_example"
    )
    
    # Step 2: Configure
    config = Config(
        hidden_dim=8,
        obs_dim=2,
        action_dim=2,
        epistemic_weight=1.0,
        pragmatic_weight=1.0,
        seed=42
    )
    
    # Step 3: Initialize
    agent = ActiveInferenceAgent(config)
    environment = ContinuousEnvironment(config)
    
    # Step 4: Setup orchestrators
    logger = Logger(output_manager=output_mgr)
    visualizer = Visualizer(output_manager=output_mgr)
    
    # Step 5: Run
    runner = ExperimentRunner(agent, environment, logger, config)
    results = runner.run(num_episodes=10)
    
    # Step 6: Visualize
    history = agent.get_history()
    visualizer.plot_comprehensive_summary(history)
    
    print(f"✓ Complete! Outputs in: {output_mgr.experiment_dir}")

if __name__ == "__main__":
    main()
```

## Key Concepts Demonstrated

### Action-Perception Loop
All examples demonstrate the core loop:
1. **Perceive**: Update beliefs via VFE minimization
2. **Plan**: Evaluate policies via EFE
3. **Act**: Select action from policy posterior

### Free Energy Computations
- **VFE**: Belief updates driven by likelihood + complexity
- **EFE**: Policy selection driven by epistemic + pragmatic values

### Configuration Tuning
Examples show different parameter settings for various behaviors:
- **Exploration**: High epistemic weight
- **Exploitation**: High pragmatic weight
- **Precision**: Faster learning with higher learning rates
- **Planning**: Longer horizons for more foresight

## Troubleshooting

### Example doesn't run
```bash
# Verify installation
pip install -e .

# Check dependencies
pip install -r requirements.txt
```

### No outputs generated
Check that `output/` directory permissions allow writing.

### Visualizations empty
Ensure examples run for sufficient steps to generate data.

## Further Reading

- **[AGENTS.md](../docs/AGENTS.md)**: Detailed agent documentation
- **[QUICKSTART.md](../docs/QUICKSTART.md)**: Step-by-step tutorial
- **[API.md](../docs/API.md)**: Complete API reference
- **[OUTPUT_STRUCTURE.md](../docs/OUTPUT_STRUCTURE.md)**: Output organization details

## Contributing Examples

When adding new examples:
1. Use unified output structure via `OutputManager`
2. Include docstring explaining purpose
3. Add entry to this README
4. Follow naming convention: `descriptive_name.py`
5. Test with `run_all_examples.py`

