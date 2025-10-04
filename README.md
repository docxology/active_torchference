# Active Torchference

PyTorch-based framework for Active Inference with clear action-perception loops.

## Overview

Active Torchference implements Active Inference agents that minimize variational and expected free energy without reinforcement learning. The framework provides modular components for building, running, and analyzing Active Inference models.

## Core Principles

**Action-Perception Loop:**
1. **Perceive**: Agent receives observation and updates beliefs by minimizing Variational Free Energy (VFE)
2. **Plan**: Agent evaluates policies through rollouts using Expected Free Energy (EFE = epistemic + pragmatic value)
3. **Act**: Agent selects action from updated policy posterior (no reward function)

**Key Features:**
- Pure Active Inference (no reinforcement learning)
- Modular architecture with clear separation of concerns
- Comprehensive test suite with TDD principles
- Orchestrators for running, visualization, animation, and logging
- Flexible configuration system
- Multiple environment implementations

## Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/active_torchference.git
cd active_torchference

# Install package (this also installs dependencies)
pip install -e .

# Verify installation
make check-env
```

### Using Makefile (Recommended)

```bash
make install      # Install package
make check-env    # Verify installation
make test         # Run test suite
make examples     # Run all examples
make help         # Show all commands
```

### Alternative: Virtual Environment (Best Practice)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install package
pip install -e .

# Verify
python check_installation.py
```

### Verify Installation

Run the installation checker:
```bash
python check_installation.py
```

Expected output:
```
✅ All checks passed!
You're ready to use Active Torchference.
```

### Troubleshooting

If you encounter `ModuleNotFoundError` or other issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for comprehensive solutions.

## Quick Start

```python
from active_torchference import Config, ActiveInferenceAgent, OutputManager
from active_torchference.environment import ContinuousEnvironment
from active_torchference.orchestrators import ExperimentRunner, Logger, Visualizer

# Step 1: Create unified output structure
output_mgr = OutputManager(
    output_root="./output",
    experiment_name="my_experiment"
)

# Step 2: Configure agent
config = Config(
    hidden_dim=8,
    obs_dim=2,
    action_dim=2,
    epistemic_weight=1.0,  # Exploration
    pragmatic_weight=1.0,  # Exploitation
)

# Step 3: Initialize agent and environment
agent = ActiveInferenceAgent(config)
environment = ContinuousEnvironment(config)

# Step 4: Create orchestrators with unified output
logger = Logger(output_manager=output_mgr)
visualizer = Visualizer(output_manager=output_mgr)

# Step 5: Run experiment
runner = ExperimentRunner(agent, environment, logger, config)
results = runner.run(num_episodes=10, max_steps_per_episode=100)

# All outputs organized in: output/my_experiment/
#   ├── config/           # Configuration files
#   ├── logs/             # Metrics and summaries
#   ├── checkpoints/      # Agent checkpoints
#   ├── visualizations/   # Plots and figures
#   ├── animations/       # GIF animations
#   └── data/             # Episode records
```

## Examples

### Run All Examples
```bash
# Run all examples with comprehensive validation
python3 run_all_examples.py
```
Runs all examples in safe-to-fail mode with full validation and logging. Generates detailed report of outputs and saved files.

### Individual Examples

#### Unified Output Structure (Recommended)
```bash
python3 examples/unified_output_example.py
```
Demonstrates complete unified output directory structure with all outputs organized into subdirectories.

#### Simple Navigation
```bash
python3 examples/simple_navigation.py
```
Agent navigates to goal in continuous 2D space.

#### Grid World Exploration
```bash
python3 examples/gridworld_exploration.py
```
Agent explores discrete grid world with epistemic drive.

#### Custom Environment
```bash
python3 examples/custom_environment.py
```
Track oscillating goal with custom dynamics (OscillatorEnvironment).

#### Epistemic-Pragmatic Balance
```bash
python3 examples/epistemic_pragmatic_balance.py
```
Compare exploration vs exploitation behavior across multiple configurations.

## Architecture

### Core Modules

**`active_torchference/`**
- `config.py`: Configuration system
- `agent.py`: Active Inference agent with action-perception loop
- `beliefs.py`: Belief state management and generative model
- `free_energy.py`: VFE and EFE computations
- `policy.py`: Policy evaluation and selection
- `environment.py`: Environment base class and implementations
- `utils.py`: Utility functions

**`active_torchference/orchestrators/`**
- `runner.py`: Experiment orchestration
- `logger.py`: Logging system
- `visualizer.py`: Plotting tools
- `animator.py`: Animation tools

### Free Energy Computations

**Variational Free Energy (VFE):**
```
VFE = -E_q[log p(o|s)] + KL[q(s)||p(s)]
    = Likelihood Term + Complexity Term
```
Minimizing VFE updates beliefs about hidden states.

**Expected Free Energy (EFE):**
```
EFE = -(Epistemic Value + Pragmatic Value)
    = -(Information Gain + Goal Achievement)
```
Minimizing EFE selects policies that balance exploration and exploitation.

## Configuration

```python
config = Config(
    # Dimensions
    hidden_dim=4,           # Hidden state dimensionality
    obs_dim=2,              # Observation dimensionality
    action_dim=2,           # Action dimensionality
    
    # Learning rates
    learning_rate_beliefs=0.1,   # Belief update rate
    learning_rate_policy=0.05,   # Policy update rate
    
    # Policy evaluation
    num_policy_iterations=10,    # Policy optimization steps
    num_rollouts=5,              # Rollouts per evaluation
    horizon=3,                   # Planning horizon
    
    # Free energy weights
    epistemic_weight=1.0,   # Exploration (information gain)
    pragmatic_weight=1.0,   # Exploitation (goal achievement)
    
    # Precision parameters
    precision_obs=1.0,      # Observation precision
    precision_prior=1.0,    # Prior precision
    
    # Device
    device="cpu",           # or "cuda"
    seed=42                 # Random seed
)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_agent.py
pytest tests/test_belief_updates.py
pytest tests/test_efe_per_policy.py
pytest tests/test_output_manager.py

# Run with coverage
pytest --cov=active_torchference tests/
```

## Validation

The framework includes comprehensive validation scripts that rigorously verify core computations:

```bash
# Validate VFE computation
python3 validation/validate_vfe_computation.py

# Validate EFE computation
python3 validation/validate_efe_computation.py
```

**VFE Validation** ensures:
- ✓ VFE computed at each timestep
- ✓ All latent dimensions actively updated
- ✓ VFE decreases with iterative minimization
- ✓ Components (likelihood + complexity) sum correctly
- ✓ Beliefs respond to observations

**EFE Validation** ensures:
- ✓ EFE computed per policy (not averaged)
- ✓ EFE computed per timestep in planning horizon
- ✓ Epistemic and pragmatic components present
- ✓ Different policies have different EFE values
- ✓ Best policy correctly identified
- ✓ Weights affect EFE appropriately

Both scripts generate validation plots in `validation/outputs/`.

## Visualization

The framework provides comprehensive visualization tools:

```python
from active_torchference.orchestrators import Visualizer, Animator

visualizer = Visualizer(save_dir="./results")
animator = Animator(save_dir="./results")

# Get agent history
history = agent.get_history()

# Create plots
visualizer.plot_free_energy(history)
visualizer.plot_beliefs(history)
visualizer.plot_trajectory_2d(history, goal=goal_position)
visualizer.plot_comprehensive_summary(history)

# Create animations
animator.animate_trajectory_2d(history, goal=goal_position)
animator.animate_comprehensive(history, goal=goal_position)

visualizer.show()
```

## Custom Environments

Create custom environments by subclassing `Environment`:

```python
from active_torchference.environment import Environment

class CustomEnvironment(Environment):
    def reset(self):
        # Initialize environment
        return observation
    
    def step(self, action):
        # Process action
        # Update state
        return observation, info
    
    def get_preferred_observation(self):
        # Return goal observation
        return preferred_obs
```

## Logging

Comprehensive experiment logging:

```python
from active_torchference.orchestrators import Logger

logger = Logger(log_dir="./logs", experiment_name="my_experiment")

# Automatically logs during experiments
runner = ExperimentRunner(agent, environment, logger)
results = runner.run(num_episodes=10)

# Save agent checkpoints
logger.save_agent_state(agent, checkpoint_name="final")

# Save metrics
logger.save_metrics()

# Print summary
logger.print_summary()
```

## Documentation

Comprehensive documentation available:

- **[AGENTS.md](docs/AGENTS.md)**: Complete guide to Active Inference agents, action-perception loops, belief updates, and policy evaluation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System design and component interactions
- **[API.md](docs/API.md)**: Complete API reference for all modules
- **[QUICKSTART.md](docs/QUICKSTART.md)**: Step-by-step tutorial for getting started
- **[OUTPUT_STRUCTURE.md](docs/OUTPUT_STRUCTURE.md)**: Unified output directory organization
- **[BELIEF_AND_POLICY_UPDATES.md](docs/BELIEF_AND_POLICY_UPDATES.md)**: Technical details on VFE and EFE minimization

## Project Structure

```
active_torchference/
├── active_torchference/        # Main package
│   ├── __init__.py
│   ├── agent.py                # Active Inference agent
│   ├── beliefs.py              # Belief state and generative model
│   ├── config.py               # Configuration system
│   ├── environment.py          # Environment implementations
│   ├── free_energy.py          # VFE and EFE calculations
│   ├── policy.py               # Policy evaluation and selection
│   ├── utils.py                # Utility functions
│   ├── output_manager.py       # Unified output management
│   └── orchestrators/          # Orchestration tools
│       ├── __init__.py
│       ├── animator.py         # Animation generation
│       ├── logger.py           # Experiment logging
│       ├── runner.py           # Experiment execution
│       └── visualizer.py       # Visualization tools
├── tests/                      # Comprehensive test suite
│   ├── test_agent.py
│   ├── test_beliefs.py
│   ├── test_belief_updates.py  # VFE minimization tests
│   ├── test_config.py
│   ├── test_environment.py
│   ├── test_free_energy.py
│   ├── test_efe_per_policy.py  # EFE computation tests
│   ├── test_integration.py
│   ├── test_output_manager.py
│   ├── test_policy.py
│   └── test_utils.py
├── validation/                 # Validation scripts
│   ├── validate_vfe_computation.py
│   ├── validate_efe_computation.py
│   └── outputs/                # Validation plots
├── examples/                   # Example scripts
│   ├── unified_output_example.py
│   ├── simple_navigation.py
│   ├── gridworld_exploration.py
│   ├── custom_environment.py
│   └── epistemic_pragmatic_balance.py
├── docs/                       # Documentation
│   ├── AGENTS.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── QUICKSTART.md
│   ├── OUTPUT_STRUCTURE.md
│   └── BELIEF_AND_POLICY_UPDATES.md
├── output/                     # Experiment outputs (generated)
├── requirements.txt
├── setup.py
└── README.md
```

## References

Active Inference is based on the Free Energy Principle:

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Friston, K., et al. (2017). Active inference: a process theory.
- Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.

## License

MIT License

## Contributing

Contributions welcome. Follow TDD principles and maintain modular architecture.

