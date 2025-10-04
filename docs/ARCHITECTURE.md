# Architecture Documentation

## Overview

Active Torchference implements a modular Active Inference framework with clear separation of concerns and adherence to software engineering best practices.

## Design Principles

1. **Modularity**: Each component has single responsibility
2. **Testability**: Comprehensive test coverage with TDD approach
3. **Clarity**: Explicit action-perception loop implementation
4. **Extensibility**: Easy to extend with custom environments and models
5. **Documentation**: Code as documentation (understated, "show not tell")

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Active Inference Agent                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Beliefs  │  │   Free   │  │  Policy  │  │ Generative│   │
│  │  State   │  │  Energy  │  │Evaluator │  │   Model   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │              ▲
                    Action              Observation
                         │              │
                         ▼              │
┌─────────────────────────────────────────────────────────────┐
│                       Environment                            │
├─────────────────────────────────────────────────────────────┤
│  Provides: Observations, Preferred Observations, Dynamics   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Orchestrators                          │
├─────────────────────────────────────────────────────────────┤
│  Runner  │  Logger  │  Visualizer  │  Animator             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System (`config.py`)

**Purpose**: Centralized parameter management

**Design Pattern**: Dataclass with validation

**Key Features**:
- Type-safe parameters
- Serialization/deserialization
- Validation on initialization
- Device and seed management

```python
@dataclass
class Config:
    hidden_dim: int = 4
    obs_dim: int = 2
    action_dim: int = 2
    # ... with post_init validation
```

### 2. Belief State Management (`beliefs.py`)

**Purpose**: Manage approximate posterior q(s) over hidden states

**Components**:
- `BeliefState`: Gaussian distribution over states
- `GenerativeModel`: Neural network p(o|s)

**Design Pattern**: Separation of distribution and transformation

**Key Operations**:
- Sample from distribution
- Update via gradient descent
- Uncertainty propagation

### 3. Free Energy Computations (`free_energy.py`)

**Purpose**: Calculate VFE and EFE for inference and planning

**Components**:
- `VariationalFreeEnergy`: For belief updating
- `ExpectedFreeEnergy`: For policy evaluation

**Mathematical Foundation**:

```
VFE = -E_q[log p(o|s)] + KL[q(s)||p(s)]
    = Likelihood Term + Complexity Term

EFE = -(Epistemic Value + Pragmatic Value)
    = -(Information Gain + Goal Achievement)
```

**Design Pattern**: Strategy pattern for different free energy types

### 4. Policy Management (`policy.py`)

**Purpose**: Evaluate and select actions

**Components**:
- `TransitionModel`: Predicts state transitions p(s'|s,a)
- `PolicyEvaluator`: Evaluates policies via rollouts

**Design Pattern**: Model-based planning with gradient descent

**Key Operations**:
- Sample candidate actions
- Perform rollouts
- Evaluate with EFE
- Update policy distribution

### 5. Active Inference Agent (`agent.py`)

**Purpose**: Orchestrate action-perception loop

**Design Pattern**: Facade pattern coordinating all components

**Action-Perception Loop**:

```python
def step(observation, preferred_obs):
    # 1. PERCEIVE: Update beliefs via VFE
    perception_info = self.perceive(observation)
    
    # 2. PLAN: Evaluate policies via EFE
    planning_info = self.plan(preferred_obs)
    
    # 3. ACT: Select from policy posterior
    action = self.act()
    
    return action, info
```

**Key Features**:
- Stateful belief and policy tracking
- History accumulation
- State save/load

### 6. Environment Abstraction (`environment.py`)

**Purpose**: Provide unified interface for environments

**Design Pattern**: Abstract base class with implementations

**Required Methods**:
- `reset()`: Initialize environment
- `step(action)`: Process action, return observation
- `get_preferred_observation()`: Return goal

**Implementations**:
- `ContinuousEnvironment`: Continuous navigation
- `GridWorld`: Discrete grid navigation

## Orchestrator Layer

### ExperimentRunner (`runner.py`)

**Purpose**: Manage experiment execution

**Features**:
- Episode management
- Callback system
- Progress tracking
- Checkpoint saving

**Design Pattern**: Observer pattern with callbacks

### Logger (`logger.py`)

**Purpose**: Comprehensive experiment logging

**Features**:
- Step-level metrics
- Episode summaries
- Custom metrics
- Multiple formats (JSON, pickle)

**Design Pattern**: Structured logging with serialization

### Visualizer (`visualizer.py`)

**Purpose**: Create static visualizations

**Features**:
- Free energy plots
- Trajectory visualization
- Belief evolution
- Multi-panel summaries

**Design Pattern**: Builder pattern for complex figures

### Animator (`animator.py`)

**Purpose**: Generate animations

**Features**:
- Trajectory animation
- Belief evolution animation
- Comprehensive multi-panel animation

**Design Pattern**: Frame-by-frame rendering with matplotlib

## Data Flow

### Forward Pass (Action Selection)

```
Observation → BeliefState → GenerativeModel → VFE → Belief Update
                    ↓
              BeliefState → TransitionModel → Rollouts
                    ↓
              Predicted Outcomes → EFE → Policy Update
                    ↓
              PolicyEvaluator → Action Selection → Action
```

### Backward Pass (Learning)

```
VFE.backward() → Belief Gradients → Belief Update
EFE.backward() → Policy Gradients → Policy Update
```

## Testing Strategy

### Test Hierarchy

1. **Unit Tests**: Individual component functionality
   - `test_config.py`: Configuration validation
   - `test_utils.py`: Utility functions
   - `test_free_energy.py`: Free energy calculations
   - `test_beliefs.py`: Belief management
   - `test_policy.py`: Policy evaluation
   - `test_environment.py`: Environment implementations

2. **Integration Tests**: Component interactions
   - `test_agent.py`: Agent behavior
   - `test_integration.py`: Full system integration

3. **Property Tests**: Invariants and properties
   - VFE non-negativity
   - Probability sum to 1
   - Gradient flow

### Test Coverage

```bash
pytest --cov=active_torchference tests/
# Target: >90% coverage
```

## Extension Points

### Custom Environments

```python
class CustomEnvironment(Environment):
    def reset(self): ...
    def step(self, action): ...
    def get_preferred_observation(self): ...
```

### Custom Generative Models

```python
class CustomGenerativeModel(nn.Module):
    def __init__(self, config):
        # Custom architecture
        pass
    
    def forward(self, hidden_state):
        # Custom prediction
        pass
```

### Custom Free Energy

```python
class CustomFreeEnergy:
    def compute(self, ...):
        # Custom computation
        pass
```

### Custom Callbacks

```python
def custom_step_callback(step, agent_info, env_info):
    # Custom logging/processing
    pass

runner.add_step_callback(custom_step_callback)
```

## Performance Considerations

### Computational Bottlenecks

1. **Policy Rollouts**: O(num_rollouts × horizon)
   - Solution: Reduce rollouts or horizon
   - Parallelize with batch operations

2. **Policy Optimization**: O(num_policy_iterations)
   - Solution: Early stopping on convergence

3. **Belief Updates**: O(hidden_dim)
   - Solution: Use efficient neural networks

### Memory Usage

- History tracking accumulates over time
- Clear history with `agent.reset()` between episodes
- Use batch operations for efficiency

## Best Practices

### Configuration

1. Start with default config
2. Adjust learning rates first
3. Tune epistemic/pragmatic weights
4. Increase complexity last

### Development

1. Write tests first (TDD)
2. Use type hints
3. Document public APIs
4. Follow modular design

### Experimentation

1. Use reproducible seeds
2. Log all experiments
3. Visualize before analyzing
4. Compare baselines

## Code Style

- **Functions**: Verb names (compute, evaluate, update)
- **Classes**: Noun names (Agent, Environment, Config)
- **Variables**: Descriptive (belief_mean, not bm)
- **Comments**: Why, not what
- **Docstrings**: Google style

## Dependencies

**Core**:
- `torch`: Neural networks and autodiff
- `numpy`: Numerical operations

**Visualization**:
- `matplotlib`: Plotting and animation
- `pillow`: Animation export

**Testing**:
- `pytest`: Test framework
- `pytest-cov`: Coverage analysis

**Utilities**:
- `tqdm`: Progress bars

## Future Enhancements

### Potential Extensions

1. **Hierarchical Active Inference**: Multi-level beliefs
2. **Discrete Actions**: Categorical policy distribution
3. **Multi-Agent**: Multiple agents with shared environment
4. **Message Passing**: Structured inference
5. **Meta-Learning**: Learning priors
6. **GPU Acceleration**: Batch environments

### Research Directions

1. Compare with RL baselines
2. Real robot experiments
3. Cognitive modeling tasks
4. Neuroscience applications

## References

### Theoretical Foundation

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Friston, K., et al. (2017). Active inference: a process theory
- Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference

### Implementation Inspiration

- PyTorch documentation
- Software engineering best practices
- Test-driven development methodology

## Contributing

See main README.md for contribution guidelines.

Key principles:
- Maintain modularity
- Follow TDD
- Document changes
- Preserve clarity

