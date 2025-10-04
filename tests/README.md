# Active Torchference Test Suite

Comprehensive test coverage for all framework components following TDD principles.

## Overview

Tests ensure correctness, reliability, and maintainability of the Active Inference implementation. All tests use `pytest` with modular organization.

## Test Structure

```
tests/
├── __init__.py                   # Test package init
├── test_agent.py                 # Agent behavior and integration
├── test_beliefs.py               # Belief state management
├── test_belief_updates.py        # VFE minimization correctness
├── test_config.py                # Configuration validation
├── test_environment.py           # Environment implementations
├── test_free_energy.py           # VFE and EFE computations
├── test_efe_per_policy.py        # EFE per-policy validation
├── test_integration.py           # Full system integration
├── test_oscillator_environment.py # Oscillator dynamics
├── test_output_manager.py        # Output organization
├── test_policy.py                # Policy evaluation and selection
└── test_utils.py                 # Utility functions
```

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Test File
```bash
pytest tests/test_agent.py
pytest tests/test_belief_updates.py
pytest tests/test_efe_per_policy.py
```

### Specific Test Function
```bash
pytest tests/test_agent.py::test_agent_step
pytest tests/test_belief_updates.py::test_vfe_decreases_with_iterations
```

### With Coverage
```bash
pytest --cov=active_torchference tests/
```

### With Verbose Output
```bash
pytest tests/ -v
pytest tests/ -vv  # Extra verbose
```

### With Output
```bash
pytest tests/ -s  # Show print statements
```

## Test Categories

### 1. Unit Tests

Test individual components in isolation.

#### Configuration (`test_config.py`)
- Initialization and defaults
- Custom parameters
- Serialization/deserialization
- Seed reproducibility

#### Utilities (`test_utils.py`)
- Tensor operations
- Gaussian log-likelihood
- KL divergence
- Softmax and entropy

#### Belief State (`test_beliefs.py`)
- State initialization
- Sampling
- Updates
- Prior management
- Generative model forward pass

#### Free Energy (`test_free_energy.py`)
- VFE computation
- EFE computation
- Component decomposition
- Categorical variants

#### Policy (`test_policy.py`)
- Transition model
- Action sampling
- Rollouts
- Policy updates
- Action selection

#### Environment (`test_environment.py`)
- Continuous environment
- Grid world
- Oscillator environment
- Reset and step mechanics
- Goal specification

### 2. Integration Tests

#### Agent Behavior (`test_agent.py`)
- Initialization
- Perceive-plan-act loop
- History tracking
- State save/load
- Deterministic behavior

**Key Tests**:
- `test_agent_step`: Full action-perception loop
- `test_agent_step_updates_beliefs`: Belief updates work
- `test_agent_step_updates_policy`: Policy optimization works
- `test_agent_deterministic_behavior`: Reproducibility

#### Belief Updates (`test_belief_updates.py`)
- Beliefs respond to observations
- VFE decreases iteratively
- Convergence to observations
- Learning rate effects
- Component tracking

**Key Tests**:
- `test_belief_updates_with_observation`: Beliefs change with observations
- `test_vfe_decreases_with_iterations`: VFE minimization works
- `test_belief_convergence_to_observation`: Convergence behavior
- `test_repeated_observations_reduce_vfe`: Learning over time

#### EFE Per Policy (`test_efe_per_policy.py`)
- EFE computed per policy (not averaged)
- Per-timestep breakdown
- Policy differentiation
- Best policy selection
- Component tracking

**Key Tests**:
- `test_efe_returns_per_policy`: Returns [num_rollouts] tensor
- `test_efe_different_policies_different_values`: Policy diversity
- `test_best_policy_selection`: Correctly identifies best policy
- `test_efe_per_timestep_structure`: Correct [horizon, num_rollouts] shape

#### Full System (`test_integration.py`)
- Agent-environment interaction
- Learning over episodes
- Goal-directed behavior
- Multi-dimensional compatibility
- Free energy component validation

**Key Tests**:
- `test_agent_environment_integration`: Basic interaction
- `test_agent_learns_to_approach_goal`: Learning behavior
- `test_agent_vfe_decreases`: VFE reduction over time
- `test_multiple_episodes`: Multi-episode stability

### 3. Output Management (`test_output_manager.py`)
- Directory creation
- Path management
- Experiment listing
- Auto-naming
- Summary generation

## Test Conventions

### Naming
- Test files: `test_<module>.py`
- Test functions: `test_<functionality>()`
- Descriptive names explaining what's tested

### Structure
```python
def test_functionality():
    """Brief description of what this tests."""
    # Arrange: Setup test conditions
    config = Config(seed=42)
    agent = ActiveInferenceAgent(config)
    
    # Act: Execute functionality
    result = agent.step(observation, preferred_obs)
    
    # Assert: Verify expected behavior
    assert result is not None
    assert "vfe" in result[1]
```

### Fixtures
Common fixtures in `conftest.py` (if needed):
- Agent configurations
- Test environments
- Sample data

### Assertions
- Use descriptive assertions
- Test both success and edge cases
- Verify shapes, types, and values

## Critical Test Areas

### Belief Updates Must:
✓ Respond to observations  
✓ Reduce VFE with iterations  
✓ Update all latent dimensions  
✓ Maintain gradient flow  
✓ Track VFE history  

### Policy Evaluation Must:
✓ Compute EFE per policy (not averaged)  
✓ Compute EFE per timestep  
✓ Differentiate between policies  
✓ Identify best policy  
✓ Include epistemic + pragmatic components  

### Agent Integration Must:
✓ Execute full action-perception loop  
✓ Update beliefs and policy  
✓ Track history correctly  
✓ Save/load state  
✓ Behave deterministically with seed  

## Testing Guidelines

### When Adding New Features
1. Write tests first (TDD)
2. Test happy path and edge cases
3. Test error handling
4. Test integration with existing components
5. Ensure tests pass before committing

### Test Quality
- **Fast**: Tests should run quickly
- **Isolated**: No dependencies between tests
- **Repeatable**: Same results every run
- **Self-checking**: Clear pass/fail
- **Timely**: Written with or before code

### Coverage Goals
- Aim for >90% code coverage
- Focus on critical paths
- Test edge cases and error conditions
- Don't test external libraries

## Debugging Failed Tests

### View Full Output
```bash
pytest tests/test_agent.py -vv -s
```

### Run Single Test
```bash
pytest tests/test_agent.py::test_agent_step -vv
```

### Use Debugger
```python
# In test file
import pdb; pdb.set_trace()
```

Or with pytest:
```bash
pytest tests/test_agent.py --pdb
```

### Check Fixtures
```bash
pytest tests/test_agent.py --setup-show
```

## Continuous Integration

Tests run automatically on:
- Every commit
- Pull requests
- Pre-merge checks

Ensure all tests pass before merging.

## Test Validation

Beyond unit tests, see `validation/` directory for:
- `validate_vfe_computation.py`: Comprehensive VFE validation
- `validate_efe_computation.py`: Comprehensive EFE validation

These generate validation plots and detailed reports.

## Related Documentation

- **[ARCHITECTURE.md](../docs/ARCHITECTURE.md)**: System design
- **[AGENTS.md](../docs/AGENTS.md)**: Agent implementation details
- **[BELIEF_AND_POLICY_UPDATES.md](../docs/BELIEF_AND_POLICY_UPDATES.md)**: VFE/EFE technical details

## Contributing Tests

When contributing:
1. Follow existing test structure
2. Use descriptive names
3. Include docstrings
4. Test edge cases
5. Maintain >90% coverage
6. Update this README if adding new test categories

