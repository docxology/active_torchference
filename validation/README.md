# Validation Scripts

Comprehensive validation of core Active Inference computations with detailed reporting and visualization.

## Overview

Validation scripts rigorously verify that VFE and EFE computations work correctly, going beyond unit tests to provide comprehensive analysis with plots and detailed reports.

## Scripts

### 1. VFE Validation

**File**: `validate_vfe_computation.py`

Validates Variational Free Energy computation and belief updates.

**Run**:
```bash
python3 validation/validate_vfe_computation.py
```

**Validates**:

1. **VFE Per Timestep**
   - ✓ VFE computed at every timestep
   - ✓ Scalar output (single value per timestep)
   - ✓ Values are reasonable

2. **VFE Per Latent Dimension**
   - ✓ All latent dimensions actively updated
   - ✓ Each dimension contributes to VFE
   - ✓ No dormant dimensions

3. **VFE Iteration Convergence**
   - ✓ VFE decreases with iterations
   - ✓ Convergence within iterations
   - ✓ Stable final values

4. **VFE Components**
   - ✓ Likelihood term present
   - ✓ Complexity term present
   - ✓ Components sum to total VFE

5. **VFE Response to Observations**
   - ✓ Different observations → different VFE
   - ✓ Beliefs adapt to observations
   - ✓ VFE reflects prediction error

**Output**:
- `validation/outputs/vfe_per_timestep.png`
- `validation/outputs/vfe_per_dimension.png`
- `validation/outputs/vfe_convergence.png`
- `validation/outputs/vfe_components.png`
- `validation/outputs/vfe_response.png`

**Report**: Console output with ✓/✗ for each validation check.

---

### 2. EFE Validation

**File**: `validate_efe_computation.py`

Validates Expected Free Energy computation and policy evaluation.

**Run**:
```bash
python3 validation/validate_efe_computation.py
```

**Validates**:

1. **EFE Per Policy**
   - ✓ EFE computed for each policy individually
   - ✓ Shape: [num_rollouts]
   - ✓ NOT averaged over policies
   - ✓ Different policies have different EFE values

2. **EFE Per Timestep**
   - ✓ EFE computed at each planning step
   - ✓ Shape: [horizon, num_rollouts]
   - ✓ Per-policy per-timestep breakdown
   - ✓ Sums correctly across horizon

3. **EFE Components**
   - ✓ Epistemic value present
   - ✓ Pragmatic value present
   - ✓ Both components contribute
   - ✓ Correct signs

4. **Policy Differentiation**
   - ✓ Different policies yield different EFE
   - ✓ Variance in EFE values
   - ✓ Policy diversity maintained

5. **Best Policy Selection**
   - ✓ Minimum EFE policy identified
   - ✓ Best policy consistent
   - ✓ Correct indexing

6. **EFE Weight Effects**
   - ✓ Epistemic weight affects exploration
   - ✓ Pragmatic weight affects exploitation
   - ✓ Weights have measurable impact

**Output**:
- `validation/outputs/efe_per_policy.png`
- `validation/outputs/efe_per_timestep_heatmap.png`
- `validation/outputs/efe_components.png`
- `validation/outputs/efe_policy_differentiation.png`
- `validation/outputs/efe_weights_comparison.png`

**Report**: Console output with ✓/✗ for each validation check.

---

## Output Structure

Validation scripts generate outputs in:

```
validation/
├── validate_vfe_computation.py
├── validate_efe_computation.py
├── outputs/                    # Generated plots and reports
│   ├── vfe_per_timestep.png
│   ├── vfe_per_dimension.png
│   ├── vfe_convergence.png
│   ├── vfe_components.png
│   ├── vfe_response.png
│   ├── efe_per_policy.png
│   ├── efe_per_timestep_heatmap.png
│   ├── efe_components.png
│   ├── efe_policy_differentiation.png
│   └── efe_weights_comparison.png
└── README.md
```

## Interpretation

### VFE Validation

**Expected Behavior**:
- VFE should decrease with belief update iterations
- All latent dimensions should show activity
- Components (likelihood + complexity) should sum to total VFE
- Different observations should produce different VFE values

**Red Flags**:
- VFE increases with iterations → check belief update logic
- Dormant dimensions → check generative model
- Components don't sum → check VFE calculation

### EFE Validation

**Expected Behavior**:
- Each policy should have its own EFE value
- EFE should vary across policies (diversity)
- Best policy should have lowest EFE
- Epistemic/pragmatic weights should affect EFE

**Red Flags**:
- All policies have same EFE → check rollout diversity
- EFE not per-policy → check evaluation structure
- Weights have no effect → check EFE components

## Validation vs Testing

### Unit Tests (`tests/`)
- Fast, focused checks
- Individual component behavior
- Automated CI/CD
- Pass/fail assertions

### Validation Scripts (`validation/`)
- Comprehensive analysis
- Detailed visualizations
- Human-interpretable reports
- Exploratory verification

Both are essential for confidence in implementation.

## Running Validation

### Complete Validation
```bash
# Run both validations
python3 validation/validate_vfe_computation.py
python3 validation/validate_efe_computation.py

# Review plots
open validation/outputs/
```

### Automated Validation
```bash
# Run via test suite
pytest tests/test_belief_updates.py -v
pytest tests/test_efe_per_policy.py -v
```

## Customizing Validation

### Modify Parameters

Edit configuration in validation scripts:

```python
# In validate_vfe_computation.py
config = Config(
    hidden_dim=8,           # Adjust dimensions
    num_belief_iterations=10,  # More iterations
    learning_rate_beliefs=0.1  # Different learning rate
)
```

### Add Custom Checks

```python
def validate_custom_property():
    """Validate custom property of VFE/EFE."""
    # Setup
    config = Config()
    agent = ActiveInferenceAgent(config)
    
    # Test
    result = agent.perceive(observation)
    
    # Validate
    assert custom_condition(result), "Custom check failed"
    print("✓ Custom validation passed")
```

## Troubleshooting

### Validation Fails

1. **Check unit tests first**: Ensure basic tests pass
2. **Review plots**: Visual inspection often reveals issues
3. **Compare with expected**: Check against theoretical predictions
4. **Adjust parameters**: Some checks may need tuning
5. **Verify installation**: Ensure package installed correctly

### Plots Not Generated

```bash
# Check matplotlib installation
pip install matplotlib pillow

# Check output directory
mkdir -p validation/outputs

# Run with verbose output
python3 validation/validate_vfe_computation.py
```

### Numerical Issues

- Reduce learning rates
- Increase precision parameters
- Check for NaN/Inf values
- Verify tensor dtypes

## Integration with Development

### Workflow

1. **Develop**: Write new feature
2. **Test**: Run unit tests
3. **Validate**: Run validation scripts
4. **Review**: Examine plots and reports
5. **Iterate**: Refine based on validation

### Pre-Commit Checks

```bash
# Before committing changes
pytest tests/
python3 validation/validate_vfe_computation.py
python3 validation/validate_efe_computation.py
```

## Related Documentation

- **[BELIEF_AND_POLICY_UPDATES.md](../docs/BELIEF_AND_POLICY_UPDATES.md)**: Technical details on VFE/EFE
- **[AGENTS.md](../docs/AGENTS.md)**: Agent implementation
- **[ARCHITECTURE.md](../docs/ARCHITECTURE.md)**: System design
- **[tests/README.md](../tests/README.md)**: Test suite documentation

## Contributing Validation

When adding validation scripts:
1. Follow existing structure
2. Generate informative plots
3. Provide clear pass/fail reporting
4. Document what's validated
5. Update this README
6. Include example outputs

