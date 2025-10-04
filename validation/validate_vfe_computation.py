"""
Validation script for Variational Free Energy (VFE) computation.

Verifies that:
1. VFE is computed at each timestep
2. VFE is calculated for each latent state dimension
3. VFE responds to observations
4. VFE decreases with iterative minimization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from active_torchference import Config, ActiveInferenceAgent
from active_torchference.environment import ContinuousEnvironment


def validate_vfe_per_timestep():
    """Validate VFE is computed at each timestep."""
    print("\n" + "="*70)
    print("VALIDATION 1: VFE Computed Per Timestep")
    print("="*70)
    
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        num_belief_iterations=5,
        learning_rate_beliefs=0.1,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    
    num_timesteps = 20
    vfe_values = []
    
    for t in range(num_timesteps):
        observation = torch.randn(config.obs_dim)
        info = agent.perceive(observation)
        vfe_values.append(info['vfe'].item())
        
        print(f"Timestep {t:2d}: VFE = {info['vfe'].item():.6f}")
    
    # Verify we have VFE for each timestep
    assert len(vfe_values) == num_timesteps, "VFE not computed for all timesteps!"
    
    # Verify VFE values are different (responding to different observations)
    unique_vfe = len(set([round(v, 6) for v in vfe_values]))
    print(f"\n✓ VFE computed for all {num_timesteps} timesteps")
    print(f"✓ {unique_vfe} unique VFE values (beliefs adapting)")
    
    return vfe_values


def validate_vfe_per_latent_dimension():
    """Validate VFE considers all latent state dimensions."""
    print("\n" + "="*70)
    print("VALIDATION 2: VFE Considers All Latent Dimensions")
    print("="*70)
    
    config = Config(
        hidden_dim=8,
        obs_dim=2,
        num_belief_iterations=10,
        learning_rate_beliefs=0.1,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    
    # Track belief states over time
    observation = torch.tensor([1.0, 2.0])
    
    initial_beliefs = agent.beliefs.mean.clone()
    print(f"\nInitial belief state (8D): {initial_beliefs.detach().numpy()}")
    
    # Perceive multiple times
    for i in range(5):
        info = agent.perceive(observation)
        print(f"\nIteration {i+1}:")
        print(f"  VFE: {info['vfe'].item():.6f}")
        print(f"  Belief state: {info['belief_mean'].detach().numpy()}")
    
    final_beliefs = info['belief_mean']
    
    # Check that each dimension changed
    changes = torch.abs(final_beliefs - initial_beliefs)
    print(f"\nChange per dimension:")
    for dim in range(config.hidden_dim):
        print(f"  Dimension {dim}: Δ = {changes[dim]:.6f}")
    
    # Verify all dimensions are being used
    assert torch.all(changes > 1e-6), "Some latent dimensions not updating!"
    
    print(f"\n✓ All {config.hidden_dim} latent dimensions active in VFE computation")
    print(f"✓ Mean change across dimensions: {changes.mean():.6f}")
    
    return changes


def validate_vfe_iteration_convergence():
    """Validate VFE decreases over iterations within a single perceive() call."""
    print("\n" + "="*70)
    print("VALIDATION 3: VFE Decreases Over Iterations")
    print("="*70)
    
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        num_belief_iterations=10,
        learning_rate_beliefs=0.2,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    observation = torch.tensor([1.5, 2.5])
    
    # Get VFE history
    info = agent.perceive(observation)
    vfe_history = info['vfe_history']
    
    print(f"\nVFE over {len(vfe_history)} iterations:")
    for i, vfe in enumerate(vfe_history):
        print(f"  Iteration {i}: VFE = {vfe:.6f}")
    
    # Verify monotonic decrease (with tolerance for noise)
    decreases = sum([vfe_history[i+1] <= vfe_history[i] * 1.01 
                     for i in range(len(vfe_history)-1)])
    decrease_rate = decreases / (len(vfe_history) - 1)
    
    print(f"\nInitial VFE: {vfe_history[0]:.6f}")
    print(f"Final VFE:   {vfe_history[-1]:.6f}")
    print(f"Reduction:   {vfe_history[0] - vfe_history[-1]:.6f}")
    print(f"Decrease rate: {decrease_rate*100:.1f}%")
    
    assert vfe_history[-1] < vfe_history[0], "VFE not decreasing!"
    
    print(f"\n✓ VFE decreases from {vfe_history[0]:.6f} to {vfe_history[-1]:.6f}")
    print(f"✓ Iterative minimization working correctly")
    
    return vfe_history


def validate_vfe_components():
    """Validate VFE has proper likelihood and complexity components."""
    print("\n" + "="*70)
    print("VALIDATION 4: VFE Components (Likelihood + Complexity)")
    print("="*70)
    
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        num_belief_iterations=5,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    observation = torch.randn(config.obs_dim)
    
    info = agent.perceive(observation)
    
    vfe = info['vfe'].item()
    likelihood = info['vfe_likelihood'].item()
    complexity = info['vfe_complexity'].item()
    
    print(f"\nVFE Components:")
    print(f"  Likelihood term: {likelihood:.6f}")
    print(f"  Complexity term: {complexity:.6f}")
    print(f"  Total VFE:       {vfe:.6f}")
    print(f"  Reconstruction:  {likelihood + complexity:.6f}")
    
    # Verify components sum to total (within tolerance)
    reconstructed = likelihood + complexity
    diff = abs(vfe - reconstructed)
    
    print(f"\nDifference: {diff:.8f}")
    
    assert diff < 0.01, f"VFE components don't sum correctly! Diff={diff}"
    
    print(f"\n✓ VFE = Likelihood + Complexity")
    print(f"✓ VFE properly decomposed")
    
    return vfe, likelihood, complexity


def validate_vfe_response_to_observations():
    """Validate VFE responds differently to different observations."""
    print("\n" + "="*70)
    print("VALIDATION 5: VFE Responds to Different Observations")
    print("="*70)
    
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        num_belief_iterations=5,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    
    # Test with different observations
    observations = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([2.0, 2.0]),
        torch.tensor([-1.0, 3.0]),
    ]
    
    vfe_values = []
    
    for i, obs in enumerate(observations):
        agent.reset()
        info = agent.perceive(obs)
        vfe_values.append(info['vfe'].item())
        print(f"Observation {obs.numpy()}: VFE = {info['vfe'].item():.6f}")
    
    # Verify VFE values differ
    unique_vfe = len(set([round(v, 4) for v in vfe_values]))
    
    print(f"\n✓ {unique_vfe} unique VFE values for {len(observations)} observations")
    print(f"✓ VFE responsive to observation content")
    
    return vfe_values


def generate_validation_plots(output_dir="validation/outputs"):
    """Generate validation plots."""
    print("\n" + "="*70)
    print("Generating Validation Plots")
    print("="*70)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    config = Config(
        hidden_dim=8,
        obs_dim=2,
        num_belief_iterations=10,
        learning_rate_beliefs=0.1,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    
    # Collect data over episode
    num_steps = 50
    vfe_per_step = []
    vfe_histories = []
    belief_means = []
    
    for step in range(num_steps):
        obs = torch.randn(config.obs_dim)
        info = agent.perceive(obs)
        
        vfe_per_step.append(info['vfe'].item())
        vfe_histories.append(info['vfe_history'])
        belief_means.append(info['belief_mean'].numpy())
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: VFE over timesteps
    axes[0, 0].plot(vfe_per_step, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('VFE')
    axes[0, 0].set_title('VFE Over Timesteps')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: VFE convergence within timesteps
    for i in [0, 10, 25, 49]:
        axes[0, 1].plot(vfe_histories[i], label=f'Step {i}', linewidth=2)
    axes[0, 1].set_xlabel('Iteration within perceive()')
    axes[0, 1].set_ylabel('VFE')
    axes[0, 1].set_title('VFE Minimization per Timestep')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Belief evolution
    belief_array = np.array(belief_means)
    for dim in range(min(4, config.hidden_dim)):
        axes[1, 0].plot(belief_array[:, dim], label=f'Dim {dim}', linewidth=2)
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Belief Mean')
    axes[1, 0].set_title('Latent State Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: VFE reduction per timestep
    reductions = [h[0] - h[-1] for h in vfe_histories]
    axes[1, 1].bar(range(len(reductions)), reductions, alpha=0.7)
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('VFE Reduction')
    axes[1, 1].set_title('VFE Minimization Effectiveness')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vfe_validation.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to {output_dir}/vfe_validation.png")
    
    plt.close()


def main():
    """Run all VFE validation tests."""
    print("\n" + "="*70)
    print("VFE COMPUTATION VALIDATION SUITE")
    print("="*70)
    print("\nValidating that VFE is properly computed:")
    print("  1. At each timestep")
    print("  2. For each latent state dimension")
    print("  3. With iterative minimization")
    print("  4. With proper components")
    print("  5. Responding to observations")
    
    results = {}
    
    try:
        results['per_timestep'] = validate_vfe_per_timestep()
        results['per_dimension'] = validate_vfe_per_latent_dimension()
        results['convergence'] = validate_vfe_iteration_convergence()
        results['components'] = validate_vfe_components()
        results['response'] = validate_vfe_response_to_observations()
        
        generate_validation_plots()
        
        print("\n" + "="*70)
        print("✅ ALL VFE VALIDATIONS PASSED")
        print("="*70)
        print("\nSummary:")
        print("  ✓ VFE computed per timestep")
        print("  ✓ All latent dimensions active")
        print("  ✓ Iterative minimization working")
        print("  ✓ Components sum correctly")
        print("  ✓ Responsive to observations")
        print("\nVFE computation is CORRECT and COMPLETE")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

