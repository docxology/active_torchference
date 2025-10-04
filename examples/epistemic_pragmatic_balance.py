"""
Demonstrates epistemic vs pragmatic balance in Active Inference.

Compares agent behavior with different weightings of exploration (epistemic)
and exploitation (pragmatic) values.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from active_torchference import (
    Config,
    ActiveInferenceAgent,
    ContinuousEnvironment,
    OutputManager
)
from active_torchference.orchestrators import ExperimentRunner, Logger


def run_experiment_with_weights(
    epistemic_weight: float,
    pragmatic_weight: float,
    base_output_dir: Path
):
    """
    Run experiment with specific epistemic/pragmatic weights.
    
    Args:
        epistemic_weight: Weight for exploration.
        pragmatic_weight: Weight for exploitation.
        base_output_dir: Base directory for outputs.
    
    Returns:
        Experiment results dictionary.
    """
    # Create output manager for this configuration
    output_mgr = OutputManager(
        output_root=str(base_output_dir),
        experiment_name=f"balance_e{epistemic_weight}_p{pragmatic_weight}"
    )
    
    config = Config(
        hidden_dim=8,
        obs_dim=2,
        action_dim=2,
        learning_rate_beliefs=0.1,
        learning_rate_policy=0.05,
        num_policy_iterations=10,
        horizon=3,
        epistemic_weight=epistemic_weight,
        pragmatic_weight=pragmatic_weight,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    goal_position = torch.tensor([2.0, 2.0])
    environment = ContinuousEnvironment(config, goal_position=goal_position)
    
    logger = Logger(output_manager=output_mgr)
    
    runner = ExperimentRunner(agent, environment, logger, config)
    
    results = runner.run(
        num_episodes=1,
        max_steps_per_episode=100,
        deterministic=False,
        verbose=False
    )
    
    return {
        "config": config,
        "results": results,
        "history": agent.get_history(),
        "logger": logger,
        "output_mgr": output_mgr
    }


def main():
    """Run comparison experiments with unified outputs."""
    
    print("Exploring Epistemic vs Pragmatic Balance in Active Inference\n")
    print("="*70)
    
    # Create base output directory
    base_output = Path("./output/epistemic_pragmatic_comparison")
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Define weight configurations to compare
    configurations = [
        {"name": "Pure Exploration", "epistemic": 2.0, "pragmatic": 0.1},
        {"name": "Balanced", "epistemic": 1.0, "pragmatic": 1.0},
        {"name": "Pure Exploitation", "epistemic": 0.1, "pragmatic": 2.0},
    ]
    
    experiments = {}
    
    # Run experiments
    for config in configurations:
        print(f"\nRunning: {config['name']}")
        print(f"  Epistemic Weight: {config['epistemic']}")
        print(f"  Pragmatic Weight: {config['pragmatic']}")
        
        results = run_experiment_with_weights(
            config["epistemic"],
            config["pragmatic"],
            base_output
        )
        
        experiments[config['name']] = results
        
        # Print summary
        logger = results["logger"]
        summary = logger.get_summary()
        print(f"  Final Distance: {summary.get('distance_final', 'N/A'):.4f}")
        print(f"  Mean VFE: {summary.get('vfe_mean', 'N/A'):.4f}")
        print(f"  Outputs: {results['output_mgr'].experiment_dir}")
    
    # Comparative visualization
    print("\n" + "="*70)
    print("Generating comparative visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot trajectories
    ax_traj = axes[0, 0]
    goal = torch.tensor([2.0, 2.0])
    ax_traj.scatter(goal[0], goal[1], c='red', s=300, marker='*', 
                   label='Goal', zorder=5)
    
    colors = ['blue', 'green', 'orange']
    for (name, exp), color in zip(experiments.items(), colors):
        history = exp["history"]
        observations = torch.stack(history["observations"]).numpy()
        
        ax_traj.plot(observations[:, 0], observations[:, 1], 
                    color=color, linewidth=2, alpha=0.7, label=name)
        ax_traj.scatter(observations[0, 0], observations[0, 1], 
                       color=color, s=100, marker='o')
    
    ax_traj.set_title("Trajectories Comparison", fontweight='bold')
    ax_traj.set_xlabel("X Position")
    ax_traj.set_ylabel("Y Position")
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    
    # Plot VFE
    ax_vfe = axes[0, 1]
    for (name, exp), color in zip(experiments.items(), colors):
        history = exp["history"]
        vfe_values = [v.item() if isinstance(v, torch.Tensor) else v 
                     for v in history["vfe"]]
        ax_vfe.plot(vfe_values, color=color, linewidth=2, label=name)
    
    ax_vfe.set_title("Variational Free Energy", fontweight='bold')
    ax_vfe.set_xlabel("Timestep")
    ax_vfe.set_ylabel("VFE")
    ax_vfe.legend()
    ax_vfe.grid(True, alpha=0.3)
    
    # Plot distance to goal
    ax_dist = axes[1, 0]
    for (name, exp), color in zip(experiments.items(), colors):
        logger = exp["logger"]
        if "distance_to_goal" in logger.metrics:
            distances = logger.metrics["distance_to_goal"]
            ax_dist.plot(distances, color=color, linewidth=2, label=name)
    
    ax_dist.set_title("Distance to Goal", fontweight='bold')
    ax_dist.set_xlabel("Timestep")
    ax_dist.set_ylabel("Distance")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    # Plot EFE
    ax_efe = axes[1, 1]
    for (name, exp), color in zip(experiments.items(), colors):
        history = exp["history"]
        efe_values = [e.item() if isinstance(e, torch.Tensor) else e 
                     for e in history["efe"]]
        ax_efe.plot(efe_values, color=color, linewidth=2, label=name)
    
    ax_efe.set_title("Expected Free Energy", fontweight='bold')
    ax_efe.set_xlabel("Timestep")
    ax_efe.set_ylabel("EFE")
    ax_efe.legend()
    ax_efe.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to base output directory
    comparison_path = base_output / "comparison_summary.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Comparative analysis saved to {comparison_path}")
    
    # Print insights
    print("\n" + "="*70)
    print("Key Insights:")
    print("-" * 70)
    print("• Pure Exploration: Agent explores more but may take longer to reach goal")
    print("• Balanced: Agent balances information gain with goal-directed behavior")
    print("• Pure Exploitation: Agent focuses on goal but may miss useful information")
    print("="*70)
    print(f"\n✓ All experiment outputs in: {base_output}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

