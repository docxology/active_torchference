"""
Example of using custom environment (OscillatorEnvironment) for Active Inference.

Demonstrates oscillating goal tracking using environment from main package.
"""

import torch
from active_torchference import (
    Config,
    ActiveInferenceAgent,
    OscillatorEnvironment,
    OutputManager
)
from active_torchference.orchestrators import ExperimentRunner, Visualizer, Logger


def main():
    """Run oscillator environment experiment with unified outputs."""
    
    # Step 1: Create unified output structure
    output_mgr = OutputManager(
        output_root="./output",
        experiment_name="oscillator_tracking"
    )
    
    # Step 2: Configuration
    config = Config(
        hidden_dim=12,
        obs_dim=2,
        action_dim=2,
        learning_rate_beliefs=0.2,
        learning_rate_policy=0.1,
        num_policy_iterations=15,
        horizon=5,
        epistemic_weight=0.3,
        pragmatic_weight=1.0,
        seed=42
    )
    
    # Step 3: Initialize agent and custom environment
    agent = ActiveInferenceAgent(config)
    environment = OscillatorEnvironment(config, frequency=0.05, amplitude=2.0)
    
    # Step 4: Setup logging and visualization with unified output
    logger = Logger(output_manager=output_mgr)
    visualizer = Visualizer(output_manager=output_mgr)
    
    # Step 5: Create experiment runner
    runner = ExperimentRunner(agent, environment, logger, config)
    
    # Track tracking error over time
    def track_error(step, agent_info, env_info):
        if "distance_to_goal" in env_info:
            logger.log_custom("tracking_error", env_info["distance_to_goal"])
    
    runner.add_step_callback(track_error)
    
    # Step 6: Run experiment
    print("Running Active Inference Oscillator Tracking Experiment...")
    results = runner.run(
        num_episodes=3,
        max_steps_per_episode=200,
        deterministic=False,
        verbose=True
    )
    
    # Step 7: Get agent history
    history = agent.get_history()
    
    # Step 8: Visualize results (saved to output/oscillator_tracking/visualizations/)
    print("\nGenerating visualizations...")
    visualizer.plot_free_energy(history)
    visualizer.plot_beliefs(history)
    visualizer.plot_trajectory_2d(history, title="Oscillator Tracking Trajectory")
    visualizer.plot_comprehensive_summary(history)
    
    # Plot tracking error
    import matplotlib.pyplot as plt
    
    if "tracking_error" in logger.metrics["custom"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(logger.metrics["custom"]["tracking_error"], linewidth=2)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Tracking Error")
        ax.set_title("Distance to Moving Goal Over Time", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.savefig(output_mgr.visualizations_dir / "tracking_error.png", dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Experiment complete! All outputs in: {output_mgr.experiment_dir}")


if __name__ == "__main__":
    main()

