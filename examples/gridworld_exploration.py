"""
Grid world exploration example using Active Inference.

Agent explores discrete grid world with epistemic and pragmatic drives.
"""

import torch
from active_torchference import (
    Config,
    ActiveInferenceAgent,
    GridWorld,
    OutputManager
)
from active_torchference.orchestrators import ExperimentRunner, Visualizer, Logger


def main():
    """Run grid world exploration experiment with unified outputs."""
    
    # Step 1: Create unified output structure
    output_mgr = OutputManager(
        output_root="./output",
        experiment_name="gridworld_exploration"
    )
    
    # Step 2: Configuration emphasizing exploration (epistemic value)
    config = Config(
        hidden_dim=8,
        obs_dim=2,
        action_dim=2,
        learning_rate_beliefs=0.15,
        learning_rate_policy=0.1,
        num_policy_iterations=15,
        horizon=4,
        epistemic_weight=1.5,  # High epistemic weight for exploration
        pragmatic_weight=1.0,
        seed=42
    )
    
    # Step 3: Initialize agent and environment
    agent = ActiveInferenceAgent(config)
    environment = GridWorld(config, grid_size=5, goal_position=(4, 4))
    
    # Step 4: Setup logging and visualization with unified output
    logger = Logger(output_manager=output_mgr)
    visualizer = Visualizer(output_manager=output_mgr)
    
    # Step 5: Create experiment runner
    runner = ExperimentRunner(agent, environment, logger, config)
    
    # Add custom callback to track exploration
    explored_positions = set()
    
    def track_exploration(step, agent_info, env_info):
        if "position" in env_info:
            explored_positions.add(env_info["position"])
            logger.log_custom("unique_positions", len(explored_positions))
    
    runner.add_step_callback(track_exploration)
    
    # Step 6: Run experiment
    print("Running Active Inference Grid World Experiment...")
    results = runner.run(
        num_episodes=3,
        max_steps_per_episode=50,
        deterministic=False,
        verbose=True
    )
    
    # Step 7: Get agent history
    history = agent.get_history()
    
    # Step 8: Visualize results (saved to output/gridworld_exploration/visualizations/)
    print("\nGenerating visualizations...")
    visualizer.plot_free_energy(history)
    visualizer.plot_beliefs(history)
    visualizer.plot_trajectory_2d(history, goal=environment.get_preferred_observation())
    visualizer.plot_comprehensive_summary(history, goal=environment.get_preferred_observation())
    
    # Print exploration statistics
    print(f"\nExploration Statistics:")
    print(f"  Unique positions visited: {len(explored_positions)}")
    print(f"  Total positions in grid: {environment.grid_size ** 2}")
    print(f"  Coverage: {len(explored_positions) / (environment.grid_size ** 2) * 100:.1f}%")
    
    print(f"\n✓ Experiment complete! All outputs in: {output_mgr.experiment_dir}")


if __name__ == "__main__":
    main()

