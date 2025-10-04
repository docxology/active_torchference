"""
Simple navigation example using Active Inference.

Agent learns to navigate to a goal in continuous 2D space.
"""

import torch
from active_torchference import (
    Config,
    ActiveInferenceAgent,
    ContinuousEnvironment,
    OutputManager
)
from active_torchference.orchestrators import ExperimentRunner, Visualizer, Animator, Logger


def main():
    """Run simple navigation experiment with unified outputs."""
    
    # Step 1: Create unified output structure
    output_mgr = OutputManager(
        output_root="./output",
        experiment_name="simple_navigation"
    )
    
    # Step 2: Configuration
    config = Config(
        hidden_dim=8,
        obs_dim=2,
        action_dim=2,
        learning_rate_beliefs=0.1,
        learning_rate_policy=0.05,
        num_policy_iterations=10,
        horizon=3,
        epistemic_weight=0.5,
        pragmatic_weight=1.0,
        seed=42
    )
    
    # Step 3: Initialize agent and environment
    agent = ActiveInferenceAgent(config)
    goal_position = torch.tensor([2.0, 2.0])
    environment = ContinuousEnvironment(config, goal_position=goal_position)
    
    # Step 4: Setup logging and visualization with unified output
    logger = Logger(output_manager=output_mgr)
    visualizer = Visualizer(output_manager=output_mgr)
    animator = Animator(output_manager=output_mgr)
    
    # Step 5: Create experiment runner
    runner = ExperimentRunner(agent, environment, logger, config)
    
    # Step 6: Run experiment
    print("Running Active Inference Navigation Experiment...")
    results = runner.run(
        num_episodes=5,
        max_steps_per_episode=100,
        deterministic=False,
        verbose=True
    )
    
    # Step 7: Get agent history
    history = agent.get_history()
    
    # Step 8: Visualize results (saved to output/simple_navigation/visualizations/)
    print("\nGenerating visualizations...")
    visualizer.plot_free_energy(history)
    visualizer.plot_beliefs(history)
    visualizer.plot_trajectory_2d(history, goal=goal_position)
    visualizer.plot_actions(history)
    visualizer.plot_comprehensive_summary(history, goal=goal_position)
    
    # Step 9: Create animations (saved to output/simple_navigation/animations/)
    print("\nGenerating animations...")
    animator.animate_trajectory_2d(history, goal=goal_position)
    animator.animate_beliefs(history)
    animator.animate_comprehensive(history, goal=goal_position)
    
    print(f"\n✓ Experiment complete! All outputs in: {output_mgr.experiment_dir}")


if __name__ == "__main__":
    main()

