"""
Demonstrates unified output directory structure.

All outputs (configs, logs, visualizations, animations, data) are organized
into subdirectories of a single top-level output folder.
"""

import torch
from active_torchference import Config, ActiveInferenceAgent, OutputManager
from active_torchference.environment import ContinuousEnvironment
from active_torchference.orchestrators import ExperimentRunner, Visualizer, Animator, Logger


def main():
    """Run experiment with unified output structure."""
    
    print("\n" + "="*70)
    print("Active Inference with Unified Output Directory Structure")
    print("="*70 + "\n")
    
    # Step 1: Create OutputManager (manages all output directories)
    output_mgr = OutputManager(
        output_root="./output",  # Single top-level directory
        experiment_name="unified_demo"
    )
    
    # Print directory structure
    output_mgr.print_structure()
    
    # Step 2: Configure experiment
    config = Config(
        hidden_dim=8,
        obs_dim=2,
        action_dim=2,
        learning_rate_beliefs=0.1,
        learning_rate_policy=0.05,
        epistemic_weight=1.0,
        pragmatic_weight=1.0,
        seed=42
    )
    
    # Step 3: Initialize agent and environment
    agent = ActiveInferenceAgent(config)
    goal_position = torch.tensor([2.0, 2.0])
    environment = ContinuousEnvironment(config, goal_position=goal_position)
    
    # Step 4: Create orchestrators with unified output manager
    logger = Logger(output_manager=output_mgr)
    visualizer = Visualizer(output_manager=output_mgr)
    animator = Animator(output_manager=output_mgr)
    
    print("✓ Output directories created:")
    print(f"  • Config:          {output_mgr.config_dir}")
    print(f"  • Logs:            {output_mgr.logs_dir}")
    print(f"  • Checkpoints:     {output_mgr.checkpoints_dir}")
    print(f"  • Visualizations:  {output_mgr.visualizations_dir}")
    print(f"  • Animations:      {output_mgr.animations_dir}")
    print(f"  • Data:            {output_mgr.data_dir}")
    print(f"  • Metadata:        {output_mgr.metadata_dir}\n")
    
    # Step 5: Run experiment
    runner = ExperimentRunner(agent, environment, logger, config)
    
    print("Running experiment...")
    results = runner.run(
        num_episodes=3,
        max_steps_per_episode=50,
        deterministic=False,
        verbose=True
    )
    
    # Step 6: Get agent history
    history = agent.get_history()
    
    # Step 7: Create visualizations (saved to visualizations/)
    print("\n✓ Generating visualizations...")
    visualizer.plot_free_energy(history)
    visualizer.plot_beliefs(history)
    visualizer.plot_trajectory_2d(history, goal=goal_position)
    visualizer.plot_actions(history)
    visualizer.plot_comprehensive_summary(history, goal=goal_position)
    print(f"  → Saved to: {output_mgr.visualizations_dir}")
    
    # Step 8: Create animations (saved to animations/)
    print("\n✓ Generating animations...")
    animator.animate_trajectory_2d(history, goal=goal_position, fps=10)
    animator.animate_beliefs(history, fps=10)
    animator.animate_comprehensive(history, goal=goal_position, fps=5)
    print(f"  → Saved to: {output_mgr.animations_dir}")
    
    # Step 9: Verify all outputs are in correct locations
    print("\n" + "="*70)
    print("Output Verification")
    print("="*70)
    
    # List contents of each directory
    for category, directory in [
        ("Config", output_mgr.config_dir),
        ("Logs", output_mgr.logs_dir),
        ("Checkpoints", output_mgr.checkpoints_dir),
        ("Visualizations", output_mgr.visualizations_dir),
        ("Animations", output_mgr.animations_dir),
        ("Data", output_mgr.data_dir)
    ]:
        files = list(directory.glob("*"))
        print(f"\n{category} ({len(files)} files):")
        for f in sorted(files)[:5]:  # Show first 5 files
            print(f"  • {f.name}")
        if len(files) > 5:
            print(f"  • ... and {len(files) - 5} more")
    
    print("\n" + "="*70)
    print("✓ All outputs organized in: ./output/unified_demo/")
    print("="*70 + "\n")
    
    # Show summary
    summary = output_mgr.get_experiment_summary()
    print("Experiment Summary:")
    for key, value in summary.items():
        if key != 'subdirectories':
            print(f"  {key}: {value}")
    
    # Demonstrate accessing specific output files
    print("\n" + "="*70)
    print("Accessing Specific Output Files")
    print("="*70)
    
    config_file = output_mgr.get_path('config', 'config.json')
    print(f"Config file:    {config_file}")
    
    metrics_file = output_mgr.get_path('logs', 'metrics.json')
    print(f"Metrics file:   {metrics_file}")
    
    checkpoint_file = output_mgr.get_path('checkpoints', 'final_checkpoint.pkl')
    print(f"Checkpoint:     {checkpoint_file}")
    
    viz_file = output_mgr.get_path('visualizations', 'summary.png')
    print(f"Visualization:  {viz_file}")
    
    anim_file = output_mgr.get_path('animations', 'trajectory_animation.gif')
    print(f"Animation:      {anim_file}")
    
    print("\n✓ Experiment complete!")
    print(f"✓ All outputs in: {output_mgr.experiment_dir}\n")


if __name__ == "__main__":
    main()

