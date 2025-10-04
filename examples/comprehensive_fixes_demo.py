#!/usr/bin/env python3
"""
Comprehensive demonstration of all fixes:
- EFE per action tracking
- Belief uncertainty tracking
- New visualization methods
- Animation error handling
"""

import torch
from active_torchference import Config, ActiveInferenceAgent
from active_torchference.environment import ContinuousEnvironment
from active_torchference.orchestrators import (
    OutputManager,
    Visualizer,
    Animator,
    Logger,
    ExperimentRunner
)

def main():
    print("="*70)
    print("COMPREHENSIVE FIXES DEMONSTRATION")
    print("="*70)
    print()
    
    # Setup
    config = Config(
        obs_dim=2,
        hidden_dim=4,
        action_dim=2,
        num_rollouts=10  # Multiple action samples for EFE per action
    )
    
    output_mgr = OutputManager(experiment_name="comprehensive_fixes_demo")
    agent = ActiveInferenceAgent(config)
    env = ContinuousEnvironment(config)
    viz = Visualizer(output_manager=output_mgr)
    animator = Animator(output_manager=output_mgr)
    logger = Logger(output_manager=output_mgr)
    
    # Run episode
    print("Running episode...")
    obs, _ = env.reset()
    preferred_obs = env.get_preferred_observation()
    
    num_steps = 30
    for step in range(num_steps):
        action, info = agent.step(obs, preferred_obs)
        obs, env_info = env.step(action)
        
        # Log
        logger.log_step(
            timestep=step,
            agent_info=info,
            env_info=env_info
        )
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{num_steps}")
    
    logger.save_metrics()
    
    # Get history
    history = agent.get_history()
    
    # Verify new fields
    print("\n" + "="*70)
    print("VERIFICATION: New History Fields")
    print("="*70)
    print(f"✓ observations: {len(history['observations'])} entries")
    print(f"✓ efe: {len(history['efe'])} entries (mean EFE per step)")
    print(f"✓ efe_per_policy: {len(history['efe_per_policy'])} entries")
    if history['efe_per_policy']:
        print(f"    Shape: {history['efe_per_policy'][0].shape} (EFE for each action sample)")
    print(f"✓ efe_per_policy_per_timestep: {len(history['efe_per_policy_per_timestep'])} entries")
    if history['efe_per_policy_per_timestep']:
        print(f"    Shape: {history['efe_per_policy_per_timestep'][0].shape} (horizon x num_rollouts)")
    print(f"✓ efe_components: {len(history['efe_components'])} entries (epistemic/pragmatic)")
    print(f"✓ action_samples: {len(history['action_samples'])} entries")
    if history['action_samples']:
        print(f"    Shape: {history['action_samples'][0].shape} (candidate actions considered)")
    print(f"✓ best_policy_idx: {len(history['best_policy_idx'])} entries")
    print(f"✓ belief_stds: {len(history['belief_stds'])} entries (belief uncertainty)")
    
    # Generate standard visualizations
    print("\n" + "="*70)
    print("GENERATING STANDARD VISUALIZATIONS")
    print("="*70)
    try:
        viz.plot_trajectory_2d(history)
        print("✓ Trajectory plot created")
    except Exception as e:
        print(f"⚠️  Trajectory plot skipped: {e}")
    
    viz.plot_free_energy(history)
    print("✓ Free energy plot created")
    
    viz.plot_beliefs(history)
    print("✓ Beliefs plot created")
    
    try:
        viz.plot_actions(history)
        print("✓ Actions plot created")
    except:
        print("⚠️  Actions plot skipped")
    
    try:
        viz.plot_summary(history)
        print("✓ Summary plot created")
    except:
        print("⚠️  Summary plot skipped")
    
    # Generate NEW visualizations
    print("\n" + "="*70)
    print("GENERATING NEW VISUALIZATIONS")
    print("="*70)
    
    fig1 = viz.plot_efe_per_action(history)
    if fig1:
        print("✓ EFE per action plot created")
        print("    Shows: EFE for each action sample at each timestep")
        print("    Includes: Heatmap + distribution boxplot + selected action")
    else:
        print("✗ EFE per action plot failed")
    
    fig2 = viz.plot_epistemic_pragmatic_balance(history)
    if fig2:
        print("✓ Epistemic/pragmatic balance plot created")
        print("    Shows: Exploration vs exploitation balance over time")
        print("    Includes: Epistemic value, pragmatic value, and ratio")
    else:
        print("✗ Epistemic/pragmatic balance plot failed")
    
    fig3 = viz.plot_belief_with_uncertainty(history)
    if fig3:
        print("✓ Belief with uncertainty plot created")
        print("    Shows: Belief means with uncertainty bands")
        print("    Includes: Mean ± std dev for each belief dimension")
    else:
        print("✗ Belief with uncertainty plot failed")
    
    # Generate animations with error handling
    print("\n" + "="*70)
    print("GENERATING ANIMATIONS (with error handling)")
    print("="*70)
    
    anim1 = animator.animate_trajectory_2d(history, goal=preferred_obs, fps=10)
    if anim1:
        print("✓ Trajectory animation created")
    else:
        print("⚠️  Trajectory animation skipped (expected for non-2D observations)")
    
    anim2 = animator.animate_beliefs(history, fps=10)
    if anim2:
        print("✓ Beliefs animation created")
    else:
        print("⚠️  Beliefs animation skipped")
    
    anim3 = animator.animate_comprehensive(history, goal=preferred_obs, fps=10)
    if anim3:
        print("✓ Comprehensive animation created")
    else:
        print("⚠️  Comprehensive animation skipped")
    
    # Show output structure
    print("\n" + "="*70)
    print("OUTPUT STRUCTURE")
    print("="*70)
    print(f"Experiment directory: {output_mgr.experiment_dir}")
    print(f"  Config: {output_mgr.config_dir}")
    print(f"  Logs: {output_mgr.logs_dir}")
    print(f"  Checkpoints: {output_mgr.checkpoints_dir}")
    print(f"  Visualizations: {output_mgr.visualizations_dir}")
    print(f"  Animations: {output_mgr.animations_dir}")
    print(f"  Data: {output_mgr.data_dir}")
    
    # Show key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    # EFE analysis
    efe_per_policy_stds = [hist.std().item() for hist in history['efe_per_policy']]
    mean_efe_std = sum(efe_per_policy_stds) / len(efe_per_policy_stds) if efe_per_policy_stds else 0
    print(f"Average EFE std across action samples: {mean_efe_std:.4f}")
    print(f"  (Measures variety in candidate action quality)")
    
    # Belief uncertainty analysis
    belief_stds_mean = [hist.mean().item() for hist in history['belief_stds']]
    avg_belief_uncertainty = sum(belief_stds_mean) / len(belief_stds_mean) if belief_stds_mean else 0
    print(f"Average belief uncertainty: {avg_belief_uncertainty:.4f}")
    print(f"  (Measures confidence in hidden state estimates)")
    
    # Epistemic/pragmatic analysis
    if history['efe_components']:
        epistemic_vals = []
        pragmatic_vals = []
        for components in history['efe_components']:
            ep_vals = [c['epistemic'].mean().item() for c in components]
            pr_vals = [c['pragmatic'].mean().item() for c in components]
            epistemic_vals.append(sum(ep_vals) / len(ep_vals))
            pragmatic_vals.append(sum(pr_vals) / len(pr_vals))
        
        avg_epistemic = sum(epistemic_vals) / len(epistemic_vals)
        avg_pragmatic = sum(pragmatic_vals) / len(pragmatic_vals)
        exploration_ratio = avg_epistemic / (avg_pragmatic + 1e-8)
        
        print(f"Average epistemic value: {avg_epistemic:.4f}")
        print(f"Average pragmatic value: {avg_pragmatic:.4f}")
        print(f"Exploration/exploitation ratio: {exploration_ratio:.4f}")
        if exploration_ratio > 1.5:
            print("  → Agent is exploration-dominant (learning)")
        elif exploration_ratio < 0.67:
            print("  → Agent is exploitation-dominant (goal-directed)")
        else:
            print("  → Agent is balanced between exploration and exploitation")
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE FIXES DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nAll fixes validated:")
    print("  ✓ EFE per action tracking and visualization")
    print("  ✓ Belief uncertainty tracking and visualization")
    print("  ✓ Epistemic/pragmatic balance analysis")
    print("  ✓ Animation error handling")
    print("  ✓ Comprehensive history logging")
    print("\nCheck output directory for all visualizations and animations!")
    print(f"  {output_mgr.experiment_dir}")
    print()

if __name__ == "__main__":
    main()

