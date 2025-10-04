"""Animation tools for Active Inference experiments."""

from typing import Optional, Dict, List
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


class Animator:
    """
    Creates animations of Active Inference agent behavior.
    
    Supports trajectory animations, belief evolution, and multi-panel views.
    Uses OutputManager for unified directory structure.
    """
    
    def __init__(self, output_manager=None, save_dir: Optional[str] = None):
        """
        Initialize animator.
        
        Args:
            output_manager: OutputManager instance for unified structure.
            save_dir: (Deprecated) Use output_manager instead.
        """
        # Support both new OutputManager and legacy save_dir
        if output_manager is not None:
            self.save_dir = output_manager.animations_dir
        elif save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
    
    def animate_trajectory_2d(
        self,
        history: Dict[str, List],
        goal: Optional[torch.Tensor] = None,
        fps: int = 10,
        filename: str = "trajectory_animation.gif"
    ):
        """
        Animate 2D trajectory with error handling.
        
        Args:
            history: Agent history dictionary.
            goal: Goal position (if any).
            fps: Frames per second.
            filename: Output filename.
        """
        try:
            if len(history["observations"]) == 0:
                print("⚠️  No observations in history - skipping trajectory animation")
                return None
            
            # Convert observations to numpy
            observations = torch.stack([
                o if isinstance(o, torch.Tensor) else torch.tensor(o)
                for o in history["observations"]
            ]).numpy()
            
            if observations.shape[1] < 2:
                print(f"⚠️  Need at least 2D observations for trajectory, got {observations.shape[1]}D")
                return None
        
            # Setup figure
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot goal
            if goal is not None:
                goal_np = goal.cpu().numpy() if isinstance(goal, torch.Tensor) else goal
                ax.scatter(
                    goal_np[0], goal_np[1],
                    c='red', s=300, marker='*',
                    label='Goal', zorder=5
                )
            
            # Initialize trajectory line and agent marker
            line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
            agent, = ax.plot([], [], 'go', markersize=15, label='Agent')
            
            # Set axis limits with padding
            x_min, x_max = observations[:, 0].min(), observations[:, 0].max()
            y_min, y_max = observations[:, 1].min(), observations[:, 1].max()
            padding = 0.1 * max(x_max - x_min, y_max - y_min)
            
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_title("Agent Trajectory Animation", fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            # Animation update function
            def update(frame):
                line.set_data(observations[:frame+1, 0], observations[:frame+1, 1])
                agent.set_data([observations[frame, 0]], [observations[frame, 1]])
                ax.set_title(f"Agent Trajectory Animation (Step {frame+1}/{len(observations)})",
                            fontweight='bold')
                return line, agent
            
            # Create animation
            anim = FuncAnimation(
                fig, update,
                frames=len(observations),
                interval=1000/fps,
                blit=True
            )
            
            # Save animation with error handling
            if self.save_dir:
                save_path = self.save_dir / filename
                try:
                    writer = PillowWriter(fps=fps)
                    anim.save(save_path, writer=writer)
                    print(f"✓ Trajectory animation saved to {save_path}")
                except Exception as e:
                    print(f"✗ Failed to save trajectory animation: {e}")
                    plt.close(fig)
                    return None
            
            plt.close(fig)
            return anim
            
        except Exception as e:
            print(f"✗ Trajectory animation generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def animate_beliefs(
        self,
        history: Dict[str, List],
        fps: int = 10,
        filename: str = "beliefs_animation.gif"
    ):
        """
        Animate belief state evolution with error handling.
        
        Args:
            history: Agent history dictionary.
            fps: Frames per second.
            filename: Output filename.
        """
        try:
            if len(history["beliefs"]) == 0:
                print("⚠️  No beliefs in history - skipping beliefs animation")
                return None
        
            # Convert beliefs to numpy
            beliefs = torch.stack([
                b if isinstance(b, torch.Tensor) else torch.tensor(b)
                for b in history["beliefs"]
            ]).numpy()
            
            hidden_dim = beliefs.shape[1]
            num_plots = min(hidden_dim, 4)
            
            # Setup figure
            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
            if num_plots == 1:
                axes = [axes]
            
            # Initialize lines
            lines = []
            for i in range(num_plots):
                line, = axes[i].plot([], [], linewidth=2)
                axes[i].set_xlim(0, len(beliefs))
                axes[i].set_ylim(beliefs[:, i].min() - 0.1, beliefs[:, i].max() + 0.1)
                axes[i].set_ylabel(f"Belief Dim {i}")
                axes[i].grid(True, alpha=0.3)
                lines.append(line)
            
            axes[-1].set_xlabel("Timestep")
            fig.suptitle("Belief States Evolution", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Animation update function
            def update(frame):
                for i, line in enumerate(lines):
                    line.set_data(range(frame+1), beliefs[:frame+1, i])
                fig.suptitle(f"Belief States Evolution (Step {frame+1}/{len(beliefs)})",
                            fontsize=14, fontweight='bold')
                return lines
            
            # Create animation
            anim = FuncAnimation(
                fig, update,
                frames=len(beliefs),
                interval=1000/fps,
                blit=True
            )
        
            # Save animation with error handling
            if self.save_dir:
                save_path = self.save_dir / filename
                try:
                    writer = PillowWriter(fps=fps)
                    anim.save(save_path, writer=writer)
                    print(f"✓ Beliefs animation saved to {save_path}")
                except Exception as e:
                    print(f"✗ Failed to save beliefs animation: {e}")
                    plt.close(fig)
                    return None
            
            plt.close(fig)
            return anim
            
        except Exception as e:
            print(f"✗ Beliefs animation generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def animate_comprehensive(
        self,
        history: Dict[str, List],
        goal: Optional[torch.Tensor] = None,
        fps: int = 10,
        filename: str = "comprehensive_animation.gif"
    ):
        """
        Create comprehensive multi-panel animation with error handling.
        
        Args:
            history: Agent history dictionary.
            goal: Goal position (if any).
            fps: Frames per second.
            filename: Output filename.
        """
        try:
            if len(history["observations"]) == 0:
                print("⚠️  No observations in history - skipping comprehensive animation")
                return None
        
            # Prepare data
            observations = torch.stack([
                o if isinstance(o, torch.Tensor) else torch.tensor(o)
                for o in history["observations"]
            ]).numpy()
            
            vfe_values = [v.item() if isinstance(v, torch.Tensor) else v 
                         for v in history["vfe"]]
            efe_values = [e.item() if isinstance(e, torch.Tensor) else e 
                         for e in history["efe"]]
            
            # Get EFE per policy for detailed plotting
            if "efe_per_policy" in history and len(history["efe_per_policy"]) > 0:
                efe_per_policy_data = torch.stack([
                    e if isinstance(e, torch.Tensor) else torch.tensor(e)
                    for e in history["efe_per_policy"]
                ]).numpy()  # Shape: [timesteps, num_rollouts]
                has_efe_per_policy = True
            else:
                has_efe_per_policy = False
            
            # Setup figure
            fig = plt.figure(figsize=(16, 10))
            
            # Trajectory subplot
            ax1 = plt.subplot(2, 2, 1)
            if goal is not None:
                goal_np = goal.cpu().numpy() if isinstance(goal, torch.Tensor) else goal
                ax1.scatter(goal_np[0], goal_np[1], c='red', s=300, marker='*', label='Goal')
            
            traj_line, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.6)
            agent_marker, = ax1.plot([], [], 'go', markersize=15)
            
            x_min, x_max = observations[:, 0].min(), observations[:, 0].max()
            y_min, y_max = observations[:, 1].min(), observations[:, 1].max()
            padding = 0.1 * max(x_max - x_min, y_max - y_min)
            ax1.set_xlim(x_min - padding, x_max + padding)
            ax1.set_ylim(y_min - padding, y_max + padding)
            ax1.set_title("Trajectory", fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # VFE subplot
            ax2 = plt.subplot(2, 2, 2)
            vfe_line, = ax2.plot([], [], 'b-', linewidth=2)
            ax2.set_xlim(0, len(vfe_values))
            ax2.set_ylim(min(vfe_values) - 0.1, max(vfe_values) + 0.1)
            ax2.set_title("Variational Free Energy", fontweight='bold')
            ax2.set_xlabel("Timestep")
            ax2.grid(True, alpha=0.3)
            
            # EFE subplot - show per-policy breakdown if available
            ax3 = plt.subplot(2, 2, 3)
            if has_efe_per_policy:
                # Plot each policy's EFE as a separate line
                efe_lines = []
                num_policies = min(efe_per_policy_data.shape[1], 10)  # Show up to 10 policies
                colors = plt.cm.viridis(np.linspace(0, 1, num_policies))
                
                for i in range(num_policies):
                    line, = ax3.plot([], [], linewidth=1.5, alpha=0.6, color=colors[i])
                    efe_lines.append(line)
                
                # Also plot mean EFE as bold line
                mean_efe_line, = ax3.plot([], [], 'r-', linewidth=3, alpha=0.9, label='Mean')
                
                ax3.set_xlim(0, len(efe_values))
                ax3.set_ylim(efe_per_policy_data.min() - 0.1, efe_per_policy_data.max() + 0.1)
                ax3.set_title("EFE Per Policy", fontweight='bold')
                ax3.legend(fontsize=8)
            else:
                # Fallback to mean EFE only
                efe_line, = ax3.plot([], [], 'r-', linewidth=2)
                ax3.set_xlim(0, len(efe_values))
                ax3.set_ylim(min(efe_values) - 0.1, max(efe_values) + 0.1)
                ax3.set_title("Expected Free Energy (Mean)", fontweight='bold')
            
            ax3.set_xlabel("Timestep")
            ax3.grid(True, alpha=0.3)
            
            # Beliefs subplot
            ax4 = plt.subplot(2, 2, 4)
            if len(history["beliefs"]) > 0:
                beliefs = torch.stack([
                    b if isinstance(b, torch.Tensor) else torch.tensor(b)
                    for b in history["beliefs"]
                ]).numpy()
                
                belief_lines = []
                for i in range(min(beliefs.shape[1], 3)):
                    line, = ax4.plot([], [], label=f'Dim {i}', linewidth=2)
                    belief_lines.append(line)
                
                ax4.set_xlim(0, len(beliefs))
                ax4.set_ylim(beliefs[:, :3].min() - 0.1, beliefs[:, :3].max() + 0.1)
            ax4.set_title("Belief States", fontweight='bold')
            ax4.set_xlabel("Timestep")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Animation update function
            def update(frame):
                # Update trajectory
                traj_line.set_data(observations[:frame+1, 0], observations[:frame+1, 1])
                agent_marker.set_data([observations[frame, 0]], [observations[frame, 1]])
                
                # Update VFE
                vfe_line.set_data(range(frame+1), vfe_values[:frame+1])
                
                # Update EFE - show per-policy breakdown if available
                if has_efe_per_policy:
                    for i, line in enumerate(efe_lines):
                        if i < efe_per_policy_data.shape[1]:
                            line.set_data(range(frame+1), efe_per_policy_data[:frame+1, i])
                    mean_efe_line.set_data(range(frame+1), efe_values[:frame+1])
                    artists_to_return = [traj_line, agent_marker, vfe_line, mean_efe_line] + efe_lines
                else:
                    efe_line.set_data(range(frame+1), efe_values[:frame+1])
                    artists_to_return = [traj_line, agent_marker, vfe_line, efe_line]
                
                # Update beliefs
                if len(history["beliefs"]) > 0:
                    for i, line in enumerate(belief_lines):
                        line.set_data(range(frame+1), beliefs[:frame+1, i])
                
                fig.suptitle(f"Active Inference Agent (Step {frame+1}/{len(observations)})",
                            fontsize=16, fontweight='bold')
                
                # Return all artists
                if len(history["beliefs"]) > 0:
                    artists_to_return.extend(belief_lines)
                return artists_to_return
            
            # Create animation
            anim = FuncAnimation(
                fig, update,
                frames=len(observations),
                interval=1000/fps,
                blit=True
            )
            
            # Save animation with error handling
            if self.save_dir:
                save_path = self.save_dir / filename
                try:
                    writer = PillowWriter(fps=fps)
                    anim.save(save_path, writer=writer)
                    print(f"✓ Comprehensive animation saved to {save_path}")
                except Exception as e:
                    print(f"✗ Failed to save comprehensive animation: {e}")
                    plt.close(fig)
                    return None
            
            plt.close(fig)
            return anim
            
        except Exception as e:
            print(f"✗ Comprehensive animation generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

