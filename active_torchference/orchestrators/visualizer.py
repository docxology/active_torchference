"""Visualization tools for Active Inference experiments."""

from typing import Optional, List, Dict, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path


class Visualizer:
    """
    Visualizes Active Inference agent behavior and metrics.
    
    Provides plotting for free energy, beliefs, trajectories, and more.
    Uses OutputManager for unified directory structure.
    """
    
    def __init__(self, output_manager=None, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            output_manager: OutputManager instance for unified structure.
            save_dir: (Deprecated) Use output_manager instead.
        """
        # Support both new OutputManager and legacy save_dir
        if output_manager is not None:
            self.save_dir = output_manager.visualizations_dir
        elif save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        
        # Style configuration
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    def plot_free_energy(
        self,
        history: Dict[str, List],
        title: str = "Free Energy Over Time"
    ) -> Figure:
        """
        Plot VFE and EFE over time.
        
        Args:
            history: Agent history dictionary.
            title: Plot title.
        
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # VFE plot
        if len(history["vfe"]) > 0:
            vfe_values = [v.item() if isinstance(v, torch.Tensor) else v 
                         for v in history["vfe"]]
            ax1.plot(vfe_values, label="VFE", color='blue', linewidth=2)
            ax1.set_ylabel("Variational Free Energy")
            ax1.set_xlabel("Timestep")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # EFE plot
        if len(history["efe"]) > 0:
            efe_values = [e.item() if isinstance(e, torch.Tensor) else e 
                         for e in history["efe"]]
            ax2.plot(efe_values, label="EFE", color='red', linewidth=2)
            ax2.set_ylabel("Expected Free Energy")
            ax2.set_xlabel("Timestep")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if self.save_dir:
            fig.savefig(self.save_dir / "free_energy.png", dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_beliefs(
        self,
        history: Dict[str, List],
        title: str = "Belief States Over Time"
    ) -> Figure:
        """
        Plot belief state evolution.
        
        Args:
            history: Agent history dictionary.
            title: Plot title.
        
        Returns:
            Matplotlib figure.
        """
        if len(history["beliefs"]) == 0:
            return None
        
        # Convert beliefs to numpy array
        beliefs = torch.stack([
            b if isinstance(b, torch.Tensor) else torch.tensor(b)
            for b in history["beliefs"]
        ]).numpy()
        
        hidden_dim = beliefs.shape[1]
        
        fig, axes = plt.subplots(
            min(hidden_dim, 4), 1,
            figsize=(10, 2 * min(hidden_dim, 4))
        )
        
        if hidden_dim == 1:
            axes = [axes]
        
        for i in range(min(hidden_dim, 4)):
            axes[i].plot(beliefs[:, i], linewidth=2)
            axes[i].set_ylabel(f"Belief Dim {i}")
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Timestep")
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if self.save_dir:
            fig.savefig(self.save_dir / "beliefs.png", dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_trajectory_2d(
        self,
        history: Dict[str, List],
        goal: Optional[torch.Tensor] = None,
        title: str = "Agent Trajectory"
    ) -> Figure:
        """
        Plot 2D trajectory of observations.
        
        Args:
            history: Agent history dictionary.
            goal: Goal position (if any).
            title: Plot title.
        
        Returns:
            Matplotlib figure.
        """
        if len(history["observations"]) == 0:
            return None
        
        # Convert observations to numpy array
        observations = torch.stack([
            o if isinstance(o, torch.Tensor) else torch.tensor(o)
            for o in history["observations"]
        ]).numpy()
        
        if observations.shape[1] < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot trajectory
        ax.plot(
            observations[:, 0],
            observations[:, 1],
            'b-',
            linewidth=2,
            alpha=0.6,
            label='Trajectory'
        )
        
        # Plot start and end
        ax.scatter(
            observations[0, 0],
            observations[0, 1],
            c='green',
            s=200,
            marker='o',
            label='Start',
            zorder=5
        )
        ax.scatter(
            observations[-1, 0],
            observations[-1, 1],
            c='blue',
            s=200,
            marker='s',
            label='End',
            zorder=5
        )
        
        # Plot goal if provided
        if goal is not None:
            goal_np = goal.cpu().numpy() if isinstance(goal, torch.Tensor) else goal
            ax.scatter(
                goal_np[0],
                goal_np[1],
                c='red',
                s=300,
                marker='*',
                label='Goal',
                zorder=5
            )
        
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        if self.save_dir:
            fig.savefig(self.save_dir / "trajectory.png", dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_actions(
        self,
        history: Dict[str, List],
        title: str = "Actions Over Time"
    ) -> Figure:
        """
        Plot action evolution.
        
        Args:
            history: Agent history dictionary.
            title: Plot title.
        
        Returns:
            Matplotlib figure.
        """
        if len(history["actions"]) == 0:
            return None
        
        # Convert actions to numpy array
        actions = torch.stack([
            a if isinstance(a, torch.Tensor) else torch.tensor(a)
            for a in history["actions"]
        ]).numpy()
        
        action_dim = actions.shape[1]
        
        fig, axes = plt.subplots(
            min(action_dim, 4), 1,
            figsize=(10, 2 * min(action_dim, 4))
        )
        
        if action_dim == 1:
            axes = [axes]
        
        for i in range(min(action_dim, 4)):
            axes[i].plot(actions[:, i], linewidth=2)
            axes[i].set_ylabel(f"Action Dim {i}")
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Timestep")
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if self.save_dir:
            fig.savefig(self.save_dir / "actions.png", dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_comprehensive_summary(
        self,
        history: Dict[str, List],
        goal: Optional[torch.Tensor] = None
    ) -> Figure:
        """
        Create comprehensive multi-panel summary figure.
        
        Args:
            history: Agent history dictionary.
            goal: Goal position (if any).
        
        Returns:
            Matplotlib figure.
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Free energies
        ax1 = plt.subplot(3, 2, 1)
        if len(history["vfe"]) > 0:
            vfe_values = [v.item() if isinstance(v, torch.Tensor) else v 
                         for v in history["vfe"]]
            ax1.plot(vfe_values, 'b-', linewidth=2)
        ax1.set_title("Variational Free Energy", fontweight='bold')
        ax1.set_xlabel("Timestep")
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 2, 2)
        if len(history["efe"]) > 0:
            efe_values = [e.item() if isinstance(e, torch.Tensor) else e 
                         for e in history["efe"]]
            ax2.plot(efe_values, 'r-', linewidth=2)
        ax2.set_title("Expected Free Energy", fontweight='bold')
        ax2.set_xlabel("Timestep")
        ax2.grid(True, alpha=0.3)
        
        # Trajectory
        ax3 = plt.subplot(3, 2, 3)
        if len(history["observations"]) > 0:
            observations = torch.stack([
                o if isinstance(o, torch.Tensor) else torch.tensor(o)
                for o in history["observations"]
            ]).numpy()
            
            if observations.shape[1] >= 2:
                ax3.plot(observations[:, 0], observations[:, 1], 'b-', linewidth=2, alpha=0.6)
                ax3.scatter(observations[0, 0], observations[0, 1], c='green', s=100, label='Start')
                ax3.scatter(observations[-1, 0], observations[-1, 1], c='blue', s=100, label='End')
                
                if goal is not None:
                    goal_np = goal.cpu().numpy() if isinstance(goal, torch.Tensor) else goal
                    ax3.scatter(goal_np[0], goal_np[1], c='red', s=150, marker='*', label='Goal')
                
                ax3.legend()
        ax3.set_title("2D Trajectory", fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Beliefs
        ax4 = plt.subplot(3, 2, 4)
        if len(history["beliefs"]) > 0:
            beliefs = torch.stack([
                b if isinstance(b, torch.Tensor) else torch.tensor(b)
                for b in history["beliefs"]
            ]).numpy()
            
            for i in range(min(beliefs.shape[1], 3)):
                ax4.plot(beliefs[:, i], label=f'Dim {i}', linewidth=2)
        ax4.set_title("Belief States", fontweight='bold')
        ax4.set_xlabel("Timestep")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Actions
        ax5 = plt.subplot(3, 2, 5)
        if len(history["actions"]) > 0:
            actions = torch.stack([
                a if isinstance(a, torch.Tensor) else torch.tensor(a)
                for a in history["actions"]
            ]).numpy()
            
            for i in range(min(actions.shape[1], 3)):
                ax5.plot(actions[:, i], label=f'Dim {i}', linewidth=2)
        ax5.set_title("Actions", fontweight='bold')
        ax5.set_xlabel("Timestep")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Statistics
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        stats_text = "Experiment Statistics\n" + "="*30 + "\n"
        stats_text += f"Total Steps: {len(history['observations'])}\n"
        
        if len(history["vfe"]) > 0:
            vfe_tensor = torch.tensor([v.item() if isinstance(v, torch.Tensor) else v 
                                      for v in history["vfe"]])
            stats_text += f"\nVFE Mean: {vfe_tensor.mean():.4f}\n"
            stats_text += f"VFE Std: {vfe_tensor.std():.4f}\n"
        
        if len(history["efe"]) > 0:
            efe_tensor = torch.tensor([e.item() if isinstance(e, torch.Tensor) else e 
                                      for e in history["efe"]])
            stats_text += f"\nEFE Mean: {efe_tensor.mean():.4f}\n"
            stats_text += f"EFE Std: {efe_tensor.std():.4f}\n"
        
        ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace')
        
        plt.tight_layout()
        
        if self.save_dir:
            fig.savefig(self.save_dir / "summary.png", dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_efe_per_action(
        self,
        history: Dict[str, List],
        title: str = "EFE Per Action Sample"
    ) -> Figure:
        """
        Plot EFE for each action sample at each timestep.
        
        Shows the distribution of EFE values across candidate actions,
        highlighting which action was selected.
        
        Args:
            history: Agent history dictionary.
            title: Plot title.
        
        Returns:
            Matplotlib figure.
        """
        if "efe_per_policy" not in history or len(history["efe_per_policy"]) == 0:
            print("⚠️  No efe_per_policy data available - EFE per action tracking not enabled")
            return None
        
        try:
            # Convert to numpy: [timesteps, num_rollouts]
            efe_per_policy = torch.stack(history["efe_per_policy"]).numpy()
            best_indices = history["best_policy_idx"]
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Heatmap of EFE across actions and timesteps
            im = axes[0].imshow(
                efe_per_policy.T,
                aspect='auto',
                cmap='viridis',
                interpolation='nearest',
                origin='lower'
            )
            axes[0].set_xlabel('Timestep', fontsize=12)
            axes[0].set_ylabel('Action Sample Index', fontsize=12)
            axes[0].set_title('EFE Heatmap (darker = lower EFE = better)', fontweight='bold')
            plt.colorbar(im, ax=axes[0], label='EFE Value')
            
            # Mark selected actions with red stars
            for t, idx in enumerate(best_indices):
                axes[0].plot(t, idx, 'r*', markersize=12, markeredgecolor='white', markeredgewidth=0.5)
            
            # Plot 2: EFE distribution at each timestep (boxplot)
            timesteps = range(len(efe_per_policy))
            boxplot_data = [efe_per_policy[t, :] for t in timesteps]
            
            # Only show every Nth boxplot if too many timesteps
            stride = max(1, len(timesteps) // 50)
            positions = list(range(0, len(timesteps), stride))
            data_to_plot = [boxplot_data[i] for i in positions]
            
            bp = axes[1].boxplot(
                data_to_plot,
                positions=positions,
                widths=stride*0.6,
                showfliers=False,
                patch_artist=True
            )
            
            # Color boxplots
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.6)
            
            axes[1].set_xlabel('Timestep', fontsize=12)
            axes[1].set_ylabel('EFE Value', fontsize=12)
            axes[1].set_title('EFE Distribution Across Action Samples', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Overlay selected action EFE
            selected_efe = [efe_per_policy[t, idx] for t, idx in enumerate(best_indices)]
            axes[1].plot(timesteps, selected_efe, 
                        'r-', linewidth=2.5, label='Selected Action', zorder=10)
            axes[1].legend(fontsize=11)
            
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            if self.save_dir:
                fig.savefig(self.save_dir / "efe_per_action.png", dpi=150, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"✗ Failed to plot EFE per action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_epistemic_pragmatic_balance(
        self,
        history: Dict[str, List],
        title: str = "Epistemic vs Pragmatic Value"
    ) -> Figure:
        """
        Plot epistemic and pragmatic components of EFE.
        
        Shows the balance between exploration (epistemic) and 
        exploitation (pragmatic) over time.
        
        Args:
            history: Agent history dictionary.
            title: Plot title.
        
        Returns:
            Matplotlib figure.
        """
        if "efe_components" not in history or len(history["efe_components"]) == 0:
            print("⚠️  No efe_components data available - component tracking not enabled")
            return None
        
        try:
            # Extract epistemic and pragmatic values
            epistemic_values = []
            pragmatic_values = []
            
            for timestep_components in history["efe_components"]:
                # Average across planning horizon and rollouts
                ep_vals = [c["epistemic"].mean().item() for c in timestep_components]
                pr_vals = [c["pragmatic"].mean().item() for c in timestep_components]
                epistemic_values.append(np.mean(ep_vals))
                pragmatic_values.append(np.mean(pr_vals))
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 11))
            
            timesteps = range(len(epistemic_values))
            
            # Plot 1: Epistemic value
            axes[0].plot(timesteps, epistemic_values, 'b-', linewidth=2.5, label='Epistemic Value')
            axes[0].fill_between(timesteps, epistemic_values, alpha=0.3)
            axes[0].set_ylabel('Epistemic Value\n(Information Gain)', fontsize=12)
            axes[0].set_title('Exploration Drive', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=11)
            
            # Plot 2: Pragmatic value
            axes[1].plot(timesteps, pragmatic_values, 'r-', linewidth=2.5, label='Pragmatic Value')
            axes[1].fill_between(timesteps, pragmatic_values, alpha=0.3, color='red')
            axes[1].set_ylabel('Pragmatic Value\n(Goal Achievement)', fontsize=12)
            axes[1].set_title('Exploitation Drive', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=11)
            
            # Plot 3: Ratio
            ratio = np.array(epistemic_values) / (np.array(pragmatic_values) + 1e-8)
            axes[2].plot(timesteps, ratio, 'g-', linewidth=2.5)
            axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=2, 
                           label='Balance Point')
            axes[2].fill_between(timesteps, ratio, 1.0, where=(ratio >= 1.0), 
                                alpha=0.3, color='blue', label='Exploration Dominant')
            axes[2].fill_between(timesteps, ratio, 1.0, where=(ratio < 1.0), 
                                alpha=0.3, color='red', label='Exploitation Dominant')
            axes[2].set_xlabel('Timestep', fontsize=12)
            axes[2].set_ylabel('Epistemic / Pragmatic\nRatio', fontsize=12)
            axes[2].set_title('Exploration vs Exploitation Balance', fontweight='bold')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(fontsize=10)
            
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            if self.save_dir:
                fig.savefig(self.save_dir / "epistemic_pragmatic_balance.png", 
                           dpi=150, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"✗ Failed to plot epistemic/pragmatic balance: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_belief_with_uncertainty(
        self,
        history: Dict[str, List],
        title: str = "Belief States with Uncertainty"
    ) -> Figure:
        """
        Plot belief means AND standard deviations over time.
        
        Shows belief evolution with uncertainty bands.
        
        Args:
            history: Agent history dictionary.
            title: Plot title.
        
        Returns:
            Matplotlib figure.
        """
        if len(history["beliefs"]) == 0:
            return None
        
        try:
            # Convert beliefs to numpy array
            beliefs = torch.stack([
                b if isinstance(b, torch.Tensor) else torch.tensor(b)
                for b in history["beliefs"]
            ]).numpy()
            
            # Get belief stds if available
            if "belief_stds" in history and len(history["belief_stds"]) > 0:
                belief_stds = torch.stack([
                    b if isinstance(b, torch.Tensor) else torch.tensor(b)
                    for b in history["belief_stds"]
                ]).numpy()
                has_uncertainty = True
            else:
                has_uncertainty = False
            
            hidden_dim = beliefs.shape[1]
            
            fig, axes = plt.subplots(
                min(hidden_dim, 4), 1,
                figsize=(12, 3 * min(hidden_dim, 4))
            )
            
            if hidden_dim == 1:
                axes = [axes]
            
            timesteps = range(len(beliefs))
            
            for i in range(min(hidden_dim, 4)):
                # Plot mean
                axes[i].plot(timesteps, beliefs[:, i], linewidth=2.5, 
                            label='Belief Mean', color='blue')
                
                # Plot uncertainty band if available
                if has_uncertainty:
                    axes[i].fill_between(
                        timesteps,
                        beliefs[:, i] - belief_stds[:, i],
                        beliefs[:, i] + belief_stds[:, i],
                        alpha=0.3,
                        color='blue',
                        label='±1 Std Dev'
                    )
                    
                    # Also plot std separately
                    ax_twin = axes[i].twinx()
                    ax_twin.plot(timesteps, belief_stds[:, i], 
                               'r--', linewidth=1.5, alpha=0.6, label='Std Dev')
                    ax_twin.set_ylabel('Std Dev', color='r', fontsize=10)
                    ax_twin.tick_params(axis='y', labelcolor='r')
                
                axes[i].set_ylabel(f"Belief Dim {i}", fontsize=11)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend(loc='upper left', fontsize=10)
            
            axes[-1].set_xlabel("Timestep", fontsize=12)
            fig.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if self.save_dir:
                fig.savefig(self.save_dir / "beliefs_with_uncertainty.png", 
                           dpi=150, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"✗ Failed to plot beliefs with uncertainty: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def show(self):
        """Display all plots (deprecated - plots are auto-saved)."""
        print("⚠️  Note: Plots are automatically saved. Use generate_html_report() for interactive viewing.")
    
    def generate_html_report(self, history: Dict[str, List], title: str = "Active Inference Analysis") -> str:
        """
        Generate interactive HTML report with all visualizations.
        
        Args:
            history: Agent history dictionary.
            title: Report title.
        
        Returns:
            Path to generated HTML file.
        """
        if self.save_dir is None:
            print("⚠️  No save directory specified - cannot generate HTML report")
            return None
        
        import base64
        from io import BytesIO
        from pathlib import Path
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .visualization {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 5px;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 12px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <h1>🧠 {title}</h1>
    <div class="timestamp">Generated: {np.datetime64('now')}</div>
    
    <h2>📊 Key Metrics</h2>
    <div class="metrics">
"""
        
        # Add metrics
        beliefs = torch.stack(history['beliefs']) if history['beliefs'] else None
        if beliefs is not None:
            belief_range = (beliefs.max() - beliefs.min()).item()
            html_content += f"""
        <div class="metric-card">
            <div class="metric-value">{len(history['beliefs'])}</div>
            <div class="metric-label">Timesteps</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{belief_range:.4f}</div>
            <div class="metric-label">Belief Range</div>
        </div>
"""
        
        if 'efe_per_policy' in history and history['efe_per_policy']:
            num_policies = history['efe_per_policy'][0].shape[0]
            html_content += f"""
        <div class="metric-card">
            <div class="metric-value">{num_policies}</div>
            <div class="metric-label">Policies Evaluated</div>
        </div>
"""
        
        html_content += """
    </div>
"""
        
        # Function to embed image as base64
        def embed_image(fig):
            if fig is None:
                return ""
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            return f'<img src="data:image/png;base64,{img_str}" />'
        
        # Generate and embed all visualizations
        visualizations = [
            ("Free Energy Evolution", self.plot_free_energy(history)),
            ("Belief States with Uncertainty", self.plot_belief_with_uncertainty(history)),
            ("EFE Per Action", self.plot_efe_per_action(history)),
            ("Epistemic vs Pragmatic Balance", self.plot_epistemic_pragmatic_balance(history)),
        ]
        
        for viz_title, fig in visualizations:
            if fig is not None:
                html_content += f"""
    <div class="visualization">
        <h2>{viz_title}</h2>
        {embed_image(fig)}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML file
        html_path = self.save_dir / "analysis_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report generated: {html_path}")
        print(f"  Open in browser: file://{html_path.absolute()}")
        
        return str(html_path)
    
    def close_all(self):
        """Close all figures."""
        plt.close('all')

