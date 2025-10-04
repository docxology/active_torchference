"""Logging system for Active Inference experiments."""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import torch


class Logger:
    """
    Logs experiment data, metrics, and agent states.
    
    Supports JSON, pickle, and PyTorch checkpoint formats.
    Uses OutputManager for unified directory structure.
    """
    
    def __init__(
        self,
        output_manager=None,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize logger.
        
        Args:
            output_manager: OutputManager instance for unified structure.
            log_dir: (Deprecated) Use output_manager instead.
            experiment_name: Name for experiment (auto-generated if None).
        """
        # Support both new OutputManager and legacy log_dir
        if output_manager is not None:
            self.output_manager = output_manager
            self.experiment_name = output_manager.experiment_name
            self.logs_dir = output_manager.logs_dir
            self.checkpoints_dir = output_manager.checkpoints_dir
            self.data_dir = output_manager.data_dir
            self.config_dir = output_manager.config_dir
        else:
            # Legacy mode for backward compatibility
            log_dir = log_dir or "./output"
            if experiment_name is None:
                experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.experiment_name = experiment_name
            experiment_dir = Path(log_dir) / experiment_name
            
            # Create subdirectories
            self.logs_dir = experiment_dir / "logs"
            self.checkpoints_dir = experiment_dir / "checkpoints"
            self.data_dir = experiment_dir / "data"
            self.config_dir = experiment_dir / "config"
            
            for directory in [self.logs_dir, self.checkpoints_dir, 
                            self.data_dir, self.config_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            self.output_manager = None
        
        # Initialize metric storage
        self.metrics = {
            "timesteps": [],
            "vfe": [],
            "efe": [],
            "distance_to_goal": [],
            "custom": {}
        }
        
        self.episode_count = 0
    
    def log_step(
        self,
        timestep: int,
        agent_info: Dict[str, Any],
        env_info: Dict[str, Any]
    ):
        """
        Log data from a single step.
        
        Args:
            timestep: Current timestep.
            agent_info: Information from agent.
            env_info: Information from environment.
        """
        self.metrics["timesteps"].append(timestep)
        
        if "vfe" in agent_info:
            vfe_val = agent_info["vfe"]
            if isinstance(vfe_val, torch.Tensor):
                vfe_val = vfe_val.item()
            self.metrics["vfe"].append(vfe_val)
        
        if "efe" in agent_info:
            efe_val = agent_info["efe"]
            if isinstance(efe_val, torch.Tensor):
                efe_val = efe_val.item()
            self.metrics["efe"].append(efe_val)
        
        if "distance_to_goal" in env_info:
            self.metrics["distance_to_goal"].append(env_info["distance_to_goal"])
    
    def log_episode(self, episode_num: int, episode_data: Dict[str, Any]):
        """
        Log episode-level data.
        
        Args:
            episode_num: Episode number.
            episode_data: Episode summary data.
        """
        episode_file = self.data_dir / f"episode_{episode_num:04d}.json"
        
        # Convert tensors to lists
        serializable_data = self._make_serializable(episode_data)
        
        with open(episode_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        self.episode_count += 1
    
    def log_custom(self, key: str, value: Any):
        """
        Log custom metric.
        
        Args:
            key: Metric name.
            value: Metric value.
        """
        if key not in self.metrics["custom"]:
            self.metrics["custom"][key] = []
        
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.tolist()
        
        self.metrics["custom"][key].append(value)
    
    def save_agent_state(self, agent, checkpoint_name: str = "final"):
        """
        Save agent state to checkpoint.
        
        Args:
            agent: Active Inference agent.
            checkpoint_name: Name for checkpoint file.
        """
        state_dict = agent.save_state()
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}_checkpoint.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state_dict, f)
    
    def save_metrics(self, filename: str = "metrics.json"):
        """
        Save accumulated metrics to file.
        
        Args:
            filename: Output filename.
        """
        metrics_path = self.logs_dir / filename
        serializable_metrics = self._make_serializable(self.metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def save_config(self, config: Any):
        """
        Save configuration to file.
        
        Args:
            config: Configuration object.
        """
        config_path = self.config_dir / "config.json"
        
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = vars(config)
        
        serializable_config = self._make_serializable(config_dict)
        
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert.
        
        Returns:
            JSON-serializable version.
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (torch.device, torch.dtype)):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(vars(obj))
        else:
            return obj
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary statistics.
        
        Returns:
            Dictionary of summary statistics.
        """
        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": len(self.metrics["timesteps"]),
            "episodes": self.episode_count,
        }
        
        if len(self.metrics["vfe"]) > 0:
            vfe_tensor = torch.tensor(self.metrics["vfe"])
            summary["vfe_mean"] = vfe_tensor.mean().item()
            summary["vfe_std"] = vfe_tensor.std().item()
            summary["vfe_final"] = self.metrics["vfe"][-1]
        
        if len(self.metrics["efe"]) > 0:
            efe_tensor = torch.tensor(self.metrics["efe"])
            summary["efe_mean"] = efe_tensor.mean().item()
            summary["efe_std"] = efe_tensor.std().item()
            summary["efe_final"] = self.metrics["efe"][-1]
        
        if len(self.metrics["distance_to_goal"]) > 0:
            dist_tensor = torch.tensor(self.metrics["distance_to_goal"])
            summary["distance_mean"] = dist_tensor.mean().item()
            summary["distance_final"] = self.metrics["distance_to_goal"][-1]
        
        return summary
    
    def print_summary(self):
        """Print experiment summary to console."""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print(f"Experiment Summary: {summary['experiment_name']}")
        print(f"{'='*60}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Episodes: {summary['episodes']}")
        
        if "vfe_mean" in summary:
            print(f"\nVariational Free Energy:")
            print(f"  Mean: {summary['vfe_mean']:.4f}")
            print(f"  Std:  {summary['vfe_std']:.4f}")
            print(f"  Final: {summary['vfe_final']:.4f}")
        
        if "efe_mean" in summary:
            print(f"\nExpected Free Energy:")
            print(f"  Mean: {summary['efe_mean']:.4f}")
            print(f"  Std:  {summary['efe_std']:.4f}")
            print(f"  Final: {summary['efe_final']:.4f}")
        
        if "distance_mean" in summary:
            print(f"\nDistance to Goal:")
            print(f"  Mean: {summary['distance_mean']:.4f}")
            print(f"  Final: {summary['distance_final']:.4f}")
        
        print(f"{'='*60}\n")

