"""Unified output management for all experiment artifacts."""

from pathlib import Path
from typing import Optional
from datetime import datetime


class OutputManager:
    """
    Manages unified output directory structure for all experiment artifacts.
    
    All outputs go into subdirectories of a single top-level output directory:
    
    output_root/
    ├── experiment_name/
    │   ├── config/              # Configuration files
    │   ├── logs/                # Metric logs and summaries
    │   ├── checkpoints/         # Agent state checkpoints
    │   ├── visualizations/      # Static plots and figures
    │   ├── animations/          # GIF/video animations
    │   ├── data/                # Raw data and episode records
    │   └── metadata/            # Experiment metadata
    """
    
    def __init__(
        self,
        output_root: str = "./output",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize unified output manager.
        
        Args:
            output_root: Root directory for all outputs.
            experiment_name: Name for this experiment (auto-generated if None).
        """
        self.output_root = Path(output_root)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"exp_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.output_root / experiment_name
        
        # Define subdirectory structure
        self.config_dir = self.experiment_dir / "config"
        self.logs_dir = self.experiment_dir / "logs"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.animations_dir = self.experiment_dir / "animations"
        self.data_dir = self.experiment_dir / "data"
        self.metadata_dir = self.experiment_dir / "metadata"
        
        # Create all directories
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create complete directory structure."""
        for directory in [
            self.config_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.visualizations_dir,
            self.animations_dir,
            self.data_dir,
            self.metadata_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, category: str, filename: str) -> Path:
        """
        Get full path for a file in a specific category.
        
        Args:
            category: One of 'config', 'logs', 'checkpoints', 'visualizations',
                     'animations', 'data', 'metadata'.
            filename: Name of the file.
        
        Returns:
            Full path to the file.
        """
        category_map = {
            'config': self.config_dir,
            'logs': self.logs_dir,
            'checkpoints': self.checkpoints_dir,
            'visualizations': self.visualizations_dir,
            'animations': self.animations_dir,
            'data': self.data_dir,
            'metadata': self.metadata_dir
        }
        
        if category not in category_map:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Must be one of {list(category_map.keys())}"
            )
        
        return category_map[category] / filename
    
    def get_experiment_summary(self) -> dict:
        """
        Get summary of experiment directory structure.
        
        Returns:
            Dictionary with directory information.
        """
        summary = {
            'experiment_name': self.experiment_name,
            'output_root': str(self.output_root),
            'experiment_dir': str(self.experiment_dir),
            'subdirectories': {
                'config': str(self.config_dir),
                'logs': str(self.logs_dir),
                'checkpoints': str(self.checkpoints_dir),
                'visualizations': str(self.visualizations_dir),
                'animations': str(self.animations_dir),
                'data': str(self.data_dir),
                'metadata': str(self.metadata_dir)
            }
        }
        return summary
    
    def print_structure(self):
        """Print directory structure to console."""
        print(f"\n{'='*70}")
        print(f"Experiment Output Structure: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"Root: {self.experiment_dir}")
        print(f"  ├── config/           # Configuration files")
        print(f"  ├── logs/             # Metric logs and summaries")
        print(f"  ├── checkpoints/      # Agent state checkpoints")
        print(f"  ├── visualizations/   # Static plots")
        print(f"  ├── animations/       # GIF/video files")
        print(f"  ├── data/             # Raw episode data")
        print(f"  └── metadata/         # Experiment metadata")
        print(f"{'='*70}\n")
    
    @staticmethod
    def list_experiments(output_root: str = "./output") -> list:
        """
        List all experiments in output root.
        
        Args:
            output_root: Root directory to scan.
        
        Returns:
            List of experiment directory names.
        """
        root = Path(output_root)
        if not root.exists():
            return []
        
        return [d.name for d in root.iterdir() if d.is_dir()]

