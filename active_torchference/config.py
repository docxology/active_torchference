"""Configuration system for Active Inference models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class Config:
    """
    Configuration for Active Inference agent and environment.
    
    Attributes:
        hidden_dim: Dimensionality of hidden states.
        obs_dim: Dimensionality of observations.
        action_dim: Dimensionality of actions.
        learning_rate_beliefs: Learning rate for belief updates.
        learning_rate_policy: Learning rate for policy updates.
        num_policy_iterations: Number of iterations for policy optimization.
        num_rollouts: Number of policy rollouts for expected free energy.
        horizon: Planning horizon for policy evaluation.
        temperature: Temperature for action selection (lower = more deterministic).
        device: PyTorch device ('cpu' or 'cuda').
        dtype: PyTorch dtype for computations.
        precision_obs: Precision (inverse variance) of observation likelihood.
        precision_state: Precision of state transitions.
        precision_prior: Precision of prior beliefs.
        epistemic_weight: Weight for epistemic value in expected free energy.
        pragmatic_weight: Weight for pragmatic value in expected free energy.
        seed: Random seed for reproducibility.
    """
    
    # Dimensionality
    hidden_dim: int = 4
    obs_dim: int = 2
    action_dim: int = 2
    
    # Learning rates
    learning_rate_beliefs: float = 0.1
    learning_rate_policy: float = 0.05
    
    # Belief updating
    num_belief_iterations: int = 5
    
    # Policy evaluation
    num_policy_iterations: int = 10
    num_rollouts: int = 5
    horizon: int = 3
    temperature: float = 1.0
    
    # Device configuration
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    # Precision parameters (inverse variances)
    precision_obs: float = 1.0
    precision_state: float = 1.0
    precision_prior: float = 1.0
    
    # Expected free energy weights
    epistemic_weight: float = 1.0
    pragmatic_weight: float = 1.0
    
    # Reproducibility
    seed: Optional[int] = None
    
    # Additional parameters
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Convert device string to torch.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "hidden_dim": self.hidden_dim,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "learning_rate_beliefs": self.learning_rate_beliefs,
            "learning_rate_policy": self.learning_rate_policy,
            "num_policy_iterations": self.num_policy_iterations,
            "num_rollouts": self.num_rollouts,
            "horizon": self.horizon,
            "temperature": self.temperature,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "precision_obs": self.precision_obs,
            "precision_state": self.precision_state,
            "precision_prior": self.precision_prior,
            "epistemic_weight": self.epistemic_weight,
            "pragmatic_weight": self.pragmatic_weight,
            "seed": self.seed,
            "extra": self.extra,
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)

