"""Belief state management for Active Inference."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from active_torchference.config import Config


class BeliefState:
    """
    Manages belief states (hidden states) for Active Inference agent.
    
    Represents approximate posterior q(s) over hidden states.
    """
    
    def __init__(self, config: Config):
        """
        Initialize belief state.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        # Initialize belief mean and log std
        self.mean = torch.zeros(
            config.hidden_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )
        
        # Log std for numerical stability
        self.log_std = torch.zeros(
            config.hidden_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )
        
        # Prior parameters
        self.prior_mean = torch.zeros(
            config.hidden_dim,
            device=self.device,
            dtype=self.dtype
        )
        
        self.prior_log_std = torch.zeros(
            config.hidden_dim,
            device=self.device,
            dtype=self.dtype
        )
    
    @property
    def std(self) -> torch.Tensor:
        """Get standard deviation from log_std."""
        return torch.exp(self.log_std)
    
    @property
    def prior_std(self) -> torch.Tensor:
        """Get prior standard deviation."""
        return torch.exp(self.prior_log_std)
    
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from belief distribution.
        
        Args:
            num_samples: Number of samples to draw.
        
        Returns:
            Samples from q(s).
        """
        eps = torch.randn(
            num_samples,
            self.config.hidden_dim,
            device=self.device,
            dtype=self.dtype
        )
        samples = self.mean + self.std * eps
        return samples
    
    def update(
        self,
        gradient: Tuple[torch.Tensor, torch.Tensor],
        learning_rate: Optional[float] = None
    ):
        """
        Update belief state using gradient descent.
        
        Args:
            gradient: Tuple of (grad_mean, grad_log_std).
            learning_rate: Learning rate (uses config default if None).
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate_beliefs
        
        grad_mean, grad_log_std = gradient
        
        with torch.no_grad():
            self.mean -= learning_rate * grad_mean
            self.log_std -= learning_rate * grad_log_std
    
    def reset(self):
        """Reset belief state to prior."""
        with torch.no_grad():
            self.mean.copy_(self.prior_mean)
            self.log_std.copy_(self.prior_log_std)
    
    def set_prior(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Set prior distribution parameters.
        
        Args:
            mean: Prior mean.
            std: Prior standard deviation.
        """
        with torch.no_grad():
            self.prior_mean = mean.to(self.device, self.dtype)
            self.prior_log_std = torch.log(std.to(self.device, self.dtype))
    
    def detach(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get detached copy of belief parameters.
        
        Returns:
            Tuple of (mean, std) without gradients.
        """
        return self.mean.detach().clone(), self.std.detach().clone()
    
    def to_dict(self) -> dict:
        """Convert belief state to dictionary."""
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "log_std": self.log_std.detach().cpu().numpy().tolist(),
            "prior_mean": self.prior_mean.detach().cpu().numpy().tolist(),
            "prior_log_std": self.prior_log_std.detach().cpu().numpy().tolist()
        }


class GenerativeModel(nn.Module):
    """
    Generative model p(o|s) for Active Inference.
    
    Maps hidden states to predicted observations.
    """
    
    def __init__(self, config: Config):
        """
        Initialize generative model.
        
        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        
        # Simple linear generative model
        self.observation_model = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim * 2, config.obs_dim)
        )
        
        self.to(config.device, config.dtype)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Generate predicted observation from hidden state.
        
        Args:
            hidden_state: Hidden state tensor.
        
        Returns:
            Predicted observation.
        """
        return self.observation_model(hidden_state)
    
    def predict_with_uncertainty(
        self,
        belief_mean: torch.Tensor,
        belief_std: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict observation with uncertainty propagation.
        
        Args:
            belief_mean: Mean of belief distribution.
            belief_std: Std of belief distribution.
            num_samples: Number of samples for Monte Carlo estimate.
        
        Returns:
            Tuple of (predicted_obs_mean, predicted_obs_std).
        """
        # Sample from belief distribution
        eps = torch.randn(
            num_samples,
            self.config.hidden_dim,
            device=self.config.device,
            dtype=self.config.dtype
        )
        state_samples = belief_mean.unsqueeze(0) + belief_std.unsqueeze(0) * eps
        
        # Forward pass through generative model
        obs_samples = self(state_samples)
        
        # Compute mean and std
        obs_mean = torch.mean(obs_samples, dim=0)
        obs_std = torch.std(obs_samples, dim=0)
        
        return obs_mean, obs_std

