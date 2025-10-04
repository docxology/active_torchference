"""Policy evaluation and selection for Active Inference."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from active_torchference.config import Config
from active_torchference.utils import softmax


class TransitionModel(nn.Module):
    """
    Transition model p(s'|s,a) for predicting state transitions.
    
    Maps current state and action to next state prediction.
    """
    
    def __init__(self, config: Config):
        """
        Initialize transition model.
        
        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        
        input_dim = config.hidden_dim + config.action_dim
        
        self.transition_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        self.to(config.device, config.dtype)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next state given current state and action.
        
        Args:
            state: Current state.
            action: Action to take.
        
        Returns:
            Predicted next state.
        """
        state_action = torch.cat([state, action], dim=-1)
        next_state = self.transition_net(state_action)
        return next_state


class PolicyEvaluator:
    """
    Evaluates policies using expected free energy and selects actions.
    
    Performs rollouts of candidate actions, evaluates them with EFE,
    and selects actions without reinforcement learning.
    """
    
    def __init__(
        self,
        config: Config,
        transition_model: TransitionModel,
        generative_model: nn.Module,
        efe_calculator
    ):
        """
        Initialize policy evaluator.
        
        Args:
            config: Configuration object.
            transition_model: Model for state transitions.
            generative_model: Model for observations.
            efe_calculator: Expected free energy calculator.
        """
        self.config = config
        self.transition_model = transition_model
        self.generative_model = generative_model
        self.efe_calculator = efe_calculator
        
        # Action distribution parameters
        self.action_mean = torch.zeros(
            config.action_dim,
            device=config.device,
            dtype=config.dtype,
            requires_grad=True
        )
        
        self.action_log_std = torch.zeros(
            config.action_dim,
            device=config.device,
            dtype=config.dtype,
            requires_grad=True
        )
    
    @property
    def action_std(self) -> torch.Tensor:
        """Get action standard deviation."""
        return torch.exp(self.action_log_std)
    
    def sample_actions(self, num_samples: int) -> torch.Tensor:
        """
        Sample candidate actions from current policy.
        
        Args:
            num_samples: Number of action samples.
        
        Returns:
            Action samples.
        """
        eps = torch.randn(
            num_samples,
            self.config.action_dim,
            device=self.config.device,
            dtype=self.config.dtype
        )
        actions = self.action_mean + self.action_std * eps
        return actions
    
    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        horizon: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform policy rollout from initial state.
        
        Args:
            initial_state: Starting state.
            actions: Sequence of actions (num_rollouts, horizon, action_dim).
            horizon: Planning horizon (uses config default if None).
        
        Returns:
            Tuple of (predicted_states, predicted_observations).
        """
        if horizon is None:
            horizon = self.config.horizon
        
        # Reshape actions if needed
        if actions.dim() == 2:
            # Single action sequence
            actions = actions.unsqueeze(0)
        
        num_rollouts = actions.shape[0]
        
        # Initialize rollout storage
        states = [initial_state.unsqueeze(0).expand(num_rollouts, -1)]
        observations = []
        
        # Perform rollout
        for t in range(horizon):
            current_state = states[-1]
            current_action = actions[:, t] if t < actions.shape[1] else actions[:, -1]
            
            # Predict next state
            next_state = self.transition_model(current_state, current_action)
            states.append(next_state)
            
            # Predict observation
            predicted_obs = self.generative_model(next_state)
            observations.append(predicted_obs)
        
        return states[1:], observations
    
    def evaluate_policy(
        self,
        current_belief_mean: torch.Tensor,
        current_belief_std: torch.Tensor,
        preferred_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Evaluate multiple policies using expected free energy.
        
        Returns EFE for each policy (action sequence), not averaged.
        
        Args:
            current_belief_mean: Mean of current belief.
            current_belief_std: Std of current belief.
            preferred_obs: Goal/preferred observations.
        
        Returns:
            Tuple of (efe_per_policy, efe_per_policy_per_timestep, info_dict).
            - efe_per_policy: [num_rollouts] - total EFE for each policy
            - efe_per_policy_per_timestep: [horizon, num_rollouts] - EFE per timestep per policy
        """
        # Sample candidate policies (action sequences)
        action_samples = self.sample_actions(self.config.num_rollouts)
        
        # Expand actions for horizon
        action_sequences = action_samples.unsqueeze(1).expand(
            -1, self.config.horizon, -1
        )
        
        # Perform rollouts for each policy
        states, observations = self.rollout(
            current_belief_mean,
            action_sequences,
            self.config.horizon
        )
        
        # Compute expected free energy for each policy at each timestep
        efe_per_timestep = []
        efe_components_history = []
        
        for t in range(self.config.horizon):
            predicted_obs = observations[t]  # [num_rollouts, obs_dim]
            
            # Estimate observation uncertainty
            predicted_obs_std = torch.ones_like(predicted_obs) * 0.1
            
            # Estimate state entropy (information gain potential)
            state_entropy = torch.sum(
                torch.log(current_belief_std + 1e-8),
                dim=-1
            ).expand(self.config.num_rollouts)
            
            # Compute EFE for each policy at this timestep
            efe_t, components = self.efe_calculator.compute(
                predicted_obs,
                predicted_obs_std,
                preferred_obs.expand_as(predicted_obs),
                state_entropy
            )
            
            efe_per_timestep.append(efe_t)  # [num_rollouts]
            efe_components_history.append(components)
        
        # Stack to get [horizon, num_rollouts]
        efe_per_policy_per_timestep = torch.stack(efe_per_timestep, dim=0)
        
        # Sum across horizon to get total EFE per policy [num_rollouts]
        efe_per_policy = efe_per_policy_per_timestep.sum(dim=0)
        
        info = {
            "action_samples": action_samples.detach(),
            "efe_per_policy": efe_per_policy.detach(),
            "efe_per_policy_per_timestep": efe_per_policy_per_timestep.detach(),
            "mean_efe": efe_per_policy.mean().detach(),
            "min_efe": efe_per_policy.min().detach(),
            "max_efe": efe_per_policy.max().detach(),
            "best_policy_idx": efe_per_policy.argmin().item(),
            "predicted_observations": observations[-1].detach(),
            "efe_components": efe_components_history
        }
        
        return efe_per_policy, efe_per_policy_per_timestep, info
    
    def update_policy(
        self,
        gradient: Tuple[torch.Tensor, torch.Tensor],
        learning_rate: Optional[float] = None
    ):
        """
        Update policy parameters using gradient descent on EFE.
        
        Args:
            gradient: Tuple of (grad_mean, grad_log_std).
            learning_rate: Learning rate (uses config default if None).
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate_policy
        
        grad_mean, grad_log_std = gradient
        
        with torch.no_grad():
            self.action_mean -= learning_rate * grad_mean
            self.action_log_std -= learning_rate * grad_log_std
    
    def select_action(
        self,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Select action from updated policy posterior.
        
        Args:
            deterministic: If True, return mean action. Otherwise sample.
        
        Returns:
            Selected action.
        """
        if deterministic:
            return self.action_mean.detach()
        else:
            eps = torch.randn_like(self.action_mean)
            action = self.action_mean + self.action_std * eps
            return action.detach()
    
    def reset(self):
        """Reset policy to default distribution."""
        with torch.no_grad():
            self.action_mean.zero_()
            self.action_log_std.zero_()
    
    def to_dict(self) -> dict:
        """Convert policy to dictionary."""
        return {
            "action_mean": self.action_mean.detach().cpu().numpy().tolist(),
            "action_log_std": self.action_log_std.detach().cpu().numpy().tolist()
        }

