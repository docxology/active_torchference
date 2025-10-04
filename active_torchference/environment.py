"""Environment base class and implementations for Active Inference."""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from active_torchference.config import Config


class Environment(ABC):
    """
    Abstract base class for Active Inference environments.
    
    Provides observations to agent and processes actions.
    """
    
    def __init__(self, config: Config):
        """
        Initialize environment.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.state = None
        self.time_step = 0
    
    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation.
        """
        pass
    
    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Execute action and return next observation.
        
        Args:
            action: Action to execute.
        
        Returns:
            Tuple of (observation, info_dict).
        """
        pass
    
    @abstractmethod
    def get_preferred_observation(self) -> torch.Tensor:
        """
        Return goal/preferred observation for agent.
        
        Returns:
            Preferred observation tensor.
        """
        pass
    
    def render(self) -> Optional[Any]:
        """
        Render environment state (optional).
        
        Returns:
            Rendering output or None.
        """
        return None


class ContinuousEnvironment(Environment):
    """
    Simple continuous environment for testing Active Inference.
    
    Agent must navigate to a goal location in continuous space.
    """
    
    def __init__(
        self,
        config: Config,
        goal_position: Optional[torch.Tensor] = None,
        noise_std: float = 0.1
    ):
        """
        Initialize continuous environment.
        
        Args:
            config: Configuration object.
            goal_position: Target position (defaults to [1.0, 1.0]).
            noise_std: Observation noise standard deviation.
        """
        super().__init__(config)
        
        if goal_position is None:
            goal_position = torch.ones(config.obs_dim, device=self.device, dtype=self.dtype)
        
        self.goal_position = goal_position
        self.noise_std = noise_std
        self.max_steps = 100
    
    def reset(self) -> torch.Tensor:
        """Reset to random initial position."""
        self.state = torch.randn(
            self.config.obs_dim,
            device=self.device,
            dtype=self.dtype
        ) * 0.5
        self.time_step = 0
        
        # Add observation noise
        observation = self.state + torch.randn_like(self.state) * self.noise_std
        return observation
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Update position based on action.
        
        Args:
            action: Movement direction and magnitude.
        
        Returns:
            Tuple of (observation, info).
        """
        # Ensure action has correct shape
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        # Truncate action to observation dimension if needed
        action = action[:self.config.obs_dim]
        
        # Update state
        self.state = self.state + action * 0.1
        self.time_step += 1
        
        # Add observation noise
        observation = self.state + torch.randn_like(self.state) * self.noise_std
        
        # Compute distance to goal
        distance = torch.norm(self.state - self.goal_position)
        
        info = {
            "state": self.state.detach(),
            "distance_to_goal": distance.item(),
            "time_step": self.time_step,
            "done": self.time_step >= self.max_steps or distance < 0.1
        }
        
        return observation, info
    
    def get_preferred_observation(self) -> torch.Tensor:
        """Return goal position as preferred observation."""
        return self.goal_position.clone()


class GridWorld(Environment):
    """
    Discrete grid world environment for Active Inference.
    
    Agent navigates a grid to reach a goal position.
    """
    
    def __init__(
        self,
        config: Config,
        grid_size: int = 5,
        goal_position: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize grid world.
        
        Args:
            config: Configuration object.
            grid_size: Size of square grid.
            goal_position: Goal (x, y) coordinates.
        """
        super().__init__(config)
        
        self.grid_size = grid_size
        self.goal_position = goal_position or (grid_size - 1, grid_size - 1)
        self.max_steps = grid_size * grid_size * 2
    
    def reset(self) -> torch.Tensor:
        """Reset to random initial position."""
        self.position = (
            torch.randint(0, self.grid_size, (1,)).item(),
            torch.randint(0, self.grid_size, (1,)).item()
        )
        self.time_step = 0
        
        observation = self._position_to_observation(self.position)
        return observation
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Move in grid based on action.
        
        Args:
            action: Movement action (interpreted as [dx, dy]).
        
        Returns:
            Tuple of (observation, info).
        """
        # Discretize action to grid movement
        dx = 1 if action[0] > 0.5 else (-1 if action[0] < -0.5 else 0)
        dy = 1 if action[1] > 0.5 else (-1 if action[1] < -0.5 else 0)
        
        # Update position with bounds checking
        new_x = max(0, min(self.grid_size - 1, self.position[0] + dx))
        new_y = max(0, min(self.grid_size - 1, self.position[1] + dy))
        self.position = (new_x, new_y)
        self.time_step += 1
        
        observation = self._position_to_observation(self.position)
        
        # Check if goal reached
        at_goal = self.position == self.goal_position
        
        # Compute Manhattan distance to goal
        distance_to_goal = abs(self.position[0] - self.goal_position[0]) + abs(self.position[1] - self.goal_position[1])
        
        info = {
            "position": self.position,
            "at_goal": at_goal,
            "distance_to_goal": float(distance_to_goal),
            "time_step": self.time_step,
            "done": at_goal or self.time_step >= self.max_steps
        }
        
        return observation, info
    
    def get_preferred_observation(self) -> torch.Tensor:
        """Return goal position as preferred observation."""
        return self._position_to_observation(self.goal_position)
    
    def _position_to_observation(self, position: Tuple[int, int]) -> torch.Tensor:
        """
        Convert grid position to observation vector.
        
        Args:
            position: (x, y) position.
        
        Returns:
            Observation tensor.
        """
        # Normalize position to [0, 1] range
        obs = torch.tensor(
            [position[0] / self.grid_size, position[1] / self.grid_size],
            device=self.device,
            dtype=self.dtype
        )
        
        # Pad to observation dimension if needed
        if obs.shape[0] < self.config.obs_dim:
            padding = torch.zeros(
                self.config.obs_dim - obs.shape[0],
                device=self.device,
                dtype=self.dtype
            )
            obs = torch.cat([obs, padding])
        
        return obs[:self.config.obs_dim]


class OscillatorEnvironment(Environment):
    """
    Oscillating goal environment for Active Inference.
    
    Agent must synchronize with oscillating goal position.
    Goal position oscillates sinusoidally, requiring agent to predict
    and track the moving target.
    """
    
    def __init__(
        self,
        config: Config,
        frequency: float = 0.1,
        amplitude: float = 2.0,
        noise_std: float = 0.1
    ):
        """
        Initialize oscillator environment.
        
        Args:
            config: Configuration object.
            frequency: Oscillation frequency.
            amplitude: Oscillation amplitude.
            noise_std: Observation noise standard deviation.
        """
        super().__init__(config)
        self.frequency = frequency
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.max_steps = 200
    
    def reset(self) -> torch.Tensor:
        """Reset environment."""
        self.state = torch.zeros(
            self.config.obs_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.time_step = 0
        
        observation = self.state + torch.randn_like(self.state) * self.noise_std
        return observation
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Execute action and update state."""
        # Ensure action has correct shape
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        action = action[:self.config.obs_dim]
        
        # Update state with action
        self.state = self.state + action * 0.1
        self.time_step += 1
        
        # Add observation noise
        observation = self.state + torch.randn_like(self.state) * self.noise_std
        
        # Compute distance to current goal position
        current_goal = self._get_current_goal()
        distance = torch.norm(self.state - current_goal)
        
        info = {
            "state": self.state.detach(),
            "goal": current_goal.detach(),
            "distance_to_goal": distance.item(),
            "time_step": self.time_step,
            "done": self.time_step >= self.max_steps
        }
        
        return observation, info
    
    def get_preferred_observation(self) -> torch.Tensor:
        """Get current goal position (oscillating)."""
        return self._get_current_goal()
    
    def _get_current_goal(self) -> torch.Tensor:
        """Compute oscillating goal position."""
        t = self.time_step * self.frequency
        
        goal = torch.zeros(
            self.config.obs_dim,
            device=self.device,
            dtype=self.dtype
        )
        goal[0] = self.amplitude * torch.sin(torch.tensor(t))
        if self.config.obs_dim > 1:
            goal[1] = self.amplitude * torch.cos(torch.tensor(t))
        
        return goal

