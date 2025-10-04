"""Tests for environments."""

import pytest
import torch
from active_torchference.config import Config
from active_torchference.environment import ContinuousEnvironment, GridWorld


def test_continuous_environment_initialization():
    """Test continuous environment initialization."""
    config = Config(obs_dim=2)
    env = ContinuousEnvironment(config)
    
    assert env.config == config
    assert env.goal_position.shape == (2,)


def test_continuous_environment_reset():
    """Test environment reset."""
    config = Config(obs_dim=2, seed=42)
    env = ContinuousEnvironment(config)
    
    obs = env.reset()
    
    assert obs.shape == (2,)
    assert env.state is not None
    assert env.time_step == 0


def test_continuous_environment_step():
    """Test environment step."""
    config = Config(obs_dim=2, action_dim=2)
    env = ContinuousEnvironment(config)
    
    env.reset()
    action = torch.tensor([0.1, 0.1])
    obs, info = env.step(action)
    
    assert obs.shape == (2,)
    assert "state" in info
    assert "distance_to_goal" in info
    assert "time_step" in info
    assert env.time_step == 1


def test_continuous_environment_goal_reaching():
    """Test reaching goal."""
    config = Config(obs_dim=2)
    goal = torch.tensor([1.0, 1.0])
    env = ContinuousEnvironment(config, goal_position=goal)
    
    # Reset near goal
    env.reset()
    with torch.no_grad():
        env.state = goal.clone()
    
    action = torch.zeros(2)
    obs, info = env.step(action)
    
    assert info["distance_to_goal"] < 0.1
    assert info["done"]


def test_continuous_environment_preferred_observation():
    """Test getting preferred observation."""
    config = Config(obs_dim=2)
    goal = torch.tensor([2.0, 3.0])
    env = ContinuousEnvironment(config, goal_position=goal)
    
    preferred_obs = env.get_preferred_observation()
    
    assert torch.allclose(preferred_obs, goal)


def test_gridworld_initialization():
    """Test grid world initialization."""
    config = Config(obs_dim=2)
    env = GridWorld(config, grid_size=5)
    
    assert env.grid_size == 5
    assert env.goal_position == (4, 4)


def test_gridworld_reset():
    """Test grid world reset."""
    config = Config(obs_dim=2, seed=42)
    env = GridWorld(config)
    
    obs = env.reset()
    
    assert obs.shape == (2,)
    assert env.position is not None
    assert 0 <= env.position[0] < env.grid_size
    assert 0 <= env.position[1] < env.grid_size


def test_gridworld_step():
    """Test grid world step."""
    config = Config(obs_dim=2)
    env = GridWorld(config)
    
    env.reset()
    initial_pos = env.position
    
    # Move right
    action = torch.tensor([1.0, 0.0])
    obs, info = env.step(action)
    
    assert obs.shape == (2,)
    assert "position" in info
    assert "at_goal" in info


def test_gridworld_bounds():
    """Test grid world boundary handling."""
    config = Config(obs_dim=2)
    env = GridWorld(config, grid_size=5)
    
    # Reset to corner
    env.reset()
    env.position = (0, 0)
    
    # Try to move out of bounds
    action = torch.tensor([-2.0, -2.0])
    obs, info = env.step(action)
    
    # Should stay in bounds
    assert 0 <= env.position[0] < env.grid_size
    assert 0 <= env.position[1] < env.grid_size


def test_gridworld_goal_reaching():
    """Test reaching goal in grid world."""
    config = Config(obs_dim=2)
    env = GridWorld(config)
    
    env.reset()
    env.position = env.goal_position
    
    action = torch.zeros(2)
    obs, info = env.step(action)
    
    assert info["at_goal"]
    assert info["done"]


def test_gridworld_preferred_observation():
    """Test grid world preferred observation."""
    config = Config(obs_dim=2)
    env = GridWorld(config, grid_size=5, goal_position=(3, 4))
    
    preferred_obs = env.get_preferred_observation()
    
    assert preferred_obs.shape == (2,)
    # Check normalized goal position
    expected = torch.tensor([3.0 / 5, 4.0 / 5])
    assert torch.allclose(preferred_obs, expected)

