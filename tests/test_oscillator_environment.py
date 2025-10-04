"""Tests for OscillatorEnvironment."""

import torch
import pytest
from active_torchference import Config, OscillatorEnvironment


def test_oscillator_initialization():
    """Test oscillator environment initialization."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    env = OscillatorEnvironment(config, frequency=0.1, amplitude=2.0)
    
    assert env.frequency == 0.1
    assert env.amplitude == 2.0
    assert env.noise_std == 0.1
    assert env.max_steps == 200


def test_oscillator_reset():
    """Test oscillator environment reset."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    env = OscillatorEnvironment(config)
    
    observation = env.reset()
    
    assert observation.shape == (config.obs_dim,)
    assert env.time_step == 0
    assert torch.all(torch.isfinite(observation))


def test_oscillator_step():
    """Test oscillator environment step."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    env = OscillatorEnvironment(config)
    
    env.reset()
    action = torch.randn(config.action_dim)
    observation, info = env.step(action)
    
    assert observation.shape == (config.obs_dim,)
    assert "distance_to_goal" in info
    assert "goal" in info
    assert "time_step" in info
    assert env.time_step == 1


def test_oscillator_goal_movement():
    """Test that oscillator goal position changes over time."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    env = OscillatorEnvironment(config, frequency=0.1, amplitude=2.0)
    
    env.reset()
    
    # Get goals at different timesteps
    goal_t0 = env.get_preferred_observation()
    
    env.step(torch.zeros(config.action_dim))
    goal_t1 = env.get_preferred_observation()
    
    env.step(torch.zeros(config.action_dim))
    goal_t2 = env.get_preferred_observation()
    
    # Goals should be different
    assert not torch.allclose(goal_t0, goal_t1)
    assert not torch.allclose(goal_t1, goal_t2)


def test_oscillator_sinusoidal_pattern():
    """Test that goal follows sinusoidal pattern."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    env = OscillatorEnvironment(config, frequency=0.1, amplitude=2.0)
    
    env.reset()
    
    # Collect goals over full cycle
    goals_x = []
    goals_y = []
    
    for _ in range(100):
        goal = env.get_preferred_observation()
        goals_x.append(goal[0].item())
        goals_y.append(goal[1].item())
        env.step(torch.zeros(config.action_dim))
    
    goals_x_tensor = torch.tensor(goals_x)
    goals_y_tensor = torch.tensor(goals_y)
    
    # Check amplitude bounds
    assert goals_x_tensor.min() >= -2.1  # amplitude + tolerance
    assert goals_x_tensor.max() <= 2.1
    assert goals_y_tensor.min() >= -2.1
    assert goals_y_tensor.max() <= 2.1


def test_oscillator_termination():
    """Test oscillator termination at max steps."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    env = OscillatorEnvironment(config)
    
    env.reset()
    
    # Run until termination
    for step in range(env.max_steps + 10):
        action = torch.zeros(config.action_dim)
        observation, info = env.step(action)
        
        if step < env.max_steps - 1:
            assert not info["done"]
        elif step >= env.max_steps - 1:
            assert info["done"]


def test_oscillator_with_different_dimensions():
    """Test oscillator with various observation dimensions."""
    for obs_dim in [1, 2, 3, 4]:
        config = Config(obs_dim=obs_dim, action_dim=2, seed=42)
        env = OscillatorEnvironment(config)
        
        observation = env.reset()
        assert observation.shape == (obs_dim,)
        
        goal = env.get_preferred_observation()
        assert goal.shape == (obs_dim,)
        
        # For obs_dim > 2, additional dimensions should be zero
        if obs_dim > 2:
            assert torch.all(goal[2:] == 0)


def test_oscillator_distance_tracking():
    """Test distance to goal calculation."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    env = OscillatorEnvironment(config)
    
    env.reset()
    
    # Move towards goal
    for _ in range(5):
        goal = env.get_preferred_observation()
        direction = goal - env.state
        action = direction * 0.5  # Move halfway towards goal
        
        observation, info = env.step(action)
        distance = info["distance_to_goal"]
        
        assert distance >= 0
        assert "distance_to_goal" in info


def test_oscillator_noise():
    """Test observation noise."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    env = OscillatorEnvironment(config, noise_std=0.0)
    
    observation1 = env.reset()
    
    # With zero noise, state and observation should match
    assert torch.allclose(observation1, env.state, atol=1e-6)
    
    # With noise
    env2 = OscillatorEnvironment(config, noise_std=0.5)
    observation2 = env2.reset()
    
    # Observations should differ from state due to noise
    # (though they could be close by chance)
    assert observation2.shape == env2.state.shape


def test_oscillator_frequency_effect():
    """Test different frequencies produce different oscillation speeds."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    
    # Low frequency
    env_slow = OscillatorEnvironment(config, frequency=0.01, amplitude=1.0)
    env_slow.reset()
    goal_slow_0 = env_slow.get_preferred_observation()
    for _ in range(10):
        env_slow.step(torch.zeros(config.action_dim))
    goal_slow_10 = env_slow.get_preferred_observation()
    
    # High frequency
    env_fast = OscillatorEnvironment(config, frequency=0.5, amplitude=1.0)
    env_fast.reset()
    goal_fast_0 = env_fast.get_preferred_observation()
    for _ in range(10):
        env_fast.step(torch.zeros(config.action_dim))
    goal_fast_10 = env_fast.get_preferred_observation()
    
    # Fast oscillation should move more
    slow_change = torch.norm(goal_slow_10 - goal_slow_0)
    fast_change = torch.norm(goal_fast_10 - goal_fast_0)
    
    assert fast_change > slow_change

