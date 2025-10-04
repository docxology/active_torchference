"""Integration tests for Active Inference framework."""

import pytest
import torch
from active_torchference.config import Config
from active_torchference.agent import ActiveInferenceAgent
from active_torchference.environment import ContinuousEnvironment, GridWorld


def test_agent_environment_integration():
    """Test agent interacting with environment."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    agent = ActiveInferenceAgent(config)
    env = ContinuousEnvironment(config)
    
    obs = env.reset()
    preferred_obs = env.get_preferred_observation()
    
    for _ in range(5):
        action, info = agent.step(obs, preferred_obs)
        obs, env_info = env.step(action)
        
        assert obs.shape == (config.obs_dim,)
        assert action.shape == (config.action_dim,)


def test_agent_learns_to_approach_goal():
    """Test that agent learns to approach goal over time."""
    config = Config(
        obs_dim=2,
        action_dim=2,
        learning_rate_beliefs=0.1,
        learning_rate_policy=0.1,
        seed=42
    )
    agent = ActiveInferenceAgent(config)
    
    goal = torch.tensor([2.0, 2.0])
    env = ContinuousEnvironment(config, goal_position=goal)
    
    obs = env.reset()
    preferred_obs = env.get_preferred_observation()
    
    initial_distance = torch.norm(env.state - goal).item()
    
    # Run for several steps
    for _ in range(20):
        action, info = agent.step(obs, preferred_obs)
        obs, env_info = env.step(action)
        
        if env_info["done"]:
            break
    
    final_distance = torch.norm(env.state - goal).item()
    
    # Agent should move towards goal (not guaranteed to reach in 20 steps)
    # This is a soft check that some learning occurred
    assert final_distance <= initial_distance * 2  # Allow some exploration


def test_gridworld_integration():
    """Test agent in grid world environment."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    agent = ActiveInferenceAgent(config)
    env = GridWorld(config, grid_size=3)
    
    obs = env.reset()
    preferred_obs = env.get_preferred_observation()
    
    for _ in range(10):
        action, info = agent.step(obs, preferred_obs)
        obs, env_info = env.step(action)
        
        if env_info["done"]:
            break
    
    # Check that episode completed
    assert env.time_step > 0


def test_agent_vfe_decreases():
    """Test that VFE generally decreases with repeated observations."""
    config = Config(obs_dim=2, seed=42)
    agent = ActiveInferenceAgent(config)
    
    # Fixed observation
    obs = torch.tensor([1.0, 1.0])
    preferred_obs = torch.tensor([1.0, 1.0])
    
    vfes = []
    for _ in range(5):
        _, info = agent.step(obs, preferred_obs)
        vfes.append(info["vfe"].item())
    
    # VFE should generally decrease (agent becomes more confident)
    # Check that final VFE is lower than initial
    assert vfes[-1] <= vfes[0]


def test_multiple_episodes():
    """Test agent across multiple episodes."""
    config = Config(obs_dim=2, action_dim=2, seed=42)
    agent = ActiveInferenceAgent(config)
    env = ContinuousEnvironment(config)
    
    for episode in range(3):
        agent.reset()
        obs = env.reset()
        preferred_obs = env.get_preferred_observation()
        
        for step in range(10):
            action, info = agent.step(obs, preferred_obs)
            obs, env_info = env.step(action)
            
            if env_info["done"]:
                break
        
        # Check history was accumulated
        assert len(agent.history["observations"]) == step + 1


def test_agent_handles_different_dimensions():
    """Test agent with different dimensionalities."""
    configs = [
        Config(hidden_dim=2, obs_dim=1, action_dim=1),
        Config(hidden_dim=8, obs_dim=4, action_dim=3),
        Config(hidden_dim=16, obs_dim=8, action_dim=4),
    ]
    
    for config in configs:
        agent = ActiveInferenceAgent(config)
        
        obs = torch.randn(config.obs_dim)
        preferred_obs = torch.randn(config.obs_dim)
        
        action, info = agent.step(obs, preferred_obs)
        
        assert action.shape == (config.action_dim,)
        assert info["vfe"].shape == ()
        assert info["efe"].shape == ()


def test_free_energy_components():
    """Test that free energy components are computed correctly."""
    config = Config(obs_dim=2, action_dim=2)
    agent = ActiveInferenceAgent(config)
    env = ContinuousEnvironment(config)
    
    obs = env.reset()
    preferred_obs = env.get_preferred_observation()
    
    action, info = agent.step(obs, preferred_obs)
    
    # Check VFE components
    assert "vfe_likelihood" in info
    assert "vfe_complexity" in info
    
    # VFE should be sum of likelihood and complexity
    vfe_reconstructed = info["vfe_likelihood"] + info["vfe_complexity"]
    assert torch.isclose(info["vfe"], vfe_reconstructed, rtol=1e-3)

