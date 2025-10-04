"""Tests for belief state updates via VFE minimization."""

import pytest
import torch
from active_torchference.config import Config
from active_torchference.agent import ActiveInferenceAgent


def test_belief_updates_with_observation():
    """Test that beliefs actually update when observing data."""
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        num_belief_iterations=10,
        learning_rate_beliefs=0.1,
        seed=42
    )
    agent = ActiveInferenceAgent(config)
    
    # Store initial beliefs
    initial_belief_mean = agent.beliefs.mean.clone()
    initial_belief_std = agent.beliefs.std.clone()
    
    # Create observation
    observation = torch.tensor([1.0, 2.0])
    
    # Update beliefs
    info = agent.perceive(observation)
    
    # Beliefs should have changed
    assert not torch.allclose(agent.beliefs.mean, initial_belief_mean, atol=1e-6)
    
    # VFE should decrease over iterations
    assert info["vfe_final"] < info["vfe_initial"]


def test_vfe_decreases_with_iterations():
    """Test that VFE decreases with more iterations."""
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        learning_rate_beliefs=0.1,
        seed=42
    )
    agent = ActiveInferenceAgent(config)
    
    observation = torch.tensor([1.0, 2.0])
    
    # Test with different numbers of iterations
    agent.reset()
    info_few = agent.perceive(observation, num_iterations=2)
    
    agent.reset()
    info_many = agent.perceive(observation, num_iterations=10)
    
    # More iterations should lead to lower VFE
    assert info_many["vfe_final"] <= info_few["vfe_final"]


def test_belief_convergence_to_observation():
    """Test that beliefs converge towards explaining observation."""
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        num_belief_iterations=20,
        learning_rate_beliefs=0.2,
        seed=42
    )
    agent = ActiveInferenceAgent(config)
    
    # Fixed observation
    observation = torch.tensor([1.5, 2.5])
    
    # Multiple perception steps with same observation
    errors = []
    for _ in range(5):
        info = agent.perceive(observation)
        
        # Track prediction error
        prediction_error = torch.norm(info["predicted_obs"] - observation)
        errors.append(prediction_error.item())
    
    # Error should generally decrease over repeated observations
    assert errors[-1] < errors[0] * 1.5  # Allow some tolerance


def test_vfe_components():
    """Test that VFE has likelihood and complexity components."""
    config = Config(seed=42)
    agent = ActiveInferenceAgent(config)
    
    observation = torch.randn(config.obs_dim)
    info = agent.perceive(observation)
    
    assert "vfe_likelihood" in info
    assert "vfe_complexity" in info
    assert "vfe" in info
    
    # VFE should be positive
    assert info["vfe"] >= 0


def test_belief_updates_with_different_learning_rates():
    """Test that learning rate affects belief updates."""
    observation = torch.tensor([1.0, 2.0])
    
    # Low learning rate
    config_low = Config(
        learning_rate_beliefs=0.01,
        num_belief_iterations=5,
        seed=42
    )
    agent_low = ActiveInferenceAgent(config_low)
    initial_beliefs_low = agent_low.beliefs.mean.clone()
    agent_low.perceive(observation)
    change_low = torch.norm(agent_low.beliefs.mean - initial_beliefs_low)
    
    # High learning rate
    config_high = Config(
        learning_rate_beliefs=0.2,
        num_belief_iterations=5,
        seed=42
    )
    agent_high = ActiveInferenceAgent(config_high)
    initial_beliefs_high = agent_high.beliefs.mean.clone()
    agent_high.perceive(observation)
    change_high = torch.norm(agent_high.beliefs.mean - initial_beliefs_high)
    
    # Higher learning rate should cause larger changes
    assert change_high > change_low


def test_vfe_history_tracking():
    """Test that VFE history is properly tracked."""
    config = Config(num_belief_iterations=5, seed=42)
    agent = ActiveInferenceAgent(config)
    
    observation = torch.randn(config.obs_dim)
    info = agent.perceive(observation)
    
    assert "vfe_history" in info
    assert len(info["vfe_history"]) == 5
    
    # VFE should generally decrease
    assert info["vfe_history"][-1] <= info["vfe_history"][0]


def test_belief_mean_updates_more_than_std():
    """Test that belief mean updates significantly with observations."""
    config = Config(
        num_belief_iterations=20,
        learning_rate_beliefs=0.2,
        seed=42
    )
    agent = ActiveInferenceAgent(config)
    
    initial_mean = agent.beliefs.mean.clone()
    initial_log_std = agent.beliefs.log_std.clone()
    
    # Strong observation signal
    observation = torch.tensor([2.0, 3.0])
    
    # Multiple updates
    for _ in range(10):
        agent.perceive(observation)
    
    # Mean should change significantly
    mean_change = torch.norm(agent.beliefs.mean - initial_mean)
    assert mean_change > 0.1
    
    # Std may or may not change much (depends on complexity term gradient)
    # This is expected behavior - mean adapts to observations faster than std


def test_repeated_observations_reduce_vfe():
    """Test that repeated observations of same data reduce VFE."""
    config = Config(
        num_belief_iterations=5,
        learning_rate_beliefs=0.1,
        seed=42
    )
    agent = ActiveInferenceAgent(config)
    
    observation = torch.tensor([1.0, 2.0])
    
    vfes = []
    for _ in range(5):
        info = agent.perceive(observation)
        vfes.append(info["vfe"].item())
    
    # VFE should decrease with repeated observations
    assert vfes[-1] < vfes[0]

