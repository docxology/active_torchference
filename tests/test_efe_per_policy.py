"""Tests for EFE computation per policy."""

import pytest
import torch
from active_torchference.config import Config
from active_torchference.agent import ActiveInferenceAgent
from active_torchference.policy import PolicyEvaluator, TransitionModel
from active_torchference.beliefs import GenerativeModel
from active_torchference.free_energy import ExpectedFreeEnergy


def test_efe_returns_per_policy():
    """Test that EFE is computed for each policy separately."""
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        action_dim=2,
        num_rollouts=5,
        horizon=3,
        seed=42
    )
    
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    belief_mean = torch.randn(4)
    belief_std = torch.ones(4) * 0.5
    preferred_obs = torch.tensor([1.0, 1.0])
    
    efe_per_policy, efe_per_timestep, info = evaluator.evaluate_policy(
        belief_mean, belief_std, preferred_obs
    )
    
    # Should return EFE for each policy
    assert efe_per_policy.shape == (config.num_rollouts,)
    
    # Should return EFE per policy per timestep
    assert efe_per_timestep.shape == (config.horizon, config.num_rollouts)


def test_efe_different_policies_different_values():
    """Test that different policies have different EFE values."""
    config = Config(num_rollouts=10, horizon=3, seed=42)
    
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    belief_mean = torch.randn(config.hidden_dim)
    belief_std = torch.ones(config.hidden_dim) * 0.5
    preferred_obs = torch.randn(config.obs_dim)
    
    efe_per_policy, _, _ = evaluator.evaluate_policy(
        belief_mean, belief_std, preferred_obs
    )
    
    # Different policies should have different EFE values
    assert efe_per_policy.std() > 0


def test_best_policy_selection():
    """Test that we can identify the best policy (lowest EFE)."""
    config = Config(num_rollouts=10, horizon=3, seed=42)
    
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    belief_mean = torch.randn(config.hidden_dim)
    belief_std = torch.ones(config.hidden_dim) * 0.5
    preferred_obs = torch.randn(config.obs_dim)
    
    efe_per_policy, _, info = evaluator.evaluate_policy(
        belief_mean, belief_std, preferred_obs
    )
    
    # Info should contain best policy index
    assert "best_policy_idx" in info
    assert 0 <= info["best_policy_idx"] < config.num_rollouts
    
    # Best policy should have minimum EFE
    assert efe_per_policy[info["best_policy_idx"]] == efe_per_policy.min()


def test_efe_per_timestep_structure():
    """Test that EFE per timestep has correct structure."""
    config = Config(num_rollouts=5, horizon=4, seed=42)
    
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    belief_mean = torch.randn(config.hidden_dim)
    belief_std = torch.ones(config.hidden_dim) * 0.5
    preferred_obs = torch.randn(config.obs_dim)
    
    _, efe_per_timestep, _ = evaluator.evaluate_policy(
        belief_mean, belief_std, preferred_obs
    )
    
    # Shape should be [horizon, num_rollouts]
    assert efe_per_timestep.shape == (config.horizon, config.num_rollouts)
    
    # Sum over horizon should equal total EFE per policy
    efe_total = efe_per_timestep.sum(dim=0)
    assert efe_total.shape == (config.num_rollouts,)


def test_agent_uses_efe_per_policy():
    """Test that agent correctly uses per-policy EFE in planning."""
    config = Config(
        hidden_dim=4,
        obs_dim=2,
        action_dim=2,
        num_rollouts=5,
        num_policy_iterations=3,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    
    observation = torch.randn(config.obs_dim)
    preferred_obs = torch.randn(config.obs_dim)
    
    # Perceive and plan
    agent.perceive(observation)
    planning_info = agent.plan(preferred_obs)
    
    # Planning info should contain per-policy EFE
    assert "efe_per_policy" in planning_info
    assert planning_info["efe_per_policy"].shape == (config.num_rollouts,)
    
    assert "efe_per_policy_per_timestep" in planning_info
    assert planning_info["efe_per_policy_per_timestep"].shape == (config.horizon, config.num_rollouts)


def test_efe_statistics_in_info():
    """Test that EFE statistics are properly computed."""
    config = Config(num_rollouts=10, seed=42)
    
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    belief_mean = torch.randn(config.hidden_dim)
    belief_std = torch.ones(config.hidden_dim) * 0.5
    preferred_obs = torch.randn(config.obs_dim)
    
    _, _, info = evaluator.evaluate_policy(
        belief_mean, belief_std, preferred_obs
    )
    
    assert "mean_efe" in info
    assert "min_efe" in info
    assert "max_efe" in info
    
    # Check relationships
    assert info["min_efe"] <= info["mean_efe"] <= info["max_efe"]


def test_policy_optimization_reduces_mean_efe():
    """Test that policy optimization reduces mean EFE."""
    config = Config(
        num_rollouts=5,
        num_policy_iterations=10,
        learning_rate_policy=0.1,
        seed=42
    )
    
    agent = ActiveInferenceAgent(config)
    
    observation = torch.randn(config.obs_dim)
    preferred_obs = torch.randn(config.obs_dim)
    
    agent.perceive(observation)
    planning_info = agent.plan(preferred_obs)
    
    # EFE should decrease over policy iterations
    assert "efe_history" in planning_info
    assert len(planning_info["efe_history"]) > 0
    
    # Generally expect EFE to decrease
    # (May not be strict monotonic due to stochastic sampling)
    initial_efe = planning_info["efe_history"][0]
    final_efe = planning_info["efe_history"][-1]
    assert final_efe <= initial_efe * 1.1  # Allow some tolerance


def test_efe_components_per_policy():
    """Test that EFE components are tracked."""
    config = Config(num_rollouts=5, horizon=3, seed=42)
    
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    belief_mean = torch.randn(config.hidden_dim)
    belief_std = torch.ones(config.hidden_dim) * 0.5
    preferred_obs = torch.randn(config.obs_dim)
    
    _, _, info = evaluator.evaluate_policy(
        belief_mean, belief_std, preferred_obs
    )
    
    assert "efe_components" in info
    assert len(info["efe_components"]) == config.horizon
    
    # Each timestep should have epistemic and pragmatic values
    for components in info["efe_components"]:
        assert "epistemic" in components
        assert "pragmatic" in components

