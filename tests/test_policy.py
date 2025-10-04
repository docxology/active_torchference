"""Tests for policy evaluation and selection."""

import pytest
import torch
from active_torchference.config import Config
from active_torchference.policy import TransitionModel, PolicyEvaluator
from active_torchference.beliefs import GenerativeModel
from active_torchference.free_energy import ExpectedFreeEnergy


def test_transition_model_initialization():
    """Test transition model initialization."""
    config = Config(hidden_dim=4, action_dim=2)
    model = TransitionModel(config)
    
    assert isinstance(model, torch.nn.Module)


def test_transition_model_forward():
    """Test transition model forward pass."""
    config = Config(hidden_dim=4, action_dim=2)
    model = TransitionModel(config)
    
    state = torch.randn(4)
    action = torch.randn(2)
    next_state = model(state, action)
    
    assert next_state.shape == (4,)


def test_transition_model_batch():
    """Test transition model with batch input."""
    config = Config(hidden_dim=4, action_dim=2)
    model = TransitionModel(config)
    
    states = torch.randn(10, 4)
    actions = torch.randn(10, 2)
    next_states = model(states, actions)
    
    assert next_states.shape == (10, 4)


def test_policy_evaluator_initialization():
    """Test policy evaluator initialization."""
    config = Config()
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    assert evaluator.action_mean.shape == (config.action_dim,)
    assert evaluator.action_log_std.shape == (config.action_dim,)


def test_policy_evaluator_sample_actions():
    """Test action sampling."""
    config = Config(action_dim=2, seed=42)
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    actions = evaluator.sample_actions(num_samples=10)
    
    assert actions.shape == (10, 2)


def test_policy_evaluator_rollout():
    """Test policy rollout."""
    config = Config(hidden_dim=4, action_dim=2, horizon=3)
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    initial_state = torch.randn(4)
    actions = torch.randn(5, 3, 2)  # 5 rollouts, 3 timesteps, 2 action dims
    
    states, observations = evaluator.rollout(initial_state, actions)
    
    assert len(states) == 3
    assert len(observations) == 3
    assert observations[0].shape == (5, config.obs_dim)


def test_policy_evaluator_evaluate():
    """Test policy evaluation."""
    config = Config(hidden_dim=4, obs_dim=2, action_dim=2)
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
    
    assert efe_per_policy.shape == (config.num_rollouts,)
    assert efe_per_timestep.shape == (config.horizon, config.num_rollouts)
    assert "action_samples" in info
    assert "efe_per_policy" in info


def test_policy_evaluator_update():
    """Test policy parameter update."""
    config = Config()
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    initial_mean = evaluator.action_mean.clone()
    
    grad_mean = torch.ones_like(evaluator.action_mean)
    grad_log_std = torch.zeros_like(evaluator.action_log_std)
    
    evaluator.update_policy((grad_mean, grad_log_std), learning_rate=0.1)
    
    expected_mean = initial_mean - 0.1 * grad_mean
    assert torch.allclose(evaluator.action_mean, expected_mean)


def test_policy_evaluator_select_action():
    """Test action selection."""
    config = Config(action_dim=2, seed=42)
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    # Deterministic
    action_det = evaluator.select_action(deterministic=True)
    assert torch.allclose(action_det, evaluator.action_mean)
    
    # Stochastic
    action_stoch = evaluator.select_action(deterministic=False)
    assert action_stoch.shape == (2,)


def test_policy_evaluator_reset():
    """Test policy reset."""
    config = Config()
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    # Modify policy
    with torch.no_grad():
        evaluator.action_mean.fill_(5.0)
    
    evaluator.reset()
    
    assert torch.allclose(evaluator.action_mean, torch.zeros_like(evaluator.action_mean))


def test_policy_evaluator_to_dict():
    """Test policy serialization."""
    config = Config()
    transition_model = TransitionModel(config)
    generative_model = GenerativeModel(config)
    efe_calculator = ExpectedFreeEnergy()
    
    evaluator = PolicyEvaluator(
        config, transition_model, generative_model, efe_calculator
    )
    
    policy_dict = evaluator.to_dict()
    
    assert "action_mean" in policy_dict
    assert "action_log_std" in policy_dict

