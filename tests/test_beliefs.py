"""Tests for belief state management."""

import pytest
import torch
from active_torchference.config import Config
from active_torchference.beliefs import BeliefState, GenerativeModel


def test_belief_state_initialization():
    """Test belief state initialization."""
    config = Config(hidden_dim=4)
    beliefs = BeliefState(config)
    
    assert beliefs.mean.shape == (4,)
    assert beliefs.log_std.shape == (4,)
    assert beliefs.mean.requires_grad
    assert beliefs.log_std.requires_grad


def test_belief_state_std_property():
    """Test standard deviation property."""
    config = Config()
    beliefs = BeliefState(config)
    
    with torch.no_grad():
        beliefs.log_std.fill_(0.0)  # log(1) = 0
    
    assert torch.allclose(beliefs.std, torch.ones_like(beliefs.std))


def test_belief_state_sample():
    """Test sampling from belief distribution."""
    config = Config(hidden_dim=4, seed=42)
    beliefs = BeliefState(config)
    
    samples = beliefs.sample(num_samples=10)
    assert samples.shape == (10, 4)


def test_belief_state_update():
    """Test belief state update."""
    config = Config()
    beliefs = BeliefState(config)
    
    initial_mean = beliefs.mean.clone()
    
    grad_mean = torch.ones_like(beliefs.mean)
    grad_log_std = torch.zeros_like(beliefs.log_std)
    
    beliefs.update((grad_mean, grad_log_std), learning_rate=0.1)
    
    # Mean should have decreased by learning_rate * gradient
    expected_mean = initial_mean - 0.1 * grad_mean
    assert torch.allclose(beliefs.mean, expected_mean)


def test_belief_state_reset():
    """Test belief state reset."""
    config = Config()
    beliefs = BeliefState(config)
    
    # Modify belief
    with torch.no_grad():
        beliefs.mean.fill_(5.0)
    
    # Reset
    beliefs.reset()
    
    # Should match prior
    assert torch.allclose(beliefs.mean, beliefs.prior_mean)


def test_belief_state_set_prior():
    """Test setting prior distribution."""
    config = Config()
    beliefs = BeliefState(config)
    
    new_prior_mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    new_prior_std = torch.tensor([0.5, 0.5, 0.5, 0.5])
    
    beliefs.set_prior(new_prior_mean, new_prior_std)
    
    assert torch.allclose(beliefs.prior_mean, new_prior_mean)
    assert torch.allclose(beliefs.prior_std, new_prior_std)


def test_belief_state_detach():
    """Test detaching belief state."""
    config = Config()
    beliefs = BeliefState(config)
    
    mean, std = beliefs.detach()
    
    assert not mean.requires_grad
    assert not std.requires_grad
    assert mean.shape == beliefs.mean.shape


def test_belief_state_to_dict():
    """Test belief state serialization."""
    config = Config()
    beliefs = BeliefState(config)
    
    state_dict = beliefs.to_dict()
    
    assert "mean" in state_dict
    assert "log_std" in state_dict
    assert "prior_mean" in state_dict
    assert "prior_log_std" in state_dict


def test_generative_model_initialization():
    """Test generative model initialization."""
    config = Config(hidden_dim=4, obs_dim=2)
    model = GenerativeModel(config)
    
    assert isinstance(model, torch.nn.Module)


def test_generative_model_forward():
    """Test generative model forward pass."""
    config = Config(hidden_dim=4, obs_dim=2)
    model = GenerativeModel(config)
    
    hidden_state = torch.randn(4)
    predicted_obs = model(hidden_state)
    
    assert predicted_obs.shape == (2,)


def test_generative_model_batch():
    """Test generative model with batch input."""
    config = Config(hidden_dim=4, obs_dim=2)
    model = GenerativeModel(config)
    
    hidden_states = torch.randn(10, 4)
    predicted_obs = model(hidden_states)
    
    assert predicted_obs.shape == (10, 2)


def test_generative_model_uncertainty_propagation():
    """Test observation prediction with uncertainty."""
    config = Config(hidden_dim=4, obs_dim=2, seed=42)
    model = GenerativeModel(config)
    
    belief_mean = torch.randn(4)
    belief_std = torch.ones(4) * 0.5
    
    obs_mean, obs_std = model.predict_with_uncertainty(
        belief_mean, belief_std, num_samples=100
    )
    
    assert obs_mean.shape == (2,)
    assert obs_std.shape == (2,)
    assert torch.all(obs_std > 0)

