"""Tests for free energy computations."""

import pytest
import torch
from active_torchference.free_energy import VariationalFreeEnergy, ExpectedFreeEnergy


def test_vfe_initialization():
    """Test VFE calculator initialization."""
    vfe = VariationalFreeEnergy(precision_obs=1.0, precision_prior=1.0)
    assert vfe.precision_obs == 1.0
    assert vfe.precision_prior == 1.0


def test_vfe_compute():
    """Test VFE computation."""
    vfe = VariationalFreeEnergy()
    
    observation = torch.tensor([1.0, 2.0])
    belief_mean = torch.tensor([1.1, 2.1])
    belief_std = torch.tensor([0.5, 0.5])
    predicted_obs = torch.tensor([1.1, 2.1])
    prior_mean = torch.tensor([0.0, 0.0])
    prior_std = torch.tensor([1.0, 1.0])
    
    total_vfe, components = vfe.compute(
        observation, belief_mean, belief_std,
        predicted_obs, prior_mean, prior_std
    )
    
    assert total_vfe.shape == ()
    assert total_vfe >= 0  # VFE should be non-negative
    assert "likelihood" in components
    assert "complexity" in components
    assert "vfe" in components


def test_vfe_perfect_prediction():
    """Test VFE with perfect prediction and prior match."""
    vfe = VariationalFreeEnergy()
    
    observation = torch.tensor([1.0, 2.0])
    belief_mean = torch.tensor([1.0, 2.0])
    belief_std = torch.tensor([1.0, 1.0])
    predicted_obs = torch.tensor([1.0, 2.0])
    prior_mean = torch.tensor([1.0, 2.0])
    prior_std = torch.tensor([1.0, 1.0])
    
    total_vfe, components = vfe.compute(
        observation, belief_mean, belief_std,
        predicted_obs, prior_mean, prior_std
    )
    
    # VFE should be near zero with perfect prediction and prior match
    assert torch.isclose(total_vfe, torch.tensor(0.0), atol=1e-4)


def test_efe_initialization():
    """Test EFE calculator initialization."""
    efe = ExpectedFreeEnergy(
        epistemic_weight=1.0,
        pragmatic_weight=1.0,
        precision_obs=1.0
    )
    assert efe.epistemic_weight == 1.0
    assert efe.pragmatic_weight == 1.0


def test_efe_compute():
    """Test EFE computation."""
    efe = ExpectedFreeEnergy()
    
    predicted_obs = torch.tensor([1.0, 2.0])
    predicted_obs_std = torch.tensor([0.1, 0.1])
    preferred_obs = torch.tensor([1.5, 2.5])
    state_entropy = torch.tensor(0.5)
    
    total_efe, components = efe.compute(
        predicted_obs, predicted_obs_std,
        preferred_obs, state_entropy
    )
    
    assert total_efe.shape == ()
    assert "epistemic" in components
    assert "pragmatic" in components
    assert "efe" in components


def test_efe_goal_matching():
    """Test EFE with goal-matching predictions."""
    efe = ExpectedFreeEnergy(epistemic_weight=0.0, pragmatic_weight=1.0)
    
    # Perfect goal match
    predicted_obs = torch.tensor([1.0, 2.0])
    preferred_obs = torch.tensor([1.0, 2.0])
    state_entropy = torch.tensor(0.1)
    
    efe_perfect, _ = efe.compute(
        predicted_obs,
        torch.ones_like(predicted_obs) * 0.1,
        preferred_obs,
        state_entropy
    )
    
    # Poor goal match
    predicted_obs_bad = torch.tensor([5.0, 6.0])
    efe_bad, _ = efe.compute(
        predicted_obs_bad,
        torch.ones_like(predicted_obs_bad) * 0.1,
        preferred_obs,
        state_entropy
    )
    
    # Perfect match should have lower EFE
    assert efe_perfect < efe_bad


def test_efe_categorical():
    """Test categorical EFE computation."""
    efe = ExpectedFreeEnergy()
    
    predicted_probs = torch.tensor([0.7, 0.2, 0.1])
    preferred_probs = torch.tensor([0.8, 0.15, 0.05])
    
    total_efe, components = efe.compute_categorical(
        predicted_probs, preferred_probs
    )
    
    assert total_efe.shape == ()
    assert "epistemic" in components
    assert "pragmatic" in components


def test_efe_epistemic_value():
    """Test epistemic value (exploration) in EFE."""
    efe = ExpectedFreeEnergy(epistemic_weight=1.0, pragmatic_weight=0.0)
    
    # High uncertainty
    predicted_probs_uncertain = torch.tensor([0.5, 0.5])
    preferred_probs = torch.tensor([1.0, 0.0])
    
    efe_uncertain, comp_uncertain = efe.compute_categorical(
        predicted_probs_uncertain, preferred_probs
    )
    
    # Low uncertainty
    predicted_probs_certain = torch.tensor([0.99, 0.01])
    efe_certain, comp_certain = efe.compute_categorical(
        predicted_probs_certain, preferred_probs
    )
    
    # Higher uncertainty should have more epistemic value (lower EFE with pure epistemic)
    assert comp_uncertain["epistemic"] > comp_certain["epistemic"]

