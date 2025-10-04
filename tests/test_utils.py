"""Tests for utility functions."""

import pytest
import torch
from active_torchference.utils import (
    ensure_tensor,
    gaussian_log_likelihood,
    softmax,
    kl_divergence_gaussian,
    entropy_categorical
)


def test_ensure_tensor():
    """Test tensor conversion utility."""
    x = [1.0, 2.0, 3.0]
    tensor = ensure_tensor(x)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3,)


def test_ensure_tensor_device_dtype():
    """Test tensor conversion with device and dtype."""
    x = torch.tensor([1.0, 2.0])
    tensor = ensure_tensor(x, dtype=torch.float64)
    assert tensor.dtype == torch.float64


def test_gaussian_log_likelihood():
    """Test Gaussian log-likelihood computation."""
    x = torch.tensor([1.0, 2.0])
    mu = torch.tensor([1.1, 2.1])
    precision = 1.0
    
    log_like = gaussian_log_likelihood(x, mu, precision)
    assert log_like.shape == ()
    assert log_like < 0  # Negative due to prediction error


def test_gaussian_log_likelihood_perfect():
    """Test log-likelihood with perfect prediction."""
    x = torch.tensor([1.0, 2.0])
    mu = torch.tensor([1.0, 2.0])
    precision = 1.0
    
    log_like = gaussian_log_likelihood(x, mu, precision)
    assert torch.isclose(log_like, torch.tensor(0.0))


def test_softmax():
    """Test temperature-scaled softmax."""
    x = torch.tensor([1.0, 2.0, 3.0])
    probs = softmax(x, temperature=1.0)
    
    assert probs.shape == x.shape
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)


def test_softmax_temperature():
    """Test softmax with different temperatures."""
    x = torch.tensor([1.0, 2.0, 3.0])
    
    probs_low = softmax(x, temperature=0.1)  # More deterministic
    probs_high = softmax(x, temperature=10.0)  # More uniform
    
    # Low temperature should be more peaked
    assert probs_low.max() > probs_high.max()


def test_kl_divergence_gaussian():
    """Test KL divergence between Gaussians."""
    mu_q = torch.tensor([0.0, 0.0])
    sigma_q = torch.tensor([1.0, 1.0])
    mu_p = torch.tensor([0.0, 0.0])
    sigma_p = torch.tensor([1.0, 1.0])
    
    kl = kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    
    # KL divergence should be zero for identical distributions
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-5)


def test_kl_divergence_different_distributions():
    """Test KL divergence for different distributions."""
    mu_q = torch.tensor([1.0])
    sigma_q = torch.tensor([1.0])
    mu_p = torch.tensor([0.0])
    sigma_p = torch.tensor([1.0])
    
    kl = kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    
    # KL should be positive for different distributions
    assert kl > 0


def test_entropy_categorical():
    """Test categorical entropy."""
    # Uniform distribution has maximum entropy
    probs_uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
    entropy_uniform = entropy_categorical(probs_uniform)
    
    # Deterministic distribution has zero entropy
    probs_det = torch.tensor([1.0, 0.0, 0.0, 0.0])
    entropy_det = entropy_categorical(probs_det)
    
    assert entropy_uniform > entropy_det
    assert torch.isclose(entropy_det, torch.tensor(0.0), atol=1e-5)

