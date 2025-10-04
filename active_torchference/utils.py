"""Utility functions for Active Inference computations."""

import torch
from typing import Optional


def ensure_tensor(
    x: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Ensure input is a tensor with correct device and dtype.
    
    Args:
        x: Input tensor or array-like.
        device: Target device.
        dtype: Target dtype.
    
    Returns:
        Tensor with correct device and dtype.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    if device is not None:
        x = x.to(device)
    
    if dtype is not None:
        x = x.to(dtype)
    
    return x


def gaussian_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    precision: float
) -> torch.Tensor:
    """
    Compute Gaussian log-likelihood with precision parameterization.
    
    Args:
        x: Observations.
        mu: Mean predictions.
        precision: Precision (inverse variance).
    
    Returns:
        Log-likelihood values.
    """
    squared_error = torch.sum((x - mu) ** 2, dim=-1)
    log_likelihood = -0.5 * precision * squared_error
    return log_likelihood


def softmax(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Temperature-scaled softmax for action selection.
    
    Args:
        x: Input logits (negative free energies).
        temperature: Temperature parameter (lower = more deterministic).
    
    Returns:
        Probability distribution.
    """
    x_scaled = x / temperature
    x_shifted = x_scaled - torch.max(x_scaled, dim=-1, keepdim=True)[0]
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


def kl_divergence_gaussian(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between two Gaussian distributions.
    
    KL(q||p) = 0.5 * [log(sigma_p^2/sigma_q^2) + (sigma_q^2 + (mu_q - mu_p)^2)/sigma_p^2 - 1]
    
    Args:
        mu_q: Mean of q distribution.
        sigma_q: Std deviation of q distribution.
        mu_p: Mean of p distribution.
        sigma_p: Std deviation of p distribution.
    
    Returns:
        KL divergence value.
    """
    kl = 0.5 * (
        torch.log(sigma_p ** 2 / sigma_q ** 2)
        + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / sigma_p ** 2
        - 1.0
    )
    return torch.sum(kl, dim=-1)


def entropy_categorical(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute entropy of categorical distribution.
    
    Args:
        probs: Probability distribution.
        eps: Small constant for numerical stability.
    
    Returns:
        Entropy value.
    """
    log_probs = torch.log(probs + eps)
    return -torch.sum(probs * log_probs, dim=-1)

