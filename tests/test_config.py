"""Tests for configuration system."""

import pytest
import torch
from active_torchference.config import Config


def test_config_initialization():
    """Test basic configuration initialization."""
    config = Config()
    assert config.hidden_dim == 4
    assert config.obs_dim == 2
    assert config.action_dim == 2
    assert isinstance(config.device, torch.device)


def test_config_custom_parameters():
    """Test configuration with custom parameters."""
    config = Config(
        hidden_dim=8,
        obs_dim=4,
        learning_rate_beliefs=0.05,
        seed=42
    )
    assert config.hidden_dim == 8
    assert config.obs_dim == 4
    assert config.learning_rate_beliefs == 0.05
    assert config.seed == 42


def test_config_to_dict():
    """Test configuration serialization to dictionary."""
    config = Config(hidden_dim=6, seed=123)
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict["hidden_dim"] == 6
    assert config_dict["seed"] == 123


def test_config_from_dict():
    """Test configuration deserialization from dictionary."""
    config_dict = {
        "hidden_dim": 10,
        "obs_dim": 5,
        "learning_rate_beliefs": 0.2,
        "seed": 999
    }
    config = Config.from_dict(config_dict)
    
    assert config.hidden_dim == 10
    assert config.obs_dim == 5
    assert config.learning_rate_beliefs == 0.2


def test_config_seed_reproducibility():
    """Test that seed ensures reproducibility."""
    config1 = Config(seed=42)
    tensor1 = torch.rand(10)
    
    config2 = Config(seed=42)
    tensor2 = torch.rand(10)
    
    assert torch.allclose(tensor1, tensor2)

