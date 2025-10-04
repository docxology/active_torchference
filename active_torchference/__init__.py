"""
Active Torchference: PyTorch-based Active Inference framework.

Implements the action-perception loop with variational and expected free energy.
"""

__version__ = "0.1.0"

from active_torchference.agent import ActiveInferenceAgent
from active_torchference.environment import (
    Environment,
    ContinuousEnvironment,
    GridWorld,
    OscillatorEnvironment
)
from active_torchference.free_energy import VariationalFreeEnergy, ExpectedFreeEnergy
from active_torchference.beliefs import BeliefState
from active_torchference.policy import PolicyEvaluator
from active_torchference.config import Config
from active_torchference.output_manager import OutputManager

__all__ = [
    "ActiveInferenceAgent",
    "Environment",
    "ContinuousEnvironment",
    "GridWorld",
    "OscillatorEnvironment",
    "VariationalFreeEnergy",
    "ExpectedFreeEnergy",
    "BeliefState",
    "PolicyEvaluator",
    "Config",
    "OutputManager",
]

