"""Free energy computations for Active Inference."""

import torch
from typing import Tuple
from active_torchference.utils import gaussian_log_likelihood, kl_divergence_gaussian, entropy_categorical


class VariationalFreeEnergy:
    """
    Computes variational free energy for belief updating.
    
    VFE = E_q[log q(s) - log p(o,s)] = -E_q[log p(o|s)] + KL[q(s)||p(s)]
    
    Where:
        - o: observations
        - s: hidden states
        - q(s): variational posterior (current beliefs)
        - p(o|s): likelihood (generative model)
        - p(s): prior over states
    """
    
    def __init__(
        self,
        precision_obs: float = 1.0,
        precision_prior: float = 1.0
    ):
        """
        Initialize variational free energy calculator.
        
        Args:
            precision_obs: Precision of observation likelihood.
            precision_prior: Precision of prior beliefs.
        """
        self.precision_obs = precision_obs
        self.precision_prior = precision_prior
    
    def compute(
        self,
        observation: torch.Tensor,
        belief_mean: torch.Tensor,
        belief_std: torch.Tensor,
        predicted_obs: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute variational free energy.
        
        Args:
            observation: Actual observation from environment.
            belief_mean: Mean of current belief state q(s).
            belief_std: Std deviation of current belief state.
            predicted_obs: Predicted observation from generative model p(o|s).
            prior_mean: Mean of prior p(s).
            prior_std: Std deviation of prior.
        
        Returns:
            Tuple of (total_vfe, components_dict).
        """
        # Likelihood term: -E_q[log p(o|s)]
        # Approximated as -log p(o|mu_q)
        likelihood_term = -gaussian_log_likelihood(
            observation,
            predicted_obs,
            self.precision_obs
        )
        
        # Complexity term: KL[q(s)||p(s)]
        complexity_term = kl_divergence_gaussian(
            belief_mean,
            belief_std,
            prior_mean,
            prior_std
        )
        
        # Total variational free energy
        vfe = likelihood_term + complexity_term
        
        components = {
            "likelihood": likelihood_term.detach(),
            "complexity": complexity_term.detach(),
            "vfe": vfe.detach()
        }
        
        return vfe, components


class ExpectedFreeEnergy:
    """
    Computes expected free energy for policy evaluation.
    
    EFE = Epistemic Value + Pragmatic Value
    
    Epistemic Value (information gain): -E_q[H[p(o|s)] - H[p(o|s,π)]]
    Pragmatic Value (goal-directed): -E_q[log p(o|C)]
    
    Where:
        - π: policy (action sequence)
        - C: preferences (goal states)
        - H[·]: entropy
    """
    
    def __init__(
        self,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 1.0,
        precision_obs: float = 1.0
    ):
        """
        Initialize expected free energy calculator.
        
        Args:
            epistemic_weight: Weight for epistemic value (exploration).
            pragmatic_weight: Weight for pragmatic value (exploitation).
            precision_obs: Precision of observation likelihood.
        """
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight
        self.precision_obs = precision_obs
    
    def compute(
        self,
        predicted_obs: torch.Tensor,
        predicted_obs_std: torch.Tensor,
        preferred_obs: torch.Tensor,
        state_entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute expected free energy for a policy.
        
        Args:
            predicted_obs: Predicted observations under policy.
            predicted_obs_std: Uncertainty in predicted observations.
            preferred_obs: Goal/preferred observations.
            state_entropy: Entropy of predicted state distribution.
        
        Returns:
            Tuple of (total_efe, components_dict).
        """
        # Epistemic value: information gain (reduction in uncertainty)
        # Approximated as negative state entropy
        epistemic_value = -state_entropy
        
        # Pragmatic value: goal-directed behavior
        # Approximated as negative prediction error relative to preferences
        pragmatic_value = gaussian_log_likelihood(
            preferred_obs,
            predicted_obs,
            self.precision_obs
        )
        
        # Total expected free energy (lower is better for policy selection)
        efe = -(
            self.epistemic_weight * epistemic_value +
            self.pragmatic_weight * pragmatic_value
        )
        
        components = {
            "epistemic": epistemic_value.detach(),
            "pragmatic": pragmatic_value.detach(),
            "efe": efe.detach()
        }
        
        return efe, components
    
    def compute_categorical(
        self,
        predicted_obs_probs: torch.Tensor,
        preferred_obs_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute expected free energy for categorical observations.
        
        Args:
            predicted_obs_probs: Predicted observation probabilities.
            preferred_obs_probs: Preferred observation probabilities.
        
        Returns:
            Tuple of (total_efe, components_dict).
        """
        # Epistemic value: expected information gain
        epistemic_value = entropy_categorical(predicted_obs_probs)
        
        # Pragmatic value: expected log preference
        eps = 1e-8
        pragmatic_value = torch.sum(
            predicted_obs_probs * torch.log(preferred_obs_probs + eps),
            dim=-1
        )
        
        # Total expected free energy
        efe = -(
            self.epistemic_weight * epistemic_value +
            self.pragmatic_weight * pragmatic_value
        )
        
        components = {
            "epistemic": epistemic_value.detach(),
            "pragmatic": pragmatic_value.detach(),
            "efe": efe.detach()
        }
        
        return efe, components

