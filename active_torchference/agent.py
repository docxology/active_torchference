"""Active Inference agent with action-perception loop."""

import torch
from typing import Dict, Optional, Tuple
from active_torchference.config import Config
from active_torchference.beliefs import BeliefState, GenerativeModel
from active_torchference.policy import PolicyEvaluator, TransitionModel
from active_torchference.free_energy import VariationalFreeEnergy, ExpectedFreeEnergy


class ActiveInferenceAgent:
    """
    Active Inference agent implementing perception-action loop.
    
    Action-Perception Loop:
    1. Receive new observation from environment
    2. Update beliefs by minimizing variational free energy
    3. Evaluate policies using expected free energy
    4. Select action from updated policy posterior
    """
    
    def __init__(self, config: Config):
        """
        Initialize Active Inference agent.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        
        # Initialize belief state
        self.beliefs = BeliefState(config)
        
        # Initialize models
        self.generative_model = GenerativeModel(config)
        self.transition_model = TransitionModel(config)
        
        # Initialize free energy calculators
        self.vfe_calculator = VariationalFreeEnergy(
            precision_obs=config.precision_obs,
            precision_prior=config.precision_prior
        )
        
        self.efe_calculator = ExpectedFreeEnergy(
            epistemic_weight=config.epistemic_weight,
            pragmatic_weight=config.pragmatic_weight,
            precision_obs=config.precision_obs
        )
        
        # Initialize policy evaluator
        self.policy_evaluator = PolicyEvaluator(
            config=config,
            transition_model=self.transition_model,
            generative_model=self.generative_model,
            efe_calculator=self.efe_calculator
        )
        
        # History tracking
        self.history = {
            "observations": [],
            "actions": [],
            "beliefs": [],
            "belief_stds": [],  # Track belief uncertainty
            "vfe": [],
            "efe": [],  # Mean EFE for backward compatibility
            "efe_per_policy": [],  # EFE for each action sample [num_rollouts]
            "efe_per_policy_per_timestep": [],  # Full breakdown [horizon, num_rollouts]
            "efe_components": [],  # Epistemic/pragmatic breakdown
            "action_samples": [],  # Candidate actions considered
            "best_policy_idx": [],  # Which action was selected
        }
    
    def perceive(
        self,
        observation: torch.Tensor,
        num_iterations: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Step 1: Process new observation and update beliefs via VFE minimization.
        
        Iteratively minimizes VFE to update belief state distribution.
        
        Args:
            observation: New observation from environment.
            num_iterations: Number of VFE minimization iterations (uses config default if None).
        
        Returns:
            Dictionary with perception info.
        """
        observation = observation.to(self.config.device, self.config.dtype)
        
        if num_iterations is None:
            num_iterations = getattr(self.config, 'num_belief_iterations', 5)
        
        vfe_history = []
        
        # Iteratively minimize VFE to update beliefs
        for iteration in range(num_iterations):
            # Ensure gradients enabled
            if not self.beliefs.mean.requires_grad:
                self.beliefs.mean.requires_grad_(True)
            if not self.beliefs.log_std.requires_grad:
                self.beliefs.log_std.requires_grad_(True)
            
            # Predict observation from current beliefs
            predicted_obs = self.generative_model(self.beliefs.mean)
            
            # Compute variational free energy
            vfe, vfe_components = self.vfe_calculator.compute(
                observation=observation,
                belief_mean=self.beliefs.mean,
                belief_std=self.beliefs.std,
                predicted_obs=predicted_obs,
                prior_mean=self.beliefs.prior_mean,
                prior_std=self.beliefs.prior_std
            )
            
            vfe_history.append(vfe.item())
            
            # Minimize VFE via gradient descent on beliefs
            vfe.backward()
            
            # Update beliefs
            if self.beliefs.mean.grad is not None and self.beliefs.log_std.grad is not None:
                belief_grad = (
                    self.beliefs.mean.grad.clone(),
                    self.beliefs.log_std.grad.clone()
                )
                
                self.beliefs.update(belief_grad)
                
                # Clear gradients
                self.beliefs.mean.grad.zero_()
                self.beliefs.log_std.grad.zero_()
        
        # Get final predictions after belief update
        with torch.no_grad():
            final_predicted_obs = self.generative_model(self.beliefs.mean)
        
        perception_info = {
            "vfe": vfe.detach(),
            "vfe_initial": vfe_history[0],
            "vfe_final": vfe_history[-1],
            "vfe_history": vfe_history,
            "vfe_likelihood": vfe_components["likelihood"],
            "vfe_complexity": vfe_components["complexity"],
            "predicted_obs": final_predicted_obs,
            "belief_mean": self.beliefs.mean.detach().clone(),
            "belief_std": self.beliefs.std.detach().clone()
        }
        
        return perception_info
    
    def plan(
        self,
        preferred_obs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Step 2: Evaluate policies using expected free energy.
        
        Evaluates multiple policies and selects the best one based on EFE.
        
        Args:
            preferred_obs: Goal/preferred observations.
        
        Returns:
            Dictionary with planning info.
        """
        preferred_obs = preferred_obs.to(self.config.device, self.config.dtype)
        
        # Ensure policy parameters require gradients
        if not self.policy_evaluator.action_mean.requires_grad:
            self.policy_evaluator.action_mean.requires_grad_(True)
        if not self.policy_evaluator.action_log_std.requires_grad:
            self.policy_evaluator.action_log_std.requires_grad_(True)
        
        # Optimize policy over multiple iterations
        planning_info = {}
        efe_history = []
        
        for iteration in range(self.config.num_policy_iterations):
            # Evaluate multiple policies with current beliefs
            efe_per_policy, efe_per_timestep, eval_info = self.policy_evaluator.evaluate_policy(
                current_belief_mean=self.beliefs.mean.detach(),
                current_belief_std=self.beliefs.std.detach(),
                preferred_obs=preferred_obs
            )
            
            # Use mean EFE for policy optimization
            mean_efe = efe_per_policy.mean()
            efe_history.append(mean_efe.item())
            
            # Minimize EFE via gradient descent on policy
            mean_efe.backward()
            
            # Update policy
            if self.policy_evaluator.action_mean.grad is not None and \
               self.policy_evaluator.action_log_std.grad is not None:
                policy_grad = (
                    self.policy_evaluator.action_mean.grad.clone(),
                    self.policy_evaluator.action_log_std.grad.clone()
                )
                
                self.policy_evaluator.update_policy(policy_grad)
                
                # Clear gradients
                self.policy_evaluator.action_mean.grad.zero_()
                self.policy_evaluator.action_log_std.grad.zero_()
            
            # Store final iteration info
            if iteration == self.config.num_policy_iterations - 1:
                planning_info = {
                    "efe": mean_efe.detach(),
                    "efe_per_policy": eval_info["efe_per_policy"],
                    "efe_per_policy_per_timestep": eval_info["efe_per_policy_per_timestep"],
                    "efe_components": eval_info["efe_components"],
                    "efe_history": efe_history,
                    "best_policy_idx": eval_info["best_policy_idx"],
                    "min_efe": eval_info["min_efe"],
                    "max_efe": eval_info["max_efe"],
                    "action_mean": self.policy_evaluator.action_mean.detach().clone(),
                    "action_std": self.policy_evaluator.action_std.detach().clone(),
                    "predicted_obs": eval_info["predicted_observations"],
                    "action_samples": eval_info["action_samples"]
                }
        
        return planning_info
    
    def act(
        self,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Step 3: Select action from updated policy posterior.
        
        Args:
            deterministic: If True, return mean action.
        
        Returns:
            Selected action.
        """
        action = self.policy_evaluator.select_action(deterministic=deterministic)
        return action
    
    def step(
        self,
        observation: torch.Tensor,
        preferred_obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Execute full action-perception loop step.
        
        Args:
            observation: New observation from environment.
            preferred_obs: Goal/preferred observations.
            deterministic: If True, select mean action.
        
        Returns:
            Tuple of (action, info_dict).
        """
        # 1. Perceive: Update beliefs via VFE
        perception_info = self.perceive(observation)
        
        # 2. Plan: Evaluate policies via EFE
        planning_info = self.plan(preferred_obs)
        
        # 3. Act: Select action from policy posterior
        action = self.act(deterministic=deterministic)
        
        # Combine info
        info = {**perception_info, **planning_info, "action": action}
        
        # Update history with comprehensive data (CLONE to avoid aliasing)
        self.history["observations"].append(observation.detach().cpu().clone())
        self.history["actions"].append(action.detach().cpu().clone())
        self.history["beliefs"].append(self.beliefs.mean.detach().cpu().clone())
        self.history["belief_stds"].append(self.beliefs.std.detach().cpu().clone())
        self.history["vfe"].append(perception_info["vfe"].cpu().clone())
        
        # Mean EFE for backward compatibility (CLONE to avoid aliasing)
        self.history["efe"].append(planning_info["efe"].cpu().clone())
        
        # Full EFE breakdown per action (CLONE to avoid aliasing)
        self.history["efe_per_policy"].append(planning_info["efe_per_policy"].cpu().clone())
        self.history["efe_per_policy_per_timestep"].append(
            planning_info["efe_per_policy_per_timestep"].cpu().clone()
        )
        self.history["efe_components"].append(planning_info["efe_components"])
        self.history["action_samples"].append(planning_info["action_samples"].cpu().clone())
        self.history["best_policy_idx"].append(planning_info["best_policy_idx"])
        
        return action, info
    
    def reset(self):
        """Reset agent state."""
        self.beliefs.reset()
        self.policy_evaluator.reset()
        
        # Clear history
        self.history = {
            "observations": [],
            "actions": [],
            "beliefs": [],
            "belief_stds": [],
            "vfe": [],
            "efe": [],
            "efe_per_policy": [],
            "efe_per_policy_per_timestep": [],
            "efe_components": [],
            "action_samples": [],
            "best_policy_idx": [],
        }
    
    def get_history(self) -> Dict[str, list]:
        """Get agent history."""
        return self.history
    
    def save_state(self) -> Dict:
        """
        Save agent state to dictionary.
        
        Returns:
            Dictionary with agent state.
        """
        return {
            "beliefs": self.beliefs.to_dict(),
            "policy": self.policy_evaluator.to_dict(),
            "config": self.config.to_dict()
        }
    
    def load_state(self, state_dict: Dict):
        """
        Load agent state from dictionary.
        
        Args:
            state_dict: Dictionary with agent state.
        """
        # Load beliefs
        beliefs_dict = state_dict["beliefs"]
        self.beliefs.mean = torch.tensor(
            beliefs_dict["mean"],
            device=self.config.device,
            dtype=self.config.dtype,
            requires_grad=True
        )
        self.beliefs.log_std = torch.tensor(
            beliefs_dict["log_std"],
            device=self.config.device,
            dtype=self.config.dtype,
            requires_grad=True
        )
        
        # Load policy
        policy_dict = state_dict["policy"]
        self.policy_evaluator.action_mean = torch.tensor(
            policy_dict["action_mean"],
            device=self.config.device,
            dtype=self.config.dtype,
            requires_grad=True
        )
        self.policy_evaluator.action_log_std = torch.tensor(
            policy_dict["action_log_std"],
            device=self.config.device,
            dtype=self.config.dtype,
            requires_grad=True
        )

