"""Experiment runner for Active Inference."""

from typing import Optional, Callable, Dict, Any
import torch
from tqdm import tqdm

from active_torchference.config import Config
from active_torchference.agent import ActiveInferenceAgent
from active_torchference.environment import Environment
from active_torchference.orchestrators.logger import Logger


class ExperimentRunner:
    """
    Orchestrates Active Inference experiments.
    
    Manages agent-environment interaction loop with logging and callbacks.
    """
    
    def __init__(
        self,
        agent: ActiveInferenceAgent,
        environment: Environment,
        logger: Optional[Logger] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            agent: Active Inference agent.
            environment: Environment for agent interaction.
            logger: Logger for experiment data (created if None).
            config: Configuration object.
        """
        self.agent = agent
        self.environment = environment
        self.config = config or agent.config
        self.logger = logger or Logger()
        
        # Callbacks
        self.step_callbacks = []
        self.episode_callbacks = []
    
    def add_step_callback(self, callback: Callable):
        """
        Add callback function called after each step.
        
        Args:
            callback: Function with signature (step, agent_info, env_info).
        """
        self.step_callbacks.append(callback)
    
    def add_episode_callback(self, callback: Callable):
        """
        Add callback function called after each episode.
        
        Args:
            callback: Function with signature (episode, episode_data).
        """
        self.episode_callbacks.append(callback)
    
    def run_episode(
        self,
        max_steps: int = 100,
        deterministic: bool = False,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Run single episode.
        
        Args:
            max_steps: Maximum steps per episode.
            deterministic: Use deterministic action selection.
            render: Render environment.
        
        Returns:
            Episode summary data.
        """
        # Reset
        observation = self.environment.reset()
        self.agent.reset()
        
        preferred_obs = self.environment.get_preferred_observation()
        
        episode_data = {
            "steps": 0,
            "total_vfe": 0.0,
            "total_efe": 0.0,
            "done": False,
            "observations": [],
            "actions": [],
            "beliefs": []
        }
        
        for step in range(max_steps):
            # Agent step
            action, agent_info = self.agent.step(
                observation,
                preferred_obs,
                deterministic=deterministic
            )
            
            # Environment step
            observation, env_info = self.environment.step(action)
            
            # Store data
            episode_data["observations"].append(observation.detach().cpu())
            episode_data["actions"].append(action.detach().cpu())
            episode_data["beliefs"].append(agent_info["belief_mean"].cpu())
            
            episode_data["total_vfe"] += agent_info["vfe"].item()
            episode_data["total_efe"] += agent_info["efe"].item()
            episode_data["steps"] = step + 1
            
            # Logging
            self.logger.log_step(step, agent_info, env_info)
            
            # Callbacks
            for callback in self.step_callbacks:
                callback(step, agent_info, env_info)
            
            # Render
            if render:
                self.environment.render()
            
            # Check termination
            if env_info.get("done", False):
                episode_data["done"] = True
                break
        
        # Average metrics
        episode_data["avg_vfe"] = episode_data["total_vfe"] / episode_data["steps"]
        episode_data["avg_efe"] = episode_data["total_efe"] / episode_data["steps"]
        
        return episode_data
    
    def run(
        self,
        num_episodes: int = 10,
        max_steps_per_episode: int = 100,
        deterministic: bool = False,
        render: bool = False,
        save_every: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run.
            max_steps_per_episode: Maximum steps per episode.
            deterministic: Use deterministic action selection.
            render: Render environment.
            save_every: Save checkpoint every N episodes.
            verbose: Show progress bar.
        
        Returns:
            Experiment results summary.
        """
        # Save config
        self.logger.save_config(self.config)
        
        results = {
            "episodes": [],
            "config": self.config.to_dict()
        }
        
        episode_iterator = range(num_episodes)
        if verbose:
            episode_iterator = tqdm(episode_iterator, desc="Episodes")
        
        for episode in episode_iterator:
            # Run episode
            episode_data = self.run_episode(
                max_steps=max_steps_per_episode,
                deterministic=deterministic,
                render=render
            )
            
            results["episodes"].append(episode_data)
            
            # Log episode
            self.logger.log_episode(episode, episode_data)
            
            # Callbacks
            for callback in self.episode_callbacks:
                callback(episode, episode_data)
            
            # Save checkpoint
            if (episode + 1) % save_every == 0:
                self.logger.save_agent_state(
                    self.agent,
                    checkpoint_name=f"episode_{episode+1}"
                )
            
            # Update progress bar
            if verbose:
                episode_iterator.set_postfix({
                    "steps": episode_data["steps"],
                    "vfe": f"{episode_data['avg_vfe']:.3f}",
                    "efe": f"{episode_data['avg_efe']:.3f}"
                })
        
        # Save final state and metrics
        self.logger.save_agent_state(self.agent, checkpoint_name="final")
        self.logger.save_metrics()
        
        # Print summary
        if verbose:
            self.logger.print_summary()
        
        return results
    
    def evaluate(
        self,
        num_episodes: int = 10,
        max_steps_per_episode: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent with deterministic policy.
        
        Args:
            num_episodes: Number of evaluation episodes.
            max_steps_per_episode: Maximum steps per episode.
            verbose: Show progress.
        
        Returns:
            Evaluation results.
        """
        return self.run(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            deterministic=True,
            render=False,
            save_every=num_episodes + 1,  # Don't save during eval
            verbose=verbose
        )

