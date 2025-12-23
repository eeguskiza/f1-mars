"""Pilot agent wrapper for F1 Mars simulator."""

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
import numpy as np


class PilotAgent:
    """
    Wrapper for the pilot reinforcement learning agent.

    The pilot agent controls steering, throttle, and braking to navigate
    the track as fast as possible.
    """

    def __init__(
        self,
        env,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        **kwargs
    ):
        """
        Initialize the pilot agent.

        Args:
            env: Gymnasium environment
            algorithm: RL algorithm to use ("PPO" or "SAC")
            policy: Policy network type
            **kwargs: Additional arguments for the algorithm
        """
        self.env = env
        self.algorithm_name = algorithm

        # Default hyperparameters
        default_params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 1,
        }

        # Merge with user-provided kwargs
        params = {**default_params, **kwargs}

        # Initialize algorithm
        if algorithm == "PPO":
            self.model = PPO(policy, env, **params)
        elif algorithm == "SAC":
            self.model = SAC(policy, env, **params)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10
    ):
        """
        Train the pilot agent.

        Args:
            total_timesteps: Number of environment steps to train
            callback: Optional training callback
            log_interval: How often to log training progress
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=True
        )

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Predict action for given observation.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy

        Returns:
            Action array [steering, throttle, brake]
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """
        Save the trained model.

        Args:
            path: File path to save model
        """
        self.model.save(path)
        print(f"Pilot agent saved to {path}")

    def load(self, path: str):
        """
        Load a trained model.

        Args:
            path: File path to load model from
        """
        if self.algorithm_name == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm_name == "SAC":
            self.model = SAC.load(path, env=self.env)

        print(f"Pilot agent loaded from {path}")

    @staticmethod
    def create_from_checkpoint(path: str, env):
        """
        Create a pilot agent from a saved checkpoint.

        Args:
            path: Path to saved model
            env: Environment to use

        Returns:
            PilotAgent instance with loaded model
        """
        # Detect algorithm from file
        # This is a simplification - in practice, you'd save metadata
        agent = PilotAgent(env)
        agent.load(path)
        return agent


class PilotTrainingCallback(BaseCallback):
    """
    Custom callback for monitoring pilot agent training.
    """

    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        """
        Initialize callback.

        Args:
            save_freq: Save model every N steps
            save_path: Directory to save checkpoints
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Returns:
            True to continue training
        """
        # Save checkpoint periodically
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = f"{self.save_path}/pilot_checkpoint_{self.n_calls}.zip"
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Checkpoint saved: {checkpoint_path}")

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        # Track best model based on mean reward
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = f"{self.save_path}/pilot_best.zip"
                self.model.save(best_path)
                if self.verbose > 0:
                    print(f"New best model saved with reward: {mean_reward:.2f}")
