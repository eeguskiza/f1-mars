"""Engineer agent wrapper for F1 Mars simulator."""

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
import numpy as np
import time


class EngineerAgent:
    """
    Wrapper for the race engineer reinforcement learning agent.

    The engineer agent makes strategic decisions about tire compounds
    and pit stop timing.
    """

    def __init__(
        self,
        env,
        algorithm: str = "DQN",
        policy: str = "MlpPolicy",
        **kwargs
    ):
        """
        Initialize the engineer agent.

        Args:
            env: Gymnasium environment (should have discrete action space for strategy)
            algorithm: RL algorithm to use (typically "DQN" for discrete actions)
            policy: Policy network type
            **kwargs: Additional arguments for the algorithm
        """
        self.env = env
        self.algorithm_name = algorithm

        # Default hyperparameters for DQN
        default_params = {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "target_update_interval": 1000,
            "verbose": 1,
        }

        # Merge with user-provided kwargs
        params = {**default_params, **kwargs}

        # Initialize algorithm
        if algorithm == "DQN":
            self.model = DQN(policy, env, **params)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 10
    ):
        """
        Train the engineer agent.

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
    ) -> int:
        """
        Predict strategy action for given observation.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy

        Returns:
            Action index (e.g., 0=continue, 1=pit_soft, 2=pit_medium, 3=pit_hard)
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def should_pit(
        self,
        observation: np.ndarray,
        current_lap: int,
        total_laps: int
    ) -> tuple:
        """
        High-level function to determine if car should pit.

        Args:
            observation: Current observation
            current_lap: Current lap number
            total_laps: Total laps in race

        Returns:
            (should_pit: bool, compound: str or None)
        """
        action = self.predict(observation, deterministic=True)

        # Action mapping
        # 0: Continue
        # 1: Pit for soft tires
        # 2: Pit for medium tires
        # 3: Pit for hard tires

        if action == 0:
            return False, None
        elif action == 1:
            return True, "soft"
        elif action == 2:
            return True, "medium"
        elif action == 3:
            return True, "hard"
        else:
            return False, None

    def save(self, path: str):
        """
        Save the trained model.

        Args:
            path: File path to save model
        """
        self.model.save(path)
        print(f"Engineer agent saved to {path}")

    def load(self, path: str):
        """
        Load a trained model.

        Args:
            path: File path to load model from
        """
        self.model = DQN.load(path, env=self.env)
        print(f"Engineer agent loaded from {path}")

    @staticmethod
    def create_from_checkpoint(path: str, env):
        """
        Create an engineer agent from a saved checkpoint.

        Args:
            path: Path to saved model
            env: Environment to use

        Returns:
            EngineerAgent instance with loaded model
        """
        agent = EngineerAgent(env)
        agent.load(path)
        return agent


class EngineerTrainingCallback(BaseCallback):
    """
    Custom callback for monitoring engineer agent training with detailed metrics.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        print_freq: int = 5000,
        verbose: int = 1
    ):
        """
        Initialize callback.

        Args:
            save_freq: Save model every N steps
            save_path: Directory to save checkpoints
            print_freq: Print training stats every N steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.print_freq = print_freq
        self.best_mean_reward = -np.inf

        # Training metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_actions = []  # Track strategy decisions
        self.start_time = None
        self.last_print_step = 0

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        self.start_time = time.time()
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"  STARTING ENGINEER TRAINING")
            print(f"{'='*70}\n")

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Returns:
            True to continue training
        """
        # Collect episode metrics
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

                # Track actions (strategy decisions)
                if 'action' in self.locals:
                    action = self.locals['action']
                    if isinstance(action, (np.ndarray, list)):
                        action = action[0] if len(action) > 0 else 0
                    self.episode_actions.append(int(action))

        # Print progress periodically
        if self.num_timesteps - self.last_print_step >= self.print_freq:
            self._print_progress()
            self.last_print_step = self.num_timesteps

        # Save checkpoint periodically
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = f"{self.save_path}/engineer_checkpoint_{self.n_calls}.zip"
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"💾 Checkpoint saved: {checkpoint_path}\n")

        return True

    def _print_progress(self):
        """Print detailed training progress."""
        if len(self.episode_rewards) == 0:
            return

        # Time statistics
        elapsed = time.time() - self.start_time if self.start_time else 0
        steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0

        print(f"\n{'─'*70}")
        print(f"  ENGINEER TRAINING PROGRESS - Step {self.num_timesteps:,}")
        print(f"{'─'*70}")
        print(f"  Elapsed time: {elapsed/60:.1f} min  |  Speed: {steps_per_sec:.0f} steps/s")

        # Recent episode stats (last 100 episodes)
        recent_n = min(100, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_n:]
        recent_lengths = self.episode_lengths[-recent_n:]

        print(f"\n  Last {recent_n} episodes:")
        print(f"    Avg reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
        print(f"    Best reward: {max(recent_rewards):.2f}")
        print(f"    Avg steps: {np.mean(recent_lengths):.1f}")

        # Strategy decision statistics (if available)
        if len(self.episode_actions) > 0:
            recent_actions = self.episode_actions[-recent_n:]
            action_names = ["Continue", "Pit-Soft", "Pit-Medium", "Pit-Hard"]

            print(f"\n  Strategy decisions (last {len(recent_actions)} episodes):")
            for action_id in range(4):
                count = sum(1 for a in recent_actions if a == action_id)
                pct = count / len(recent_actions) * 100 if recent_actions else 0
                action_name = action_names[action_id] if action_id < len(action_names) else f"Action-{action_id}"
                print(f"    {action_name}: {count} ({pct:.1f}%)")

        # Update best mean reward
        mean_reward = np.mean(recent_rewards)
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            print(f"\n  🏆 New best mean reward: {self.best_mean_reward:.2f}")

        print(f"{'─'*70}\n")

        # Clear old data to save memory (keep last 1000)
        max_keep = 1000
        if len(self.episode_rewards) > max_keep:
            self.episode_rewards = self.episode_rewards[-max_keep:]
            self.episode_lengths = self.episode_lengths[-max_keep:]
            self.episode_actions = self.episode_actions[-max_keep:]

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout (DQN doesn't have rollouts, but keeping for consistency)."""
        pass

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            total_time = time.time() - self.start_time if self.start_time else 0
            print(f"\n{'='*70}")
            print(f"  ENGINEER TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"  Total time: {total_time/60:.1f} minutes")
            print(f"  Total episodes: {len(self.episode_rewards)}")
            if len(self.episode_rewards) > 0:
                print(f"  Best mean reward: {self.best_mean_reward:.2f}")
            print(f"{'='*70}\n")


class StrategyAnalyzer:
    """
    Utility class for analyzing race strategy decisions.
    """

    @staticmethod
    def evaluate_tire_choice(
        current_lap: int,
        total_laps: int,
        tire_wear: float,
        track_characteristics: dict
    ) -> str:
        """
        Heuristic for evaluating tire compound choice.

        Args:
            current_lap: Current lap in race
            total_laps: Total laps
            tire_wear: Current tire wear [0, 1]
            track_characteristics: Dict with track info (degradation rate, etc.)

        Returns:
            Recommended compound ("soft", "medium", "hard")
        """
        laps_remaining = total_laps - current_lap

        # Simple heuristic logic
        if laps_remaining < 5:
            return "soft"  # Sprint to finish
        elif laps_remaining < 15:
            return "medium"  # Balanced choice
        else:
            return "hard"  # Long stint

    @staticmethod
    def estimate_pit_loss(speed: float, pit_speed_limit: float = 80) -> float:
        """
        Estimate time loss from pit stop.

        Args:
            speed: Current racing speed
            pit_speed_limit: Pit lane speed limit

        Returns:
            Estimated time loss in seconds
        """
        # Simplified calculation
        pit_lane_length = 300  # units
        pit_stop_time = 3.0  # seconds for tire change

        travel_time = pit_lane_length / pit_speed_limit
        racing_time = pit_lane_length / speed

        return (travel_time - racing_time) + pit_stop_time
