#!/usr/bin/env python3
"""
Example: Training with Curriculum Learning

Demonstrates how to use the CurriculumWrapper to progressively
train an F1 agent from simple to complex tasks.
"""

import sys
sys.path.insert(0, '.')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from f1_mars.envs import F1Env, CurriculumWrapper, wrap_with_curriculum


class CurriculumLoggingCallback(BaseCallback):
    """
    Callback to log curriculum progress during training.
    """

    def __init__(self, curriculum_env: CurriculumWrapper, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum_env = curriculum_env
        self.last_level = 0

    def _on_step(self) -> bool:
        # Log curriculum info every 1000 steps
        if self.n_calls % 1000 == 0:
            info = self.curriculum_env.get_curriculum_info()

            # Log to TensorBoard
            if self.logger:
                self.logger.record("curriculum/level", info['level'])
                self.logger.record("curriculum/episodes_at_level", info['episodes_at_level'])
                self.logger.record("curriculum/success_rate", info['success_rate'])

                if info['avg_lap_time'] > 0:
                    self.logger.record("curriculum/avg_lap_time", info['avg_lap_time'])

            # Print level changes
            if info['level'] != self.last_level:
                print(f"\n{'='*60}")
                print(f"  CURRICULUM LEVEL CHANGE: {self.last_level} → {info['level']}")
                print(f"  Level: {info['level_name']}")
                print(f"  Episodes: {info['episode_count']} (at level: {info['episodes_at_level']})")
                print(f"  Success rate: {info['success_rate']:.1%}")
                print(f"{'='*60}\n")
                self.last_level = info['level']

        return True


def main():
    """Train agent with curriculum learning."""

    # Create base environment
    base_env = F1Env(max_laps=3)

    # Wrap with curriculum learning
    # Option 1: Using convenience function
    env = wrap_with_curriculum(
        base_env,
        initial_level=0,  # Start at basic level
        enable_logging=True
    )

    # Option 2: Using class directly with custom config
    # custom_config = {
    #     "window_size": 30,  # Consider last 30 episodes
    #     "min_episodes_advance": 30,  # Need 30 episodes before advancing
    #     "retreat_threshold": 0.2,  # Retreat if success < 20%
    # }
    # env = CurriculumWrapper(base_env, config=custom_config, initial_level=0)

    print("\n" + "="*70)
    print("  CURRICULUM LEARNING TRAINING")
    print("="*70)
    print(f"Starting at Level {env.current_level}: {env.LEVEL_CONFIGS[0]['name']}")
    print(f"Will automatically progress through 4 difficulty levels")
    print("="*70 + "\n")

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="logs/curriculum/",
        device="cpu"
    )

    # Create callback
    callback = CurriculumLoggingCallback(env, verbose=1)

    # Train
    total_timesteps = 500000

    print(f"Training for {total_timesteps:,} timesteps...\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Print final curriculum state
    final_info = env.get_curriculum_info()

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"Final level: {final_info['level']} ({final_info['level_name']})")
    print(f"Total episodes: {final_info['episode_count']}")
    print(f"Final success rate: {final_info['success_rate']:.1%}")

    if final_info['avg_lap_time'] > 0:
        print(f"Average lap time: {final_info['avg_lap_time']:.2f}s")

    print("="*70 + "\n")

    # Save model
    model_path = "trained_models/curriculum_pilot.zip"
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
