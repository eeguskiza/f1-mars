"""Integration tests for curriculum learning with training pipeline."""

import pytest
import sys
sys.path.insert(0, '.')

from f1_mars.envs import F1Env, CurriculumWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def test_curriculum_with_ppo():
    """Test that curriculum wrapper works with PPO training."""
    # Create env with curriculum
    base_env = F1Env(max_laps=1)
    curriculum_env = CurriculumWrapper(base_env, initial_level=0, enable_logging=False)

    # Wrap in vectorized env
    vec_env = DummyVecEnv([lambda: curriculum_env])

    # Create PPO model
    model = PPO("MlpPolicy", vec_env, device="cpu", verbose=0)

    # Train for a few steps
    model.learn(total_timesteps=100, progress_bar=False)

    # Verify training completed
    assert model.num_timesteps >= 100

    vec_env.close()


def test_curriculum_info_in_training():
    """Test that curriculum info is available during training."""
    base_env = F1Env(max_laps=1)
    curriculum_env = CurriculumWrapper(base_env, initial_level=0, enable_logging=False)

    # Reset and check info
    obs, info = curriculum_env.reset()

    assert 'curriculum' in info
    assert info['curriculum']['level'] == 0
    assert info['curriculum']['level_name'] == 'Basic'

    # Take a step and check info
    action = curriculum_env.action_space.sample()
    obs, reward, term, trunc, info = curriculum_env.step(action)

    assert 'curriculum' in info

    curriculum_env.close()


def test_curriculum_with_multiple_episodes():
    """Test curriculum across multiple episodes."""
    base_env = F1Env(max_laps=1)
    curriculum_env = CurriculumWrapper(base_env, initial_level=0, enable_logging=False)

    # Run 5 episodes
    for episode in range(5):
        obs, info = curriculum_env.reset()
        done = False
        steps = 0
        max_steps = 50

        while not done and steps < max_steps:
            action = curriculum_env.action_space.sample()
            obs, reward, term, trunc, info = curriculum_env.step(action)
            done = term or trunc
            steps += 1

    # Check that episodes were counted
    assert curriculum_env.episode_count == 5

    curriculum_env.close()


def test_curriculum_level_persistence():
    """Test that curriculum level persists across resets."""
    base_env = F1Env(max_laps=1)
    curriculum_env = CurriculumWrapper(base_env, initial_level=2, enable_logging=False)

    # Check initial level
    assert curriculum_env.current_level == 2

    # Reset multiple times
    for _ in range(3):
        curriculum_env.reset()

    # Level should still be 2 (not enough data to change)
    assert curriculum_env.current_level == 2

    curriculum_env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
