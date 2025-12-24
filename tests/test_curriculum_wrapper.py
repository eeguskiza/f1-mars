"""Tests for CurriculumWrapper."""

import pytest
import numpy as np
from f1_mars.envs import F1Env, CurriculumWrapper


class TestCurriculumWrapper:
    """Test suite for curriculum learning wrapper."""

    def test_wrapper_initialization(self):
        """Test that wrapper initializes correctly."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=0)

        assert curriculum_env.current_level == 0
        assert curriculum_env.episode_count == 0
        assert len(curriculum_env.episode_results) == 0

        curriculum_env.close()

    def test_initial_level_bounds(self):
        """Test that initial level is bounded to 0-3."""
        env = F1Env()

        # Test lower bound
        curriculum_env = CurriculumWrapper(env, initial_level=-5)
        assert curriculum_env.current_level == 0

        # Test upper bound
        curriculum_env = CurriculumWrapper(env, initial_level=10)
        assert curriculum_env.current_level == 3

        curriculum_env.close()

    def test_reset_increments_episode_count(self):
        """Test that reset increments episode counter."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=0)

        initial_count = curriculum_env.episode_count

        curriculum_env.reset()
        assert curriculum_env.episode_count == initial_count + 1

        curriculum_env.reset()
        assert curriculum_env.episode_count == initial_count + 2

        curriculum_env.close()

    def test_step_returns_correct_format(self):
        """Test that step returns correct tuple format."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=0)

        obs, info = curriculum_env.reset()

        # Take a step
        action = curriculum_env.action_space.sample()
        obs, reward, terminated, truncated, info = curriculum_env.step(action)

        # Check return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check curriculum info is added
        assert 'curriculum' in info
        assert 'level' in info['curriculum']

        curriculum_env.close()

    def test_curriculum_info_structure(self):
        """Test that get_curriculum_info returns correct structure."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=1)

        info = curriculum_env.get_curriculum_info()

        # Check all required keys are present
        required_keys = [
            'level',
            'level_name',
            'episode_count',
            'episodes_at_level',
            'success_rate',
            'avg_lap_time',
            'target_lap_time',
            'success_threshold',
            'tyre_wear_multiplier'
        ]

        for key in required_keys:
            assert key in info, f"Missing key: {key}"

        # Check level is correct
        assert info['level'] == 1
        assert info['level_name'] == 'Intermediate'

        curriculum_env.close()

    def test_manual_level_override(self):
        """Test manual level setting."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=0)

        # Set to level 2
        curriculum_env.set_level(2)
        assert curriculum_env.current_level == 2
        assert curriculum_env.episodes_at_level == 0

        # Check that history is cleared
        assert len(curriculum_env.episode_results) == 0

        curriculum_env.close()

    def test_manual_level_bounds(self):
        """Test that manual level setting respects bounds."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=0)

        # Test invalid level
        with pytest.raises(ValueError):
            curriculum_env.set_level(-1)

        with pytest.raises(ValueError):
            curriculum_env.set_level(4)

        curriculum_env.close()

    def test_level_configs_exist(self):
        """Test that all level configurations are defined."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env)

        for level in range(4):
            assert level in curriculum_env.LEVEL_CONFIGS
            config = curriculum_env.LEVEL_CONFIGS[level]

            # Check required keys
            assert 'name' in config
            assert 'track_difficulty' in config
            assert 'tyre_wear_multiplier' in config
            assert 'initial_velocity' in config
            assert 'progress_bonus' in config
            assert 'success_threshold' in config
            assert 'target_lap_time' in config

        curriculum_env.close()

    def test_level_0_has_progress_bonus(self):
        """Test that level 0 (basic) provides progress bonus."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=0)

        config = curriculum_env.LEVEL_CONFIGS[0]
        assert config['progress_bonus'] > 0

        curriculum_env.close()

    def test_level_0_has_no_wear(self):
        """Test that level 0 (basic) has no tyre wear."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=0)

        config = curriculum_env.LEVEL_CONFIGS[0]
        assert config['tyre_wear_multiplier'] == 0.0

        curriculum_env.close()

    def test_level_progression_thresholds(self):
        """Test that success thresholds increase with level."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env)

        thresholds = [
            curriculum_env.LEVEL_CONFIGS[i]['success_threshold']
            for i in range(4)
        ]

        # Thresholds should generally increase
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1], \
                f"Threshold at level {i} should be <= level {i+1}"

        curriculum_env.close()

    def test_episode_result_recording(self):
        """Test that episode results are recorded correctly."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=0)

        # Run one episode
        obs, info = curriculum_env.reset()

        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = curriculum_env.action_space.sample()
            obs, reward, terminated, truncated, info = curriculum_env.step(action)
            done = terminated or truncated
            steps += 1

        # After episode, should have one result (but only if episode finished)
        if done:
            assert len(curriculum_env.episode_results) > 0

        curriculum_env.close()

    def test_custom_config(self):
        """Test that custom config overrides defaults."""
        env = F1Env()

        custom_config = {
            "window_size": 50,
            "min_episodes_advance": 100,
            "retreat_threshold": 0.1
        }

        curriculum_env = CurriculumWrapper(env, config=custom_config)

        assert curriculum_env.window_size == 50
        assert curriculum_env.min_episodes_before_advance == 100
        assert curriculum_env.retreat_threshold == 0.1

        curriculum_env.close()

    def test_repr_string(self):
        """Test string representation."""
        env = F1Env()
        curriculum_env = CurriculumWrapper(env, initial_level=1)

        repr_str = repr(curriculum_env)

        assert "CurriculumWrapper" in repr_str
        assert "level=1" in repr_str
        assert "Intermediate" in repr_str

        curriculum_env.close()
