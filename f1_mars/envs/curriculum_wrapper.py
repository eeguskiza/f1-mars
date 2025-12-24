"""
Curriculum Learning Wrapper for F1-MARS.

Implements progressive difficulty levels to facilitate agent learning.
Automatically adjusts environment parameters based on agent performance.
"""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any
from pathlib import Path


class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum learning wrapper that progressively increases difficulty.

    Levels:
    - Level 0 (Basic): Simple oval, no tyre wear, moving start, extra rewards
    - Level 1 (Intermediate): Moderate tracks, reduced wear, normal rewards
    - Level 2 (Advanced): All tracks, normal wear, full mechanics
    - Level 3 (Expert): Hard tracks, increased wear, full challenge

    The wrapper automatically advances or retreats levels based on performance.
    """

    # Level configurations
    LEVEL_CONFIGS = {
        0: {
            "name": "Basic",
            "track_difficulty": 0,
            "tyre_wear_multiplier": 0.0,  # No wear
            "initial_velocity": 20.0,  # m/s (~72 km/h)
            "progress_bonus": 0.05,  # Extra reward for progress
            "success_threshold": 0.6,  # 60% episodes complete 1 lap
            "target_lap_time": 25.0,  # seconds
        },
        1: {
            "name": "Intermediate",
            "track_difficulty": 1,
            "tyre_wear_multiplier": 0.5,  # Half wear
            "initial_velocity": 0.0,  # Start from stop
            "progress_bonus": 0.0,
            "success_threshold": 0.7,  # 70% success
            "target_lap_time": 32.0,
        },
        2: {
            "name": "Advanced",
            "track_difficulty": 2,
            "tyre_wear_multiplier": 1.0,  # Normal wear
            "initial_velocity": 0.0,
            "progress_bonus": 0.0,
            "success_threshold": 0.75,  # 75% success
            "target_lap_time": 38.0,
        },
        3: {
            "name": "Expert",
            "track_difficulty": 3,
            "tyre_wear_multiplier": 1.5,  # Increased wear
            "initial_velocity": 0.0,
            "progress_bonus": 0.0,
            "success_threshold": 0.8,  # 80% success
            "target_lap_time": 45.0,
        },
    }

    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        initial_level: int = 0,
        enable_logging: bool = True
    ):
        """
        Initialize curriculum wrapper.

        Args:
            env: Base F1Env to wrap
            config: Optional configuration overrides
            initial_level: Starting curriculum level (0-3)
            enable_logging: Whether to log level changes
        """
        super().__init__(env)

        self.config = config or {}
        self.enable_logging = enable_logging

        # Curriculum state
        self.current_level = max(0, min(3, initial_level))
        self.episode_count = 0
        self.episodes_at_level = 0

        # Performance tracking
        self.window_size = self.config.get("window_size", 20)
        self.episode_results = deque(maxlen=self.window_size)
        self.lap_times = deque(maxlen=self.window_size)

        # Thresholds for level changes
        self.min_episodes_before_advance = self.config.get("min_episodes_advance", 20)
        self.min_episodes_before_retreat = self.config.get("min_episodes_retreat", 50)
        self.retreat_threshold = self.config.get("retreat_threshold", 0.3)

        # Store original env parameters for restoration
        self._original_track_path = getattr(env, 'track_path', None)
        self._original_wear_rate = None

        # Episode tracking
        self._episode_laps_completed = 0
        self._episode_max_progress = 0.0
        self._episode_start_time = 0

        self._log(f"CurriculumWrapper initialized at level {self.current_level}")

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment with curriculum level settings.

        Returns:
            obs: Observation
            info: Info dict with curriculum info added
        """
        # Evaluate progress and potentially change level
        if self.episode_count > 0:
            self._evaluate_progress()

        # Apply level-specific settings before reset
        self._apply_level_settings()

        # Reset base environment
        obs, info = self.env.reset(**kwargs)

        # Apply initial velocity if needed
        level_config = self.LEVEL_CONFIGS[self.current_level]
        initial_velocity = level_config["initial_velocity"]

        if initial_velocity > 0:
            # Set initial velocity (modify car state)
            if hasattr(self.env, 'car'):
                self.env.car.velocity = initial_velocity
                self.env.car.acceleration = 0.0

        # Reset episode tracking
        self._episode_laps_completed = 0
        self._episode_max_progress = 0.0
        self._episode_start_time = info.get('total_time', 0.0)

        # Add curriculum info to info dict
        info['curriculum'] = self.get_curriculum_info()

        self.episode_count += 1
        self.episodes_at_level += 1

        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step environment with curriculum-modified rewards.

        Args:
            action: Action to take

        Returns:
            obs, reward, terminated, truncated, info (with curriculum info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track episode progress
        self._episode_laps_completed = info.get('lap', 0)
        current_progress = info.get('lap_progress', 0.0)
        self._episode_max_progress = max(self._episode_max_progress, current_progress)

        # Apply level-specific reward modifications
        level_config = self.LEVEL_CONFIGS[self.current_level]
        progress_bonus = level_config["progress_bonus"]

        if progress_bonus > 0:
            # Calculate progress this step
            distance_traveled = info.get('velocity', 0) * 0.016  # dt = 0.016
            if distance_traveled > 0:
                reward += progress_bonus

        # Record episode result if finished
        if terminated or truncated:
            self._record_episode_result(info)

        # Add curriculum info
        info['curriculum'] = self.get_curriculum_info()

        return obs, reward, terminated, truncated, info

    def _apply_level_settings(self):
        """Apply curriculum level settings to environment."""
        level_config = self.LEVEL_CONFIGS[self.current_level]

        # 1. Set track based on difficulty
        track_difficulty = level_config["track_difficulty"]
        self._set_track_by_difficulty(track_difficulty)

        # 2. Modify tyre wear rate
        wear_multiplier = level_config["tyre_wear_multiplier"]
        if hasattr(self.env, 'tyres'):
            if self._original_wear_rate is None:
                # Store original on first call
                self._original_wear_rate = self.env.tyres.wear_rate

            self.env.tyres.wear_rate = self._original_wear_rate * wear_multiplier

        # 3. Log settings
        if self.enable_logging:
            self._log(
                f"Applied Level {self.current_level} ({level_config['name']}) settings: "
                f"difficulty={track_difficulty}, wear={wear_multiplier:.1f}x"
            )

    def _set_track_by_difficulty(self, difficulty: int):
        """
        Set track based on difficulty level.

        Args:
            difficulty: Difficulty level (0-3)
        """
        # Try to use tracks from tracks/ directory
        try:
            from tracks import get_tracks_by_difficulty, TRACKS_DIR

            track_names = get_tracks_by_difficulty(difficulty)

            if track_names:
                # Use first track of this difficulty
                track_path = str(TRACKS_DIR / f"{track_names[0]}.json")

                # Reload track if path changed
                if track_path != self._original_track_path:
                    from f1_mars.envs.track import Track

                    self.env.track = Track(track_path)
                    self._original_track_path = track_path
                    self._log(f"Loaded track: {track_names[0]} (difficulty {difficulty})")
            else:
                self._log(f"No track found for difficulty {difficulty}, using default")

        except ImportError:
            # Tracks module not available, use default
            self._log(f"Tracks module not available, using default track")

    def _evaluate_progress(self):
        """
        Evaluate agent progress and adjust curriculum level.

        Checks recent performance and decides whether to advance,
        retreat, or maintain current level.
        """
        if len(self.episode_results) < self.window_size // 2:
            # Not enough data yet
            return

        # Calculate success rate (completed at least 1 lap)
        recent_results = list(self.episode_results)
        success_count = sum(1 for r in recent_results if r['laps_completed'] >= 1)
        success_rate = success_count / len(recent_results)

        # Get level config
        level_config = self.LEVEL_CONFIGS[self.current_level]
        success_threshold = level_config["success_threshold"]

        # Check for advancement
        can_advance = (
            self.current_level < 3 and
            self.episodes_at_level >= self.min_episodes_before_advance and
            success_rate >= success_threshold
        )

        # Check for retreat
        can_retreat = (
            self.current_level > 0 and
            self.episodes_at_level >= self.min_episodes_before_retreat and
            success_rate < self.retreat_threshold
        )

        if can_advance:
            self._advance_level(success_rate)
        elif can_retreat:
            self._retreat_level(success_rate)

    def _advance_level(self, success_rate: float):
        """
        Advance to next curriculum level.

        Args:
            success_rate: Current success rate that triggered advancement
        """
        old_level = self.current_level
        self.current_level += 1
        self.episodes_at_level = 0

        new_config = self.LEVEL_CONFIGS[self.current_level]

        self._log(
            f"📈 ADVANCED: Level {old_level} → {self.current_level} "
            f"({new_config['name']}) after {self.episode_count} episodes "
            f"(success rate: {success_rate:.1%})"
        )

        # Clear performance history
        self.episode_results.clear()
        self.lap_times.clear()

    def _retreat_level(self, success_rate: float):
        """
        Retreat to previous curriculum level.

        Args:
            success_rate: Current success rate that triggered retreat
        """
        old_level = self.current_level
        self.current_level -= 1
        self.episodes_at_level = 0

        new_config = self.LEVEL_CONFIGS[self.current_level]

        self._log(
            f"📉 RETREATED: Level {old_level} → {self.current_level} "
            f"({new_config['name']}) after {self.episode_count} episodes "
            f"(success rate: {success_rate:.1%})"
        )

        # Keep some performance history
        # (don't clear completely to avoid oscillation)
        if len(self.episode_results) > 10:
            # Keep only last 10
            for _ in range(len(self.episode_results) - 10):
                self.episode_results.popleft()
                if self.lap_times:
                    self.lap_times.popleft()

    def _record_episode_result(self, info: Dict):
        """
        Record episode result for progress evaluation.

        Args:
            info: Episode info dict
        """
        result = {
            'laps_completed': info.get('lap', 0),
            'max_progress': self._episode_max_progress,
            'final_reward': info.get('episode_reward', 0),
            'success': info.get('lap', 0) >= 1,
        }

        # Record lap time if completed at least 1 lap
        if result['laps_completed'] >= 1:
            elapsed_time = info.get('total_time', 0) - self._episode_start_time
            avg_lap_time = elapsed_time / result['laps_completed']
            self.lap_times.append(avg_lap_time)
            result['avg_lap_time'] = avg_lap_time

        self.episode_results.append(result)

    def get_curriculum_info(self) -> Dict[str, Any]:
        """
        Get current curriculum state information.

        Returns:
            Dictionary with curriculum state
        """
        level_config = self.LEVEL_CONFIGS[self.current_level]

        # Calculate recent success rate
        if self.episode_results:
            recent_results = list(self.episode_results)
            success_rate = sum(r['success'] for r in recent_results) / len(recent_results)
        else:
            success_rate = 0.0

        # Calculate average lap time
        avg_lap_time = np.mean(self.lap_times) if self.lap_times else 0.0

        return {
            'level': self.current_level,
            'level_name': level_config['name'],
            'episode_count': self.episode_count,
            'episodes_at_level': self.episodes_at_level,
            'success_rate': success_rate,
            'avg_lap_time': avg_lap_time,
            'target_lap_time': level_config['target_lap_time'],
            'success_threshold': level_config['success_threshold'],
            'tyre_wear_multiplier': level_config['tyre_wear_multiplier'],
        }

    def set_level(self, level: int):
        """
        Manually set curriculum level (for testing/debugging).

        Args:
            level: Level to set (0-3)
        """
        if level < 0 or level > 3:
            raise ValueError(f"Level must be 0-3, got {level}")

        old_level = self.current_level
        self.current_level = level
        self.episodes_at_level = 0

        self._log(f"🔧 MANUAL OVERRIDE: Level {old_level} → {level}")

        # Clear history
        self.episode_results.clear()
        self.lap_times.clear()

    def _log(self, message: str):
        """
        Log curriculum event.

        Args:
            message: Message to log
        """
        if self.enable_logging:
            print(f"[Curriculum] {message}")

    def __repr__(self) -> str:
        """String representation."""
        level_config = self.LEVEL_CONFIGS[self.current_level]
        return (
            f"CurriculumWrapper(level={self.current_level} ({level_config['name']}), "
            f"episodes={self.episode_count}, at_level={self.episodes_at_level})"
        )


class CurriculumCallback:
    """
    Callback for logging curriculum progress to TensorBoard.

    Use with Stable-Baselines3 training.
    """

    def __init__(self, curriculum_env):
        """
        Initialize callback.

        Args:
            curriculum_env: CurriculumWrapper instance
        """
        self.curriculum_env = curriculum_env

    def on_step(self) -> bool:
        """Called after each environment step."""
        # Get curriculum info
        info = self.curriculum_env.get_curriculum_info()

        # Log to TensorBoard (if logger available)
        if hasattr(self, 'logger') and self.logger:
            self.logger.record("curriculum/level", info['level'])
            self.logger.record("curriculum/episodes_at_level", info['episodes_at_level'])
            self.logger.record("curriculum/success_rate", info['success_rate'])

            if info['avg_lap_time'] > 0:
                self.logger.record("curriculum/avg_lap_time", info['avg_lap_time'])

        return True


# Convenience function
def wrap_with_curriculum(
    env: gym.Env,
    initial_level: int = 0,
    config: Optional[Dict] = None,
    enable_logging: bool = True
) -> CurriculumWrapper:
    """
    Wrap environment with curriculum learning.

    Args:
        env: Base environment
        initial_level: Starting level (0-3)
        config: Optional configuration
        enable_logging: Whether to log level changes

    Returns:
        Wrapped environment
    """
    return CurriculumWrapper(
        env,
        config=config,
        initial_level=initial_level,
        enable_logging=enable_logging
    )
