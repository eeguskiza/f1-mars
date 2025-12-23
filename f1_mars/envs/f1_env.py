"""Main Gymnasium environment for F1 Mars simulator."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from f1_mars.envs.car import Car
from f1_mars.envs.track import Track
from f1_mars.envs.tyres import TyreSystem
from f1_mars.utils.config import *


class F1Env(gym.Env):
    """
    F1 Mars Gymnasium Environment.

    A 2D racing simulator where an agent learns to drive a Formula 1 car
    around a track while managing tire degradation and race strategy.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        track_name: str = "example_circuit",
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the F1 environment.

        Args:
            track_name: Name of the track to load (without .json extension)
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()

        self.render_mode = render_mode
        self.track_name = track_name

        # Initialize components
        self.track = Track(track_name)
        self.car = Car()
        self.tyre_system = TyreSystem()

        # Define action space: [steering, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Define observation space
        # LIDAR (16) + speed (1) + steering (1) + tire wear (4) + fuel (1) + checkpoint_dist (1) + lap_progress (1)
        obs_dim = LIDAR_RAYS + 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.current_lap = 0
        self.last_checkpoint = 0
        self.episode_reward = 0.0

        # Renderer (initialized on first render call)
        self.renderer = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset all components
        self.car.reset(self.track.start_position)
        self.tyre_system.reset()
        self.current_step = 0
        self.current_lap = 0
        self.last_checkpoint = 0
        self.episode_reward = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: [steering, throttle, brake]

        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode ended (crash/completion)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        steering, throttle, brake = action

        # Update car physics
        for _ in range(PHYSICS_STEPS_PER_FRAME):
            self.car.update(steering, throttle, brake, PHYSICS_DT)

        # Update tire degradation
        self.tyre_system.update(self.car.speed)

        # Check collisions
        collision = self.track.check_collision(self.car.position)

        # Check checkpoint progress
        checkpoint_passed = self.track.check_checkpoint(
            self.car.position, self.last_checkpoint
        )

        # Calculate reward
        reward = self._calculate_reward(collision, checkpoint_passed)
        self.episode_reward += reward

        # Update state
        if checkpoint_passed:
            self.last_checkpoint += 1
            if self.last_checkpoint >= len(self.track.checkpoints):
                self.last_checkpoint = 0
                self.current_lap += 1

        self.current_step += 1

        # Check termination conditions
        terminated = collision or self.current_lap >= 3
        truncated = self.current_step >= MAX_EPISODE_STEPS

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.renderer is None:
            from f1_mars.rendering import Renderer
            self.renderer = Renderer(self.track, SCREEN_WIDTH, SCREEN_HEIGHT)

        return self.renderer.render(self.car, self.render_mode)

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation vector.

        Returns:
            Observation array
        """
        # LIDAR distances
        lidar = self._get_lidar_distances()

        # Car state
        car_state = np.array([
            self.car.speed / CAR_MAX_SPEED,  # Normalized speed
            self.car.steering / CAR_MAX_STEERING,  # Normalized steering
        ])

        # Tire wear
        tire_wear = self.tyre_system.get_wear_vector()

        # Other metrics
        fuel = np.array([self.car.fuel / 100.0])  # Normalized fuel
        checkpoint_dist = self._get_checkpoint_distance()
        lap_progress = np.array([self.last_checkpoint / len(self.track.checkpoints)])

        observation = np.concatenate([
            lidar,
            car_state,
            tire_wear,
            fuel,
            checkpoint_dist,
            lap_progress
        ])

        return observation.astype(np.float32)

    def _get_lidar_distances(self) -> np.ndarray:
        """Calculate LIDAR distance measurements."""
        # TODO: Implement raycasting for distance sensors
        return np.ones(LIDAR_RAYS) * LIDAR_RANGE

    def _get_checkpoint_distance(self) -> np.ndarray:
        """Get normalized distance to next checkpoint."""
        # TODO: Implement checkpoint distance calculation
        return np.array([0.5])

    def _calculate_reward(self, collision: bool, checkpoint_passed: bool) -> float:
        """
        Calculate reward for current step.

        Args:
            collision: Whether car collided with track boundary
            checkpoint_passed: Whether car passed a checkpoint

        Returns:
            Reward value
        """
        reward = 0.0

        if collision:
            reward += REWARD_COLLISION

        if checkpoint_passed:
            reward += REWARD_CHECKPOINT

        # Speed bonus (encourage faster driving)
        reward += self.car.speed / CAR_MAX_SPEED * REWARD_SPEED_BONUS

        # Penalty for driving backwards
        if self.car.speed < 0:
            reward += REWARD_REVERSE

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information dictionary."""
        return {
            "lap": self.current_lap,
            "checkpoint": self.last_checkpoint,
            "speed": self.car.speed,
            "tire_wear": self.tyre_system.get_average_wear(),
            "fuel": self.car.fuel,
            "episode_reward": self.episode_reward,
        }
