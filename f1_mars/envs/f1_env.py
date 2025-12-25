"""Main Gymnasium environment for F1 Mars simulator - BALANCED."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from f1_mars.envs.car import Car
from f1_mars.envs.track import Track
from f1_mars.envs.tyres import TyreSet, TyreCompound
from f1_mars.utils.geometry import raycast


class F1Env(gym.Env):
    """
    F1 Mars Gymnasium Environment - BALANCED.

    A 2D racing simulator where an agent learns to drive a Formula 1 car
    around a track while managing tire degradation and race strategy.

    This environment implements the pilot agent. The engineer agent will
    be integrated later as a wrapper or meta-controller.

    Action Space:
        Box([-1, 0, 0], [1, 1, 1]): [steering, throttle, brake]
        - steering: -1 (full left) to 1 (full right)
        - throttle: 0 to 1
        - brake: 0 to 1

    Observation Space:
        Box with 25-30 dimensions:
        1. Velocity (normalized)
        2. Steering angle (normalized)
        3. Heading relative to track (radians)
        4. Lateral offset from centerline (normalized)
        5-15. LIDAR rays (11 rays from -75° to +75°, normalized distances)
        16-20. Track curvature at next 5 points ahead
        21. Tyre wear (%)
        22. Current grip multiplier
        23. Engineer signal (0-2: continue/pit/change_compound)
        24. Current lap
        25. Total laps
        26. Distance along track (normalized)

    Reward Function (AGGRESSIVE for F1 Racing):
        1. Speed reward (PRINCIPAL): velocity projected along track direction × 2.0
           - DOUBLED weight to force F1-level speeds (250-350 km/h)
           - Extra bonus (+0.5) for speeds >216 km/h (60 m/s)
           - Penalty for going slow or backwards
        2. Progress reward: +0.1 per meter forward along track
        3. Checkpoint bonus: +5 per checkpoint
        4. Lap completion: +50 + time bonus (2 points per second under reference)
        5. Penalties (AGGRESSIVE):
           - Low speed: -1.0 proportional if < 20 m/s (72 km/h) - FORCES F1 SPEEDS
           - Off-track: -1.0 base + speed-proportional penalty
           - Time step: -0.01 per step (encourages fast completion)
           - High wear: -0.1 proportional if > 70%
           - Dead tyres: -50
        6. Bonus: grip utilization when cornering aggressively on-track
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        track_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        config: Optional[Dict] = None,
        max_laps: int = 3
    ):
        """
        Initialize the F1 environment.

        Args:
            track_path: Path to track JSON file (if None, uses default oval)
            render_mode: Rendering mode ("human" or "rgb_array")
            config: Optional configuration overrides for car physics
            max_laps: Maximum laps before episode terminates
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_laps = max_laps
        self.config = config or {}

        # Load track
        if track_path is None:
            # Use default oval track
            from f1_mars.envs.track import TrackGenerator
            track_data = TrackGenerator.generate_oval(length=1000, width=12.0)
            self.track = Track.__new__(Track)
            self.track.load_from_dict(track_data)
        else:
            self.track = Track(track_path)

        # Initialize car at track start position
        self.car = Car(
            x=self.track.start_x,
            y=self.track.start_y,
            heading=self.track.start_heading,
            config=self.config
        )

        # Initialize tyres with default compound
        self.tyres = TyreSet(TyreCompound.MEDIUM)

        # Get track boundary segments for LIDAR (computed once for efficiency)
        self.boundary_segments = self.track.get_boundary_segments(num_samples=200)

        # Define action space: [steering, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Define observation space
        # 1 velocity + 1 steering + 1 heading_rel + 1 lateral_offset +
        # 11 LIDAR + 5 curvatures + 1 tyre_wear + 1 grip +
        # 1 engineer_signal + 1 current_lap + 1 total_laps + 1 distance_along
        obs_dim = 26
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # LIDAR configuration
        self.lidar_angles = np.linspace(-75, 75, 11)  # degrees
        self.lidar_max_distance = 50.0  # meters

        # Physics timestep configuration
        self.dt = 1.0 / 60.0  # 60 FPS
        self.physics_substeps = 1  # Car class handles subdivision internally

        # State variables
        self.current_step = 0
        self.current_lap = 1
        self.last_checkpoint_index = -1
        self.episode_reward = 0.0
        self.last_distance_along = 0.0
        self.lap_start_time = 0.0
        self.total_time = 0.0
        self.reference_lap_time = 30.0  # seconds, estimated

        # Engineer signal (0=continue, 1=pit, 2=change_compound)
        self.engineer_signal = 0

        # Renderer (initialized on first render call)
        self.renderer = None

        # Info tracking
        self.off_track_frames = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (e.g., starting compound)

        Returns:
            observation: Initial observation vector
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        options = options or {}

        # Reset car to start position
        self.car.reset(
            x=self.track.start_x,
            y=self.track.start_y,
            heading=self.track.start_heading
        )

        # Reset tyres (optionally with different compound)
        starting_compound = options.get('compound', TyreCompound.MEDIUM)
        self.tyres.reset(starting_compound)

        # Reset state variables
        self.current_step = 0
        self.current_lap = 1
        self.last_checkpoint_index = -1
        self.episode_reward = 0.0
        self.last_distance_along = 0.0
        self.lap_start_time = 0.0
        self.total_time = 0.0
        self.engineer_signal = 0
        self.off_track_frames = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep in the environment.

        Args:
            action: [steering, throttle, brake] in ranges [-1,1], [0,1], [0,1]

        Returns:
            observation: Current observation vector
            reward: Reward for this step
            terminated: Whether episode ended (crash/completion/dead tyres)
            truncated: Whether episode was truncated (max steps)
            info: Additional information dictionary
        """
        steering_input, throttle, brake = action

        # Clamp actions to valid ranges
        steering_input = np.clip(steering_input, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # Get tyre grip for physics
        grip_multiplier = self.tyres.get_grip()

        # Store old position for progress tracking
        old_distance_along = self.last_distance_along
        old_checkpoint = self.last_checkpoint_index

        # Update car physics (with internal timestep subdivision)
        self.car.update(
            dt=self.dt,
            throttle=throttle,
            brake=brake,
            steering_input=steering_input,
            grip_multiplier=grip_multiplier
        )

        # Update tyre degradation
        lateral_force = self.car.get_lateral_force()
        self.tyres.update(
            dt=self.dt,
            speed=self.car.velocity,
            lateral_force=lateral_force,
            throttle=throttle,
            braking=brake
        )

        # Get track information
        closest_pt, distance_along, lateral_offset, track_heading = \
            self.track.get_closest_point_on_track(self.car.position)

        # Check if on track
        on_track = self.track.is_on_track(self.car.position)
        if not on_track:
            self.off_track_frames += 1
        else:
            self.off_track_frames = 0

        # Check checkpoint progress
        current_checkpoint = self.track.get_checkpoint_index(distance_along)
        checkpoint_passed = False

        if current_checkpoint > self.last_checkpoint_index:
            checkpoint_passed = True
            self.last_checkpoint_index = current_checkpoint
        elif current_checkpoint == 0 and self.last_checkpoint_index == len(self.track.checkpoint_indices) - 1:
            # Crossed finish line
            checkpoint_passed = True
            self.last_checkpoint_index = 0

        # Check if lap completed
        lap_completed = False
        if distance_along < 50.0 and old_distance_along > self.track.total_length - 50.0:
            # Crossed finish line
            lap_completed = True
            actual_lap_time = self.total_time - self.lap_start_time
            self.lap_start_time = self.total_time
            self.current_lap += 1

        # Update tracking
        self.last_distance_along = distance_along
        self.total_time += self.dt
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward(
            old_distance_along=old_distance_along,
            new_distance_along=distance_along,
            old_checkpoint=old_checkpoint,
            new_checkpoint=self.last_checkpoint_index,
            lap_completed=lap_completed,
            actual_lap_time=actual_lap_time if lap_completed else 0.0,
            on_track=on_track
        )
        self.episode_reward += reward

        # Check termination conditions
        terminated = False

        # Crash: too long off track
        if self.off_track_frames > 60:  # 1 second off track
            terminated = True

        # Dead tyres
        if self.tyres.is_dead():
            terminated = True

        # Completed max laps
        if self.current_lap > self.max_laps:
            terminated = True

        # Truncation: max episode steps (e.g., 60 seconds = 3600 steps)
        truncated = self.current_step >= 3600

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns:
            np.ndarray if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode is None:
            return None

        if self.renderer is None:
            # Lazy initialization of renderer
            try:
                from f1_mars.rendering.renderer import Renderer
                self.renderer = Renderer(
                    track=self.track,
                    width=1200,
                    height=800
                )
            except ImportError:
                # Rendering not available
                return None

        return self.renderer.render(self.car, self.tyres, self.render_mode)

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def set_engineer_signal(self, signal: int):
        """
        Set engineer signal for the pilot agent.

        Args:
            signal: 0 = continue, 1 = pit, 2 = change compound
        """
        self.engineer_signal = np.clip(signal, 0, 2)

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation vector.

        Returns:
            Observation array with 26 dimensions
        """
        # Get track info
        closest_pt, distance_along, lateral_offset, track_heading = \
            self.track.get_closest_point_on_track(self.car.position)

        # 1. Velocity (normalized by max speed)
        velocity_norm = self.car.velocity / self.car.max_speed

        # 2. Steering angle (normalized)
        steering_norm = self.car.steering_angle / self.car.max_steering

        # 3. Heading relative to track (wrapped to [-π, π])
        heading_error = self.car.heading - track_heading
        heading_rel = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # 4. Lateral offset (normalized by track width)
        track_width = self.track.widths[0] if hasattr(self.track, 'widths') else 12.0
        lateral_offset_norm = lateral_offset / (track_width / 2.0)

        # 5-15. LIDAR rays (11 rays)
        lidar_distances = self._calculate_raycast_observations()

        # 16-20. Track curvature ahead at 5 points (every 50m)
        curvatures = []
        for i in range(1, 6):
            future_distance = distance_along + i * 50.0
            if future_distance > self.track.total_length:
                future_distance -= self.track.total_length
            curvature = self.track.get_curvature_at_distance(future_distance)
            curvatures.append(curvature)

        # 21. Tyre wear (0-100%)
        tyre_wear_norm = self.tyres.wear / 100.0

        # 22. Current grip multiplier
        grip = self.tyres.get_grip()

        # 23. Engineer signal (0, 1, or 2)
        engineer_sig = float(self.engineer_signal)

        # 24. Current lap
        current_lap_norm = float(self.current_lap) / float(self.max_laps)

        # 25. Total laps (normalized)
        total_laps_norm = float(self.max_laps) / 10.0  # Assuming max 10 laps

        # 26. Distance along track (normalized)
        distance_norm = distance_along / self.track.total_length

        # Concatenate all observations
        observation = np.array([
            velocity_norm,
            steering_norm,
            heading_rel,
            lateral_offset_norm,
            *lidar_distances,
            *curvatures,
            tyre_wear_norm,
            grip,
            engineer_sig,
            current_lap_norm,
            total_laps_norm,
            distance_norm
        ], dtype=np.float32)

        return observation

    def _calculate_raycast_observations(self) -> np.ndarray:
        """
        Calculate LIDAR raycast distances.

        Casts 11 rays from -75° to +75° relative to car heading.

        Returns:
            Array of 11 normalized distances [0, 1]
        """
        distances = np.zeros(11, dtype=np.float32)

        car_pos = self.car.position
        car_heading = self.car.heading

        for i, angle_deg in enumerate(self.lidar_angles):
            # Convert to radians and add to car heading
            angle_rad = np.radians(angle_deg) + car_heading

            # Raycast to find intersection
            distance = raycast(
                origin=car_pos,
                direction=angle_rad,
                segments=self.boundary_segments,
                max_distance=self.lidar_max_distance
            )

            # Normalize to [0, 1]
            distances[i] = distance / self.lidar_max_distance

        return distances

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _calculate_reward(
        self,
        old_distance_along: float,
        new_distance_along: float,
        old_checkpoint: int,
        new_checkpoint: int,
        lap_completed: bool,
        actual_lap_time: float,
        on_track: bool
    ) -> float:
        """
        Reward function balanceada para racing - FIXED VERSION.

        Principios clave (basados en literatura de racing RL):
        1. Velocidad en dirección del track es el reward PRINCIPAL
        2. Penalización obligatoria por velocidad baja (evita óptimo local)
        3. Penalizaciones moderadas que NO dominan
        4. Time penalty para incentivar terminar rápido

        FIX: Aumenta dramáticamente el incentivo de velocidad y penaliza fuertemente
        ir lento para evitar el óptimo local de "quedarse casi parado".

        Args:
            old_distance_along: Previous distance along track
            new_distance_along: Current distance along track
            old_checkpoint: Previous checkpoint index
            new_checkpoint: Current checkpoint index
            lap_completed: Whether lap was completed
            actual_lap_time: Actual lap time (if completed)
            on_track: Whether car is on track

        Returns:
            Total reward for this step
        """
        reward = 0.0

        # === 1. SPEED REWARD (COMPONENTE PRINCIPAL) ===
        # Velocidad proyectada en la dirección del track
        # Este reward denso es el driver principal del comportamiento

        track_heading = self.track.get_direction_at_distance(new_distance_along)
        heading_diff = self._normalize_angle(self.car.heading - track_heading)

        # Velocidad efectiva en dirección del track
        # Positiva si va en dirección correcta, negativa si va en reversa
        velocity_along_track = self.car.velocity * np.cos(heading_diff)

        # Escalar: a máxima velocidad en dirección correcta = +1.5 por step
        speed_reward = (velocity_along_track / self.car.max_speed) * 1.5
        reward += speed_reward

        # === 2. PROGRESS REWARD ===
        # Bonus por metros avanzados en el circuito
        progress = new_distance_along - old_distance_along

        # Manejar wrap-around en línea de meta
        if progress < -self.track.total_length / 2:
            progress += self.track.total_length
        elif progress > self.track.total_length / 2:
            progress -= self.track.total_length

        progress_reward = progress * 0.1  # +0.1 por metro avanzado
        reward += progress_reward

        # === 3. PENALIZACIÓN POR VELOCIDAD BAJA (CRÍTICO) ===
        # Esto EVITA el óptimo local de quedarse parado
        min_speed = 10.0  # m/s (~36 km/h) - mínimo aceptable
        if self.car.velocity < min_speed:
            # Penalización proporcional: más lento = más penalización
            speed_deficit = (min_speed - self.car.velocity) / min_speed
            reward -= speed_deficit * 1.0  # Hasta -1.0 si está completamente parado

        # === 4. TIME STEP PENALTY ===
        # Pequeño costo por existir - incentiva terminar rápido
        reward -= 0.02

        # === 5. CHECKPOINT BONUS ===
        checkpoint_reached = (new_checkpoint > old_checkpoint) or \
                           (new_checkpoint == 0 and old_checkpoint == len(self.track.checkpoint_indices) - 1)
        if checkpoint_reached:
            reward += 5.0

        # === 6. LAP COMPLETION BONUS ===
        if lap_completed:
            reference_time = getattr(self.track, 'reference_lap_time', self.reference_lap_time) or 25.0

            # Bonus base + bonus por tiempo bajo referencia
            time_bonus = max(0, (reference_time - actual_lap_time) * 2.0)
            reward += 30.0 + time_bonus

        # === 7. PENALIZACIONES (MODERADAS) ===

        # Off-track: penalización reducida para no dominar
        if not on_track:
            reward -= 0.5  # Era -1.0, ahora -0.5

        # Desgaste excesivo de neumáticos
        if self.tyres.wear > 70:
            reward -= 0.05 * (self.tyres.wear - 70) / 30  # Máximo -0.05

        # Dead tyres
        if self.tyres.is_dead():
            reward -= 10.0  # Reducido de -50.0

        # === 8. BONUS POR CONDUCCIÓN AGRESIVA (EN PISTA) ===
        # Recompensar usar el coche cerca del límite
        if on_track:
            lateral_force = self.car.get_lateral_force()
            if lateral_force > 0.3:
                reward += lateral_force * 0.2  # Bonus por curvar fuerte

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information dictionary.

        Returns:
            Dictionary with detailed telemetry and state info
        """
        # Get track position
        _, distance_along, lateral_offset, _ = \
            self.track.get_closest_point_on_track(self.car.position)

        # Get tyre state
        tyre_state = self.tyres.get_state()

        return {
            # Lap info
            'lap': self.current_lap,
            'max_laps': self.max_laps,
            'checkpoint': self.last_checkpoint_index,
            'total_checkpoints': len(self.track.checkpoint_indices),

            # Position info
            'distance_along': distance_along,
            'track_length': self.track.total_length,
            'lap_progress': distance_along / self.track.total_length,
            'lateral_offset': lateral_offset,

            # Car state
            'position_x': self.car.position[0],
            'position_y': self.car.position[1],
            'velocity': self.car.velocity,
            'velocity_kmh': self.car.velocity * 3.6,
            'heading': self.car.heading,
            'steering_angle': self.car.steering_angle,

            # Tyre state
            'tyre_compound': tyre_state['compound'],
            'tyre_wear': tyre_state['wear'],
            'tyre_temperature': tyre_state['temperature'],
            'tyre_grip': tyre_state['current_grip'],

            # Episode info
            'episode_reward': self.episode_reward,
            'total_time': self.total_time,
            'timestep': self.current_step,

            # Engineer signal
            'engineer_signal': self.engineer_signal,

            # Track status
            'on_track': self.track.is_on_track(self.car.position),
            'off_track_frames': self.off_track_frames,
        }
