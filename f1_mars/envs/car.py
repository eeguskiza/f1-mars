"""Car physics model for F1 Mars simulator using bicycle kinematic model."""

import numpy as np
from typing import Dict, Optional


class Car:
    """
    Formula 1 car physics using simplified bicycle kinematic model for 2D top-down view.

    The bicycle model simplifies the four-wheeled car into a two-wheeled system,
    with front wheels that steer and rear wheels that follow. This is suitable
    for relatively low-speed racing simulation where tire slip is abstracted.

    Coordinate System:
        - X-axis: Positive to the right
        - Y-axis: Positive upward (standard mathematical convention)
        - heading: 0 radians points along positive X-axis, increases counterclockwise

    Dimensions:
        - Length: 4.5 meters (typical F1 car length)
        - Width: 2.0 meters (typical F1 car width)
        - Wheelbase: 3.5 meters (distance between front and rear axles)

    Performance:
        - Max speed: ~97 m/s (~350 km/h)
        - 0-100 km/h: ~2.5 seconds
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        config: Optional[Dict] = None
    ):
        """
        Initialize the car at a given position and orientation.

        Args:
            x: Initial X position in world coordinates (meters)
            y: Initial Y position in world coordinates (meters)
            heading: Initial heading angle in radians (0 = facing right)
            config: Optional dictionary to override default physical parameters
        """
        # State variables
        self.position = np.array([x, y], dtype=np.float64)
        self.velocity = 0.0  # Scalar speed (always forward in car frame)
        self.heading = heading  # Orientation in radians
        self.steering_angle = 0.0  # Current steering angle of front wheels

        # Geometric properties
        self.wheelbase = 3.5  # Distance between axles (meters)
        self.length = 4.5  # Overall car length (meters)
        self.width = 2.0  # Overall car width (meters)
        self.mass = 800.0  # Car + driver mass (kg)

        # Physical parameters (BALANCED for realistic F1 simulation)
        self.max_speed = 97.0  # Maximum velocity ~350 km/h (m/s)
        self.min_speed = 0.0  # Cannot go backwards in this model
        self.max_steering = 0.5  # Maximum steering angle (radians, ~28 degrees)
        self.acceleration_power = 35.0  # Forward acceleration (0-100 in ~2.5s)
        self.brake_power = 80.0  # Braking deceleration (stronger than accel)
        self.drag_coefficient = 0.004  # Aerodynamic drag (reduced for higher top speed)
        self.rolling_resistance = 0.001  # Rolling resistance (reduced)
        self.steering_speed = 2.0  # Steering response rate (rad/s)

        # Apply config overrides if provided
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def reset(self, x: float, y: float, heading: float):
        """
        Reset the car to a new position and orientation.

        Args:
            x: New X position (meters)
            y: New Y position (meters)
            heading: New heading angle (radians)
        """
        self.position = np.array([x, y], dtype=np.float64)
        self.velocity = 0.0
        self.heading = heading
        self.steering_angle = 0.0

    def update(
        self,
        dt: float,
        throttle: float,
        brake: float,
        steering_input: float,
        grip_multiplier: float = 1.0
    ):
        """
        Update car physics for one timestep using bicycle kinematic model.

        IMPORTANT: For numerical stability, large dt values are automatically
        subdivided into smaller steps.

        Args:
            dt: Time step in seconds (typically 1/60 = 0.016s)
            throttle: Throttle pedal input, range [0, 1]
            brake: Brake pedal input, range [0, 1]
            steering_input: Steering wheel input, range [-1, 1]
                           (negative = left, positive = right)
            grip_multiplier: Tire grip factor, range [0, 1]
                           (1.0 = full grip, lower = reduced grip from tire wear)
        """
        # Subdivide large timesteps for numerical stability
        max_dt = 0.02  # 20ms max per physics step
        steps = max(1, int(np.ceil(dt / max_dt)))
        sub_dt = dt / steps

        for _ in range(steps):
            self._physics_step(sub_dt, throttle, brake, steering_input, grip_multiplier)

    def _physics_step(
        self,
        dt: float,
        throttle: float,
        brake: float,
        steering_input: float,
        grip_multiplier: float
    ):
        """
        Single physics integration step.

        Args:
            dt: Small timestep (< 0.02s recommended)
            throttle: Throttle [0, 1]
            brake: Brake [0, 1]
            steering_input: Steering [-1, 1]
            grip_multiplier: Grip factor [0, 1]
        """
        # === STEERING UPDATE ===
        # Steering has inertia (not instantaneous)
        # Grip affects max steering angle (less grip = less turning ability)
        target_steering = steering_input * self.max_steering * grip_multiplier
        steering_rate = self.steering_speed * dt
        steering_change = np.clip(
            target_steering - self.steering_angle,
            -steering_rate,
            steering_rate
        )
        self.steering_angle += steering_change

        # === LONGITUDINAL ACCELERATIONS ===
        # Engine acceleration (direct acceleration, not force)
        engine_accel = throttle * self.acceleration_power

        # Braking acceleration (negative)
        brake_accel = brake * self.brake_power

        # Aerodynamic drag acceleration (quadratic with velocity)
        drag_accel = self.drag_coefficient * self.velocity ** 2

        # Rolling resistance acceleration
        rolling_accel = self.rolling_resistance * self.velocity

        # Net acceleration
        acceleration = engine_accel - brake_accel - drag_accel - rolling_accel

        # === TRACTION LIMIT ===
        # Grip affects maximum acceleration/deceleration
        max_traction = grip_multiplier * 40.0  # m/s² with perfect grip
        acceleration = np.clip(acceleration, -max_traction, max_traction)

        # === UPDATE VELOCITY ===
        self.velocity += acceleration * dt
        self.velocity = np.clip(self.velocity, self.min_speed, self.max_speed)

        # === BICYCLE MODEL FOR ROTATION ===
        # Only turn if moving and steering
        if abs(self.steering_angle) > 0.001 and self.velocity > 0.5:
            # Calculate turning radius: R = L / tan(δ)
            turning_radius = self.wheelbase / np.tan(self.steering_angle)

            # Angular velocity: ω = v / R
            angular_velocity = self.velocity / turning_radius

            # Grip affects turning ability (understeer with low grip)
            angular_velocity *= grip_multiplier

            # Update heading
            self.heading += angular_velocity * dt

        # Normalize heading to [-π, π]
        self.heading = np.arctan2(np.sin(self.heading), np.cos(self.heading))

        # === UPDATE POSITION ===
        # Velocity is always in the direction of heading
        dx = self.velocity * np.cos(self.heading) * dt
        dy = self.velocity * np.sin(self.heading) * dt
        self.position += np.array([dx, dy])

    def get_lateral_force(self) -> float:
        """
        Calculate normalized lateral force [0, 1] for tire wear calculation.

        Lateral force increases with:
        - Steering angle
        - Speed (cornering faster = more lateral force)

        Note: Uses linear speed dependency (not quadratic) for better tyre wear
        dynamics. While physically lateral G ~ v^2, tyre wear is more directly
        related to slip angle which has a more linear relationship with speed.

        Returns:
            Normalized lateral force in range [0, 1]
        """
        if self.velocity < 1.0:
            return 0.0

        # Tyre wear-oriented lateral force
        # Uses linear speed dependency for more aggressive wear at moderate speeds
        speed_normalized = self.velocity / self.max_speed
        steering_normalized = abs(self.steering_angle / self.max_steering)

        # Linear combination: steering contributes 60%, speed contributes 40%
        # This gives meaningful wear even at moderate speeds
        lateral_force = steering_normalized * (0.6 + 0.4 * speed_normalized)

        return np.clip(lateral_force, 0.0, 1.0)

    def get_state(self) -> Dict:
        """
        Get the current state of the car.

        Returns:
            Dictionary containing:
                - position: np.array([x, y])
                - velocity: float (speed magnitude in m/s)
                - velocity_kmh: float (speed in km/h)
                - heading: float (orientation in radians)
                - steering_angle: float (current wheel angle)
                - lateral_force: float (normalized lateral G)
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity,
            'velocity_kmh': self.velocity * 3.6,
            'heading': self.heading,
            'steering_angle': self.steering_angle,
            'lateral_force': self.get_lateral_force()
        }

    def get_corners(self) -> np.ndarray:
        """
        Calculate the four corner positions of the car for collision detection.

        The car is modeled as a rectangle with dimensions length × width.
        Corners are returned in clockwise order starting from front-right.

        Returns:
            Array of shape (4, 2) with corner positions in world coordinates:
                [[front_right_x, front_right_y],
                 [front_left_x, front_left_y],
                 [rear_left_x, rear_left_y],
                 [rear_right_x, rear_right_y]]
        """
        half_length = self.length / 2.0
        half_width = self.width / 2.0

        # Define corners in local car frame (origin at center)
        # Front of car is +X direction in local frame
        local_corners = np.array([
            [half_length, -half_width],   # Front-right
            [half_length, half_width],    # Front-left
            [-half_length, half_width],   # Rear-left
            [-half_length, -half_width],  # Rear-right
        ])

        # Rotation matrix for heading angle
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        rotation_matrix = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])

        # Rotate corners and translate to world position
        world_corners = (rotation_matrix @ local_corners.T).T + self.position

        return world_corners

    def get_front_position(self) -> np.ndarray:
        """
        Get the position of the front center of the car.

        This is useful for:
            - Raycasting sensors (LIDAR mounted on front)
            - Collision detection
            - Visual feedback

        Returns:
            np.array([x, y]) position of the front center in world coordinates
        """
        front_offset = self.length / 2.0
        front_x = self.position[0] + front_offset * np.cos(self.heading)
        front_y = self.position[1] + front_offset * np.sin(self.heading)
        return np.array([front_x, front_y])
