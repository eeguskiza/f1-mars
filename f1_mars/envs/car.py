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
        - Wheelbase: 4.0 meters (distance between front and rear axles)
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
        self.wheelbase = 4.0  # Distance between axles (meters)
        self.length = 4.5  # Overall car length (meters)
        self.width = 2.0  # Overall car width (meters)

        # Physical parameters (can be overridden via config)
        self.max_speed = 300.0  # Maximum velocity (m/s)
        self.max_steering = 0.6  # Maximum steering angle (radians, ~34 degrees)
        self.acceleration_power = 150.0  # Forward acceleration (m/s²)
        self.brake_power = 300.0  # Braking deceleration (m/s²)
        self.drag_coefficient = 0.02  # Drag factor (dimensionless)
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

        Physics Pipeline:
            1. Smoothly interpolate steering angle toward input
            2. Calculate longitudinal acceleration from throttle/brake
            3. Apply aerodynamic drag
            4. Update velocity (clamped to valid range)
            5. Calculate turning radius and angular velocity
            6. Update heading based on steering and grip
            7. Update position based on velocity and heading

        Args:
            dt: Time step in seconds (typically 1/60 or smaller)
            throttle: Throttle pedal input, range [0, 1]
            brake: Brake pedal input, range [0, 1]
            steering_input: Steering wheel input, range [-1, 1]
                           (negative = left, positive = right)
            grip_multiplier: Tire grip factor, range [0, 1]
                           (1.0 = full grip, lower = reduced grip from tire wear)
        """
        # 1. Update steering angle with smooth interpolation (not instantaneous)
        target_steering = steering_input * self.max_steering
        max_steering_change = self.steering_speed * dt
        steering_delta = np.clip(
            target_steering - self.steering_angle,
            -max_steering_change,
            max_steering_change
        )
        self.steering_angle += steering_delta

        # 2. Calculate longitudinal acceleration
        # Throttle adds forward acceleration, brake adds negative acceleration
        acceleration = throttle * self.acceleration_power - brake * self.brake_power

        # 3. Apply aerodynamic drag (proportional to velocity)
        drag_force = self.velocity * self.drag_coefficient
        acceleration -= drag_force

        # 4. Update velocity
        self.velocity += acceleration * dt
        # Clamp velocity to valid range [0, max_speed]
        self.velocity = np.clip(self.velocity, 0.0, self.max_speed)

        # 5. Apply bicycle model for rotation
        # Only turn if we're moving and steering
        if abs(self.steering_angle) > 1e-6 and self.velocity > 0.1:
            # Calculate turning radius using bicycle model
            # R = L / tan(δ), where L = wheelbase, δ = steering angle
            turning_radius = self.wheelbase / np.tan(self.steering_angle)

            # Angular velocity ω = v / R
            angular_velocity = self.velocity / turning_radius

            # Apply grip multiplier (simulates tire slip/wear)
            angular_velocity *= grip_multiplier

            # 6. Update heading
            self.heading += angular_velocity * dt
            # Normalize heading to [-π, π]
            self.heading = np.arctan2(np.sin(self.heading), np.cos(self.heading))

        # 7. Update position based on velocity and heading
        # Velocity is always in the direction of heading (no lateral slip in this model)
        dx = self.velocity * np.cos(self.heading) * dt
        dy = self.velocity * np.sin(self.heading) * dt
        self.position += np.array([dx, dy])

    def get_state(self) -> Dict:
        """
        Get the current state of the car.

        Returns:
            Dictionary containing:
                - position: np.array([x, y])
                - velocity: float (speed magnitude)
                - heading: float (orientation in radians)
                - steering_angle: float (current wheel angle)
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity,
            'heading': self.heading,
            'steering_angle': self.steering_angle
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
