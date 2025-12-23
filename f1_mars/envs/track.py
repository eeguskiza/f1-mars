"""Track system for F1 Mars simulator with spline-based centerline."""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import cumulative_trapezoid


class Track:
    """
    Racing track defined by a smooth spline centerline with variable width.

    The track is represented as a closed loop with:
    - Centerline: Smooth cubic spline through control points
    - Width: Variable or constant width at each point
    - Checkpoints: Progress markers along the track
    - Pit lane: Optional pit stop area

    Distance along track is measured from the start position along the centerline.
    """

    def __init__(self, track_path: str):
        """
        Load track from JSON file and compute spline representation.

        Args:
            track_path: Path to JSON file or track name (searches in TRACKS_DIR)

        JSON Format:
            {
                "name": "Circuit Name",
                "centerline": [[x1, y1], [x2, y2], ...],
                "widths": [w1, w2, ...] or single_width,
                "pit_entry": [x, y],
                "pit_exit": [x, y],
                "pit_lane": [[x1, y1], [x2, y2], ...],
                "start_position": [x, y, heading],
                "checkpoints": [0, 15, 30, ...],  // indices in centerline
                "total_laps": 5
            }
        """
        # Load track data from file
        track_path = Path(track_path)
        if not track_path.suffix:
            # Try adding .json and looking in TRACKS_DIR
            track_path = Path("tracks") / f"{track_path}.json"

        if not track_path.exists():
            raise FileNotFoundError(f"Track file not found: {track_path}")

        with open(track_path, 'r') as f:
            data = json.load(f)

        self.load_from_dict(data)

    def load_from_dict(self, data: Dict):
        """
        Load track from dictionary (useful for procedural generation).

        Args:
            data: Dictionary with track definition (see __init__ for format)
        """
        # Basic properties
        self.name = data.get("name", "Unnamed Circuit")
        self.total_laps = data.get("total_laps", 5)

        # Centerline points (ensure it's a closed loop)
        centerline_points = np.array(data["centerline"], dtype=np.float64)
        if centerline_points.ndim != 2 or centerline_points.shape[1] != 2:
            raise ValueError("Centerline must be Nx2 array of [x, y] points")

        # Ensure closed loop (last point connects to first)
        if not np.allclose(centerline_points[0], centerline_points[-1]):
            centerline_points = np.vstack([centerline_points, centerline_points[0]])

        self.control_points = centerline_points
        self.num_control_points = len(self.control_points)

        # Width(s) - can be single value or array
        widths_data = data.get("widths", data.get("width", 80.0))
        if isinstance(widths_data, (int, float)):
            self.widths = np.full(self.num_control_points, widths_data, dtype=np.float64)
        else:
            self.widths = np.array(widths_data, dtype=np.float64)
            if len(self.widths) != self.num_control_points:
                raise ValueError("Number of widths must match number of centerline points")

        # Start position
        start_data = data.get("start_position", [centerline_points[0, 0],
                                                   centerline_points[0, 1], 0.0])
        self.start_x = start_data[0]
        self.start_y = start_data[1]
        self.start_heading = start_data[2] if len(start_data) > 2 else 0.0

        # Checkpoints (indices in control points)
        self.checkpoint_indices = data.get("checkpoints", list(range(0, self.num_control_points, 5)))

        # Pit lane (optional)
        self.pit_entry = data.get("pit_entry", None)
        self.pit_exit = data.get("pit_exit", None)
        self.pit_lane = data.get("pit_lane", None)

        # Compute spline representation
        self._build_spline()

    def _build_spline(self):
        """
        Build cubic spline interpolation of centerline and precompute track length.

        Creates smooth interpolation through control points and calculates
        cumulative arc length for distance-based queries.
        """
        # Parameter t for each control point (0 to num_points-1)
        t_control = np.arange(self.num_control_points, dtype=np.float64)

        # Create periodic cubic splines for x(t) and y(t)
        # Periodic boundary conditions ensure smooth closed loop
        self.spline_x = CubicSpline(t_control, self.control_points[:, 0],
                                     bc_type='periodic')
        self.spline_y = CubicSpline(t_control, self.control_points[:, 1],
                                     bc_type='periodic')

        # Create interpolator for width
        self.spline_width = interp1d(t_control, self.widths, kind='linear',
                                      fill_value='extrapolate', assume_sorted=True)

        # Compute arc length parameterization
        # Sample the spline densely to compute distances
        num_samples = self.num_control_points * 100  # Dense sampling
        t_dense = np.linspace(0, self.num_control_points - 1, num_samples)

        x_dense = self.spline_x(t_dense)
        y_dense = self.spline_y(t_dense)

        # Compute derivatives (velocities)
        dx_dense = self.spline_x.derivative()(t_dense)
        dy_dense = self.spline_y.derivative()(t_dense)

        # Speed at each sample point: ds/dt = sqrt((dx/dt)^2 + (dy/dt)^2)
        speed = np.sqrt(dx_dense**2 + dy_dense**2)

        # Cumulative arc length using trapezoidal integration
        # distances[i] = total arc length from start to sample i
        dt = t_dense[1] - t_dense[0]
        distances = np.concatenate([[0], cumulative_trapezoid(speed, dx=dt)])

        self.total_length = distances[-1]

        # Create inverse mapping: distance -> parameter t
        # This allows us to query the spline by arc length distance
        self.distance_to_t = interp1d(distances, t_dense, kind='cubic',
                                       fill_value='extrapolate', assume_sorted=True)

        # Precompute checkpoint distances
        self.checkpoint_distances = []
        for idx in self.checkpoint_indices:
            # Find distance to this control point
            t_checkpoint = float(idx)
            # Find closest sample to this t value
            closest_sample = np.argmin(np.abs(t_dense - t_checkpoint))
            self.checkpoint_distances.append(distances[closest_sample])

    def get_point_at_distance(self, distance: float) -> np.ndarray:
        """
        Get point on centerline at given distance from start.

        Args:
            distance: Arc length distance from start (wraps around for closed loop)

        Returns:
            np.array([x, y]) - Position on centerline
        """
        # Wrap distance to [0, total_length)
        distance = distance % self.total_length

        # Convert distance to parameter t
        t = float(self.distance_to_t(distance))

        # Wrap t to valid range
        t = t % self.num_control_points

        # Evaluate splines
        x = float(self.spline_x(t))
        y = float(self.spline_y(t))

        return np.array([x, y])

    def get_direction_at_distance(self, distance: float) -> float:
        """
        Get track heading (direction) at given distance.

        Args:
            distance: Arc length distance from start

        Returns:
            Heading angle in radians (direction of track tangent)
        """
        distance = distance % self.total_length
        t = float(self.distance_to_t(distance))
        t = t % self.num_control_points

        # Get derivatives (tangent vector)
        dx = float(self.spline_x.derivative()(t))
        dy = float(self.spline_y.derivative()(t))

        # Heading is arctan2(dy, dx)
        return np.arctan2(dy, dx)

    def get_width_at_distance(self, distance: float) -> float:
        """
        Get track width at given distance.

        Args:
            distance: Arc length distance from start

        Returns:
            Track width at that position
        """
        distance = distance % self.total_length
        t = float(self.distance_to_t(distance))
        t = t % self.num_control_points

        return float(self.spline_width(t))

    def get_closest_point_on_track(self, position: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Find closest point on track centerline to given position.

        Uses numerical search along the track to find minimum distance point.

        Args:
            position: Query position [x, y]

        Returns:
            Tuple of:
                - closest_point: np.array([x, y]) on centerline
                - distance_along_track: Arc length from start to closest point
                - lateral_offset: Perpendicular distance (positive = right of centerline)
                - track_heading: Direction of track at closest point
        """
        # Sample track at fine resolution
        num_samples = self.num_control_points * 20
        distances = np.linspace(0, self.total_length, num_samples)

        # Compute all sample points
        points = np.array([self.get_point_at_distance(d) for d in distances])

        # Find closest sample
        dists_to_pos = np.linalg.norm(points - position, axis=1)
        closest_idx = np.argmin(dists_to_pos)

        # Refine using nearby samples (local search)
        search_start = max(0, closest_idx - 5)
        search_end = min(num_samples, closest_idx + 6)
        local_distances = distances[search_start:search_end]

        # Find exact minimum in local region
        def distance_to_position(d):
            pt = self.get_point_at_distance(d)
            return np.linalg.norm(pt - position)

        # Try each local sample
        best_distance = distances[closest_idx]
        best_dist_value = distance_to_position(best_distance)

        for d in local_distances:
            dist_value = distance_to_position(d)
            if dist_value < best_dist_value:
                best_dist_value = dist_value
                best_distance = d

        # Get final closest point
        closest_point = self.get_point_at_distance(best_distance)
        track_heading = self.get_direction_at_distance(best_distance)

        # Calculate lateral offset (signed distance perpendicular to track)
        # Positive = to the right of centerline (in direction of travel)
        to_position = position - closest_point

        # Track tangent direction
        track_tangent = np.array([np.cos(track_heading), np.sin(track_heading)])

        # Right perpendicular (rotate tangent 90° clockwise)
        track_right = np.array([track_tangent[1], -track_tangent[0]])

        # Project displacement onto right direction
        lateral_offset = np.dot(to_position, track_right)

        return closest_point, best_distance, lateral_offset, track_heading

    def is_on_track(self, position: np.ndarray) -> bool:
        """
        Check if position is within track boundaries.

        Args:
            position: Position to check [x, y]

        Returns:
            True if position is on track, False if off track
        """
        _, distance_along, lateral_offset, _ = self.get_closest_point_on_track(position)
        width_at_point = self.get_width_at_distance(distance_along)

        # On track if lateral distance is less than half width
        return abs(lateral_offset) <= width_at_point / 2.0

    def get_checkpoint_index(self, distance_along_track: float) -> int:
        """
        Get index of the last checkpoint passed at given distance.

        Args:
            distance_along_track: Current distance along track

        Returns:
            Index of last checkpoint passed (-1 if none)
        """
        distance_along_track = distance_along_track % self.total_length

        last_checkpoint = -1
        for i, checkpoint_dist in enumerate(self.checkpoint_distances):
            if distance_along_track >= checkpoint_dist:
                last_checkpoint = i
            else:
                break

        return last_checkpoint

    def get_curvature_at_distance(self, distance: float) -> float:
        """
        Calculate track curvature (1/radius) at given distance.

        Curvature κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

        Args:
            distance: Arc length distance from start

        Returns:
            Curvature value (1/radius). Positive for left turns, negative for right.
        """
        distance = distance % self.total_length
        t = float(self.distance_to_t(distance))
        t = t % self.num_control_points

        # First derivatives
        dx = self.spline_x.derivative(1)(t)
        dy = self.spline_y.derivative(1)(t)

        # Second derivatives
        ddx = self.spline_x.derivative(2)(t)
        ddy = self.spline_y.derivative(2)(t)

        # Curvature formula
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2)**1.5

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def get_track_boundaries(self, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample track boundaries for rendering.

        Args:
            num_samples: Number of points to sample along track

        Returns:
            Tuple of (left_boundary, right_boundary), each Nx2 array
        """
        distances = np.linspace(0, self.total_length, num_samples)

        left_boundary = []
        right_boundary = []

        for d in distances:
            center = self.get_point_at_distance(d)
            heading = self.get_direction_at_distance(d)
            width = self.get_width_at_distance(d)

            # Perpendicular direction (left is +90°, right is -90°)
            perp_angle_left = heading + np.pi / 2
            perp_angle_right = heading - np.pi / 2

            half_width = width / 2.0

            left_point = center + half_width * np.array([np.cos(perp_angle_left),
                                                          np.sin(perp_angle_left)])
            right_point = center + half_width * np.array([np.cos(perp_angle_right),
                                                           np.sin(perp_angle_right)])

            left_boundary.append(left_point)
            right_boundary.append(right_point)

        return np.array(left_boundary), np.array(right_boundary)


class TrackGenerator:
    """
    Procedural track generator for creating random or template circuits.
    """

    @staticmethod
    def generate_oval(length: float = 1000.0, width: float = 80.0,
                      aspect_ratio: float = 2.0) -> Dict:
        """
        Generate a simple oval circuit.

        Args:
            length: Approximate total track length
            width: Track width
            aspect_ratio: Ratio of straight length to turn radius

        Returns:
            Dictionary suitable for Track.load_from_dict()
        """
        # Estimate radius from length
        # Oval has 2 straights + 2 semicircles
        # L = 2*straight + 2*π*r
        # With aspect_ratio = straight/(2*r)
        r = length / (2 * np.pi + 2 * aspect_ratio)
        straight = aspect_ratio * 2 * r

        # Create points for oval
        num_points_turn = 20
        num_points_straight = 15

        centerline = []

        # Right straight (bottom)
        for i in range(num_points_straight):
            x = -straight/2 + (i / (num_points_straight - 1)) * straight
            y = -r
            centerline.append([x, y])

        # Right turn
        for i in range(num_points_turn):
            angle = -np.pi/2 + (i / (num_points_turn - 1)) * np.pi
            x = straight/2 + r * np.cos(angle)
            y = r * np.sin(angle)
            centerline.append([x, y])

        # Left straight (top)
        for i in range(num_points_straight):
            x = straight/2 - (i / (num_points_straight - 1)) * straight
            y = r
            centerline.append([x, y])

        # Left turn
        for i in range(num_points_turn):
            angle = np.pi/2 + (i / (num_points_turn - 1)) * np.pi
            x = -straight/2 + r * np.cos(angle)
            y = r * np.sin(angle)
            centerline.append([x, y])

        return {
            "name": "Generated Oval",
            "centerline": centerline,
            "widths": width,
            "start_position": [centerline[0][0], centerline[0][1], 0.0],
            "checkpoints": [0, num_points_straight,
                          num_points_straight + num_points_turn,
                          2*num_points_straight + num_points_turn],
            "total_laps": 5
        }

    @staticmethod
    def generate_random_track(complexity: int = 10, seed: int = 42,
                             size: float = 1000.0, width: float = 80.0) -> Dict:
        """
        Generate a random circuit using Fourier series for smooth curves.

        Args:
            complexity: Number of frequency components (higher = more complex)
            seed: Random seed for reproducibility
            size: Approximate size of track
            width: Track width

        Returns:
            Dictionary suitable for Track.load_from_dict()
        """
        np.random.seed(seed)

        # Generate random Fourier coefficients for smooth closed curve
        # x(θ) = Σ a_n * cos(nθ) + b_n * sin(nθ)
        # y(θ) = Σ c_n * cos(nθ) + d_n * sin(nθ)

        num_harmonics = complexity
        a_coeffs = np.random.randn(num_harmonics) * size / (np.arange(num_harmonics) + 1)
        b_coeffs = np.random.randn(num_harmonics) * size / (np.arange(num_harmonics) + 1)
        c_coeffs = np.random.randn(num_harmonics) * size / (np.arange(num_harmonics) + 1)
        d_coeffs = np.random.randn(num_harmonics) * size / (np.arange(num_harmonics) + 1)

        # Ensure track is roughly circular (DC component)
        a_coeffs[0] = size / 2
        c_coeffs[0] = 0

        # Generate points
        num_points = 50
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)

        centerline = []
        for t in theta:
            x = sum(a_coeffs[n] * np.cos((n+1)*t) + b_coeffs[n] * np.sin((n+1)*t)
                   for n in range(num_harmonics))
            y = sum(c_coeffs[n] * np.cos((n+1)*t) + d_coeffs[n] * np.sin((n+1)*t)
                   for n in range(num_harmonics))
            centerline.append([float(x), float(y)])

        # Generate checkpoints evenly spaced
        checkpoint_spacing = max(1, num_points // 8)
        checkpoints = list(range(0, num_points, checkpoint_spacing))

        return {
            "name": f"Random Circuit (seed={seed})",
            "centerline": centerline,
            "widths": width,
            "start_position": [centerline[0][0], centerline[0][1], 0.0],
            "checkpoints": checkpoints,
            "total_laps": 5
        }
