"""Geometric utility functions for F1 Mars simulator."""

import numpy as np
from typing import Tuple, List, Optional, Union


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to the range [-π, π].

    Uses atan2 trick for efficient normalization without loops.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in range [-π, π]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the minimum signed difference between two angles.

    The result is in range [-π, π] and represents the shortest rotation
    from angle1 to angle2. Positive means counter-clockwise.

    Args:
        angle1: First angle in radians
        angle2: Second angle in radians

    Returns:
        Signed difference (angle2 - angle1) in range [-π, π]
    """
    diff = angle2 - angle1
    return normalize_angle(diff)


def rotate_point(
    point: Union[np.ndarray, Tuple[float, float]],
    angle: float,
    origin: Optional[Union[np.ndarray, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Rotate a point around an origin by given angle.

    Args:
        point: Point to rotate as np.array([x, y]) or (x, y)
        angle: Rotation angle in radians (counter-clockwise positive)
        origin: Center of rotation, defaults to (0, 0)

    Returns:
        Rotated point as np.array([x, y])
    """
    # Convert to numpy arrays
    point = np.asarray(point, dtype=np.float64)

    if origin is None:
        origin = np.array([0.0, 0.0])
    else:
        origin = np.asarray(origin, dtype=np.float64)

    # Translate to origin
    translated = point - origin

    # Rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Apply rotation
    rotated = np.array([
        translated[0] * cos_a - translated[1] * sin_a,
        translated[0] * sin_a + translated[1] * cos_a
    ])

    # Translate back
    return rotated + origin


def line_segment_distance(
    point: Union[np.ndarray, Tuple[float, float]],
    line_start: Union[np.ndarray, Tuple[float, float]],
    line_end: Union[np.ndarray, Tuple[float, float]]
) -> Tuple[float, np.ndarray]:
    """
    Calculate distance from point to line segment and find closest point.

    Args:
        point: Query point as np.array([x, y]) or (x, y)
        line_start: Start of line segment
        line_end: End of line segment

    Returns:
        Tuple of (distance, closest_point) where:
            - distance: Perpendicular distance to segment
            - closest_point: Nearest point on the segment as np.array([x, y])
    """
    # Convert to numpy arrays
    point = np.asarray(point, dtype=np.float64)
    line_start = np.asarray(line_start, dtype=np.float64)
    line_end = np.asarray(line_end, dtype=np.float64)

    # Vector from start to end
    line_vec = line_end - line_start
    line_length_sq = np.dot(line_vec, line_vec)

    # Handle degenerate case (segment is a point)
    if line_length_sq < 1e-10:
        closest_point = line_start.copy()
        distance = np.linalg.norm(point - closest_point)
        return distance, closest_point

    # Project point onto line
    # t = dot(point - start, line_vec) / ||line_vec||^2
    t = np.dot(point - line_start, line_vec) / line_length_sq

    # Clamp t to [0, 1] to stay on segment
    t = np.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest_point = line_start + t * line_vec

    # Distance
    distance = np.linalg.norm(point - closest_point)

    return distance, closest_point


def raycast(
    origin: Union[np.ndarray, Tuple[float, float]],
    direction: float,
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    max_distance: float
) -> float:
    """
    Cast a ray from origin in given direction and find first intersection.

    This is optimized for multiple calls per frame (LIDAR sensors).
    Uses vectorized operations where possible.

    Args:
        origin: Ray origin as np.array([x, y]) or (x, y)
        direction: Ray direction in radians
        segments: List of line segments as [(start, end), ...]
                 where start and end are (x, y) tuples
        max_distance: Maximum ray distance

    Returns:
        Distance to first intersection, or max_distance if no hit
    """
    if not segments:
        return max_distance

    origin = np.asarray(origin, dtype=np.float64)

    # Ray direction vector
    ray_dir = np.array([np.cos(direction), np.sin(direction)])

    # Ray endpoint
    ray_end = origin + max_distance * ray_dir

    min_dist = max_distance

    # Check intersection with each segment
    # This could be further optimized with spatial partitioning for many segments
    for seg_start, seg_end in segments:
        # Line-line intersection using parametric form
        # Ray: P = origin + t * ray_dir
        # Segment: Q = seg_start + s * seg_vec

        seg_start = np.asarray(seg_start, dtype=np.float64)
        seg_end = np.asarray(seg_end, dtype=np.float64)
        seg_vec = seg_end - seg_start

        # Solve: origin + t*ray_dir = seg_start + s*seg_vec
        # In matrix form: [ray_dir | -seg_vec] * [t; s] = seg_start - origin

        # Cross product for determinant
        denom = ray_dir[0] * seg_vec[1] - ray_dir[1] * seg_vec[0]

        # Parallel check
        if abs(denom) < 1e-10:
            continue

        diff = seg_start - origin

        # Solve for t and s using Cramer's rule
        t = (diff[0] * seg_vec[1] - diff[1] * seg_vec[0]) / denom
        s = (diff[0] * ray_dir[1] - diff[1] * ray_dir[0]) / denom

        # Check if intersection is valid
        # t >= 0: intersection is in front of origin
        # s in [0, 1]: intersection is on the segment
        if t >= 0 and 0 <= s <= 1:
            dist = t
            if dist < min_dist:
                min_dist = dist

    return min_dist


def raycast_batch(
    origin: Union[np.ndarray, Tuple[float, float]],
    directions: np.ndarray,
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    max_distance: float
) -> np.ndarray:
    """
    Cast multiple rays at once (optimized for LIDAR with many rays).

    Args:
        origin: Ray origin
        directions: Array of ray directions in radians, shape (N,)
        segments: List of line segments
        max_distance: Maximum ray distance

    Returns:
        Array of distances, shape (N,)
    """
    num_rays = len(directions)
    distances = np.full(num_rays, max_distance, dtype=np.float64)

    for i, direction in enumerate(directions):
        distances[i] = raycast(origin, direction, segments, max_distance)

    return distances


def polyline_length(points: np.ndarray) -> float:
    """
    Calculate total length of a polyline (connected line segments).

    Args:
        points: Array of shape (N, 2) representing N points

    Returns:
        Total length of the polyline
    """
    if len(points) < 2:
        return 0.0

    # Vectorized calculation of segment lengths
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)

    return np.sum(segment_lengths)


def interpolate_along_polyline(
    points: np.ndarray,
    distance: float
) -> Tuple[np.ndarray, int, float]:
    """
    Find point at given distance along a polyline.

    Args:
        points: Array of shape (N, 2) representing polyline vertices
        distance: Distance from start of polyline

    Returns:
        Tuple of:
            - interpolated_point: np.array([x, y]) at the distance
            - segment_index: Index of the segment containing the point
            - heading: Direction angle at that point (radians)
    """
    if len(points) < 2:
        return points[0], 0, 0.0

    # Calculate cumulative distances
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])

    total_length = cumulative_distances[-1]

    # Wrap distance to valid range
    distance = distance % total_length if total_length > 0 else 0

    # Find segment containing the distance
    segment_index = np.searchsorted(cumulative_distances, distance, side='right') - 1
    segment_index = np.clip(segment_index, 0, len(points) - 2)

    # Distance along this segment
    segment_start_dist = cumulative_distances[segment_index]
    distance_in_segment = distance - segment_start_dist
    segment_length = segment_lengths[segment_index]

    # Interpolation parameter
    if segment_length > 1e-10:
        t = distance_in_segment / segment_length
    else:
        t = 0.0

    # Interpolate position
    p_start = points[segment_index]
    p_end = points[segment_index + 1]
    interpolated_point = p_start + t * (p_end - p_start)

    # Calculate heading (direction of segment)
    segment_vec = p_end - p_start
    heading = np.arctan2(segment_vec[1], segment_vec[0])

    return interpolated_point, segment_index, heading


def point_in_polygon(
    point: Union[np.ndarray, Tuple[float, float]],
    polygon: np.ndarray
) -> bool:
    """
    Test if a point is inside a polygon using ray casting algorithm.

    Args:
        point: Test point as np.array([x, y]) or (x, y)
        polygon: Array of shape (N, 2) representing polygon vertices

    Returns:
        True if point is inside polygon, False otherwise
    """
    point = np.asarray(point, dtype=np.float64)

    if len(polygon) < 3:
        return False

    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]

        # Check if point is on horizontal line segment
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_intersection:
                            inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def closest_point_on_polyline(
    point: Union[np.ndarray, Tuple[float, float]],
    polyline: np.ndarray
) -> Tuple[np.ndarray, float, int]:
    """
    Find the closest point on a polyline to a given point.

    Args:
        point: Query point
        polyline: Array of shape (N, 2) representing polyline

    Returns:
        Tuple of:
            - closest_point: Nearest point on polyline
            - distance: Distance to that point
            - segment_index: Index of closest segment
    """
    point = np.asarray(point, dtype=np.float64)

    if len(polyline) < 2:
        return polyline[0], np.linalg.norm(point - polyline[0]), 0

    min_distance = float('inf')
    closest_point = None
    closest_segment = 0

    for i in range(len(polyline) - 1):
        dist, pt = line_segment_distance(point, polyline[i], polyline[i + 1])
        if dist < min_distance:
            min_distance = dist
            closest_point = pt
            closest_segment = i

    return closest_point, min_distance, closest_segment


def circle_line_intersection(
    circle_center: Union[np.ndarray, Tuple[float, float]],
    circle_radius: float,
    line_start: Union[np.ndarray, Tuple[float, float]],
    line_end: Union[np.ndarray, Tuple[float, float]]
) -> bool:
    """
    Test if a line segment intersects with a circle.

    Useful for collision detection between circular objects and track boundaries.

    Args:
        circle_center: Center of circle
        circle_radius: Radius of circle
        line_start: Start of line segment
        line_end: End of line segment

    Returns:
        True if line segment intersects circle
    """
    distance, _ = line_segment_distance(circle_center, line_start, line_end)
    return distance <= circle_radius


def signed_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate signed angle from vector v1 to v2.

    Args:
        v1: First vector as np.array([x, y])
        v2: Second vector as np.array([x, y])

    Returns:
        Signed angle in radians [-π, π]
    """
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    return angle_difference(angle1, angle2)


def perpendicular_vector(v: np.ndarray, clockwise: bool = False) -> np.ndarray:
    """
    Get perpendicular vector to given vector.

    Args:
        v: Input vector as np.array([x, y])
        clockwise: If True, rotate 90° clockwise, else counter-clockwise

    Returns:
        Perpendicular vector
    """
    if clockwise:
        return np.array([v[1], -v[0]])
    else:
        return np.array([-v[1], v[0]])


def line_line_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two line segments.

    Args:
        p1, p2: Endpoints of first line segment
        p3, p4: Endpoints of second line segment

    Returns:
        Intersection point (x, y) or None if no intersection
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    return None


def distance_2d(p1: Union[np.ndarray, Tuple[float, float]],
                p2: Union[np.ndarray, Tuple[float, float]]) -> float:
    """
    Calculate Euclidean distance between two 2D points.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Euclidean distance
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    return np.linalg.norm(p2 - p1)
