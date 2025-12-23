"""Tests for geometry utility functions."""

import pytest
import numpy as np
from f1_mars.utils.geometry import (
    rotate_point,
    line_line_intersection,
    line_segment_distance,
    normalize_angle,
    distance_2d,
)


class TestRotatePoint:
    """Tests for rotate_point function."""

    def test_rotate_90_degrees(self):
        """Test rotating a point 90 degrees."""
        point = (1.0, 0.0)
        origin = (0.0, 0.0)
        angle = np.pi / 2  # 90 degrees

        result = rotate_point(point, angle, origin)

        # Should be approximately (0, 1)
        assert abs(result[0]) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10

    def test_rotate_180_degrees(self):
        """Test rotating a point 180 degrees."""
        point = (1.0, 0.0)
        origin = (0.0, 0.0)
        angle = np.pi  # 180 degrees

        result = rotate_point(point, angle, origin)

        # Should be approximately (-1, 0)
        assert abs(result[0] + 1.0) < 1e-10
        assert abs(result[1]) < 1e-10

    def test_rotate_around_offset_center(self):
        """Test rotating around a non-origin center."""
        point = (2.0, 1.0)
        origin = (1.0, 1.0)
        angle = np.pi / 2

        result = rotate_point(point, angle, origin)

        # Point should rotate around (1, 1)
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 2.0) < 1e-10


class TestLineIntersection:
    """Tests for line_line_intersection function."""

    def test_intersecting_lines(self):
        """Test two lines that intersect."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([2.0, 2.0])
        p3 = np.array([0.0, 2.0])
        p4 = np.array([2.0, 0.0])

        result = line_line_intersection(p1, p2, p3, p4)

        assert result is not None
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10

    def test_parallel_lines(self):
        """Test parallel lines that don't intersect."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([0.0, 1.0])
        p4 = np.array([1.0, 1.0])

        result = line_line_intersection(p1, p2, p3, p4)

        assert result is None


class TestLineSegmentDistance:
    """Tests for line_segment_distance function."""

    def test_perpendicular_distance(self):
        """Test perpendicular distance to a line segment."""
        point = np.array([1.0, 1.0])
        segment_start = np.array([0.0, 0.0])
        segment_end = np.array([2.0, 0.0])

        dist, closest = line_segment_distance(point, segment_start, segment_end)

        assert abs(dist - 1.0) < 1e-10

    def test_distance_to_endpoint(self):
        """Test distance when closest point is an endpoint."""
        point = np.array([3.0, 1.0])
        segment_start = np.array([0.0, 0.0])
        segment_end = np.array([2.0, 0.0])

        dist, closest = line_segment_distance(point, segment_start, segment_end)

        # Distance to (2, 0)
        expected = np.sqrt((3.0 - 2.0)**2 + (1.0 - 0.0)**2)
        assert abs(dist - expected) < 1e-10


class TestNormalizeAngle:
    """Tests for normalize_angle function."""

    def test_angle_within_range(self):
        """Test angle already in range."""
        angle = np.pi / 2
        result = normalize_angle(angle)
        assert abs(result - angle) < 1e-10

    def test_angle_above_range(self):
        """Test angle above pi."""
        angle = 2 * np.pi + 0.5
        result = normalize_angle(angle)
        assert abs(result - 0.5) < 1e-10

    def test_angle_below_range(self):
        """Test angle below -pi."""
        angle = -2 * np.pi - 0.5
        result = normalize_angle(angle)
        assert abs(result + 0.5) < 1e-10


class TestDistance:
    """Tests for distance_2d function."""

    def test_distance_horizontal(self):
        """Test horizontal distance."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 0.0])
        result = distance_2d(p1, p2)
        assert abs(result - 3.0) < 1e-10

    def test_distance_diagonal(self):
        """Test diagonal distance (3-4-5 triangle)."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        result = distance_2d(p1, p2)
        assert abs(result - 5.0) < 1e-10
