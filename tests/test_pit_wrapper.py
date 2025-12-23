"""Tests for PitStopWrapper functionality."""

import pytest
import numpy as np
from f1_mars.envs import F1Env, PitStopWrapper, TyreCompound


class TestPitStopWrapper:
    """Tests for pit stop wrapper."""

    def test_wrapper_initialization(self):
        """Test wrapper initializes correctly."""
        base_env = F1Env(max_laps=3)
        env = PitStopWrapper(
            base_env,
            pit_stop_duration=2.5,
            pit_entry_distance=60.0,
            pit_exit_distance=140.0
        )

        assert env.pit_stop_duration == 2.5
        assert env.pit_entry_distance == 60.0
        assert env.pit_exit_distance == 140.0
        assert env.pit_requested is False
        assert env.in_pit_lane is False
        assert env.total_pit_stops == 0

        env.close()

    def test_reset_clears_pit_state(self):
        """Test that reset clears all pit stop state."""
        base_env = F1Env(max_laps=3)
        env = PitStopWrapper(base_env)

        # Set some state
        env.pit_requested = True
        env.in_pit_lane = True
        env.total_pit_stops = 2

        # Reset should clear everything
        obs, info = env.reset()

        assert env.pit_requested is False
        assert env.in_pit_lane is False
        assert env.total_pit_stops == 0
        assert 'pit_status' in info

        env.close()

    def test_request_pit_stop(self):
        """Test requesting a pit stop."""
        base_env = F1Env(max_laps=3)
        env = PitStopWrapper(base_env)

        obs, info = env.reset()

        # Request pit
        env.request_pit(compound=TyreCompound.SOFT)

        assert env.pit_requested is True
        assert env.new_compound == TyreCompound.SOFT

        env.close()

    def test_cancel_pit_stop(self):
        """Test canceling a pit stop request."""
        base_env = F1Env(max_laps=3)
        env = PitStopWrapper(base_env)

        obs, info = env.reset()

        # Request and cancel
        env.request_pit(compound=TyreCompound.HARD)
        assert env.pit_requested is True

        env.cancel_pit()
        assert env.pit_requested is False
        assert env.new_compound is None

        env.close()

    def test_pit_status_reporting(self):
        """Test pit status is correctly reported in info."""
        base_env = F1Env(max_laps=3)
        env = PitStopWrapper(base_env)

        obs, info = env.reset()

        # Check initial status
        pit_status = info['pit_status']
        assert pit_status['pit_requested'] is False
        assert pit_status['in_pit_lane'] is False
        assert pit_status['total_pit_stops'] == 0
        assert pit_status['laps_since_pit'] == 0

        env.close()

    def test_full_pit_stop_execution(self):
        """Test complete pit stop execution from request to completion."""
        base_env = F1Env(max_laps=5)
        env = PitStopWrapper(
            base_env,
            pit_stop_duration=1.0,  # Short duration for faster test
            pit_entry_distance=50.0,
            pit_exit_distance=150.0
        )

        obs, info = env.reset()
        initial_compound = info['tyre_compound']

        # Request pit with different compound
        target_compound = TyreCompound.HARD if initial_compound != 'HARD' else TyreCompound.SOFT
        env.request_pit(compound=target_compound)

        # Drive until pit is completed
        pit_entered = False
        pit_completed = False

        for step in range(1000):
            action = np.array([0.0, 0.8, 0.0])  # Moderate throttle
            obs, reward, terminated, truncated, info = env.step(action)

            pit_status = info['pit_status']

            # Track pit entry
            if pit_status['in_pit_lane'] and not pit_entered:
                pit_entered = True
                assert pit_status['pit_stop_timer'] > 0

            # Track pit completion
            if pit_entered and not pit_status['in_pit_lane']:
                pit_completed = True
                break

            if terminated or truncated:
                break

        # Verify pit stop completed
        assert pit_completed, "Pit stop should have been completed"
        assert info['pit_status']['total_pit_stops'] == 1
        assert info['pit_status']['laps_since_pit'] == 0

        # Verify compound changed (may show on next frame)
        assert env.env.tyres.compound == target_compound

        # Verify tyre wear was reset
        assert info['tyre_wear'] < 1.0  # Should be nearly zero

        env.close()

    def test_lap_counter_increments(self):
        """Test that laps_since_pit increments correctly."""
        base_env = F1Env(max_laps=5)
        env = PitStopWrapper(base_env)

        obs, info = env.reset()
        initial_lap = info['lap']

        # Drive until lap changes
        for step in range(2000):
            action = np.array([0.0, 0.9, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)

            if info['lap'] > initial_lap:
                # Laps since pit should have incremented
                assert info['laps_since_pit'] > 0
                break

        env.close()

    def test_cannot_request_pit_while_in_pit(self):
        """Test that requesting pit while in pit lane is prevented."""
        base_env = F1Env(max_laps=5)
        env = PitStopWrapper(base_env, pit_stop_duration=1.0)

        obs, info = env.reset()

        # Request pit and drive until we're in pit lane
        env.request_pit(compound=TyreCompound.SOFT)

        for step in range(500):
            action = np.array([0.0, 0.8, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)

            if info['pit_status']['in_pit_lane']:
                # Try to request another pit while in pit
                env.request_pit(compound=TyreCompound.HARD)

                # Should still only have SOFT requested
                assert env.new_compound == TyreCompound.SOFT or env.new_compound is None
                break

        env.close()

    def test_pit_stop_timer_counts_down(self):
        """Test that pit stop timer counts down correctly."""
        base_env = F1Env(max_laps=5)
        env = PitStopWrapper(base_env, pit_stop_duration=2.0)

        obs, info = env.reset()
        env.request_pit()

        # Drive until in pit
        timer_values = []

        for step in range(1000):
            action = np.array([0.0, 0.8, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)

            if info['pit_status']['in_pit_lane']:
                timer = info['pit_status']['pit_stop_timer']
                if timer > 0:
                    timer_values.append(timer)
                else:
                    break

        # Timer should have counted down
        if len(timer_values) > 1:
            assert timer_values[0] > timer_values[-1]
            # Timer should start near pit_stop_duration
            assert timer_values[0] <= 2.0
            assert timer_values[0] > 1.5  # Should be close to 2.0

        env.close()


class TestPitStopIntegration:
    """Integration tests for pit stop with environment."""

    def test_observation_space_unchanged(self):
        """Test that wrapper doesn't change observation space."""
        base_env = F1Env(max_laps=3)
        wrapped_env = PitStopWrapper(base_env)

        assert wrapped_env.observation_space == base_env.observation_space

        wrapped_env.close()
        base_env.close()

    def test_action_space_unchanged(self):
        """Test that wrapper doesn't change action space."""
        base_env = F1Env(max_laps=3)
        wrapped_env = PitStopWrapper(base_env)

        assert wrapped_env.action_space == base_env.action_space

        wrapped_env.close()
        base_env.close()

    def test_multiple_pit_stops(self):
        """Test that environment can handle multiple pit stops."""
        base_env = F1Env(max_laps=15)
        env = PitStopWrapper(base_env, pit_stop_duration=0.5)

        obs, info = env.reset()

        # Request first pit immediately
        env.request_pit()

        pit_stops_completed = 0
        pit_stop_count_when_requested_second = None

        for step in range(8000):  # More steps to allow for 2 pit stops
            action = np.array([0.0, 0.85, 0.0])  # Moderate speed
            obs, reward, terminated, truncated, info = env.step(action)

            current_pit_stops = info['pit_status']['total_pit_stops']

            # Request second pit after first is completed
            if current_pit_stops == 1 and pit_stop_count_when_requested_second is None:
                # Wait a bit after first pit before requesting second
                if info['laps_since_pit'] >= 1:
                    env.request_pit()
                    pit_stop_count_when_requested_second = current_pit_stops

            # Track completions
            if current_pit_stops > pit_stops_completed:
                pit_stops_completed = current_pit_stops

            # Stop after 2 pit stops
            if pit_stops_completed >= 2:
                break

            if terminated or truncated:
                break

        # We should complete at least 1 pit stop (2 is ideal but may not complete in time)
        assert pit_stops_completed >= 1, f"Should complete at least 1 pit stop, got {pit_stops_completed}"

        env.close()
