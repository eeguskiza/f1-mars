"""Pit Stop Wrapper for F1 Mars Environment.

Adds pit stop functionality to the base F1 environment, allowing the engineer
agent to request pit stops for tyre changes and strategic decisions.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional
from f1_mars.envs.tyres import TyreCompound


class PitStopWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds pit stop functionality.

    This wrapper monitors car position and handles:
    - Pit stop requests from engineer agent
    - Pit lane entry/exit detection
    - Pit stop simulation (car stopped for N seconds)
    - Tyre changes during pit stop
    - Pit lane time penalties

    The pit lane is defined as a section of the track near the start/finish line.
    When a pit stop is requested and the car crosses the pit entry, it is
    diverted to the pit lane, stopped for pit_stop_duration, tyres are changed,
    and the car is released at the pit exit.

    Attributes:
        pit_requested (bool): Whether a pit stop has been requested
        in_pit_lane (bool): Whether car is currently in pit lane
        pit_stop_timer (float): Countdown timer during pit stop (seconds)
        pit_stop_duration (float): Total duration of pit stop (seconds)
        laps_since_pit (int): Laps completed since last pit stop
        total_pit_stops (int): Total pit stops completed
        new_compound (TyreCompound): Compound to use after pit stop
    """

    def __init__(
        self,
        env: gym.Env,
        pit_stop_duration: float = 3.0,
        pit_entry_distance: float = 50.0,
        pit_exit_distance: float = 150.0
    ):
        """
        Initialize the pit stop wrapper.

        Args:
            env: Base F1 environment to wrap
            pit_stop_duration: Time car is stationary during pit stop (seconds)
            pit_entry_distance: Distance along track where pit entry is (meters)
            pit_exit_distance: Distance along track where pit exit is (meters)
        """
        super().__init__(env)

        # Pit stop configuration
        self.pit_stop_duration = pit_stop_duration
        self.pit_entry_distance = pit_entry_distance
        self.pit_exit_distance = pit_exit_distance

        # Pit stop state
        self.pit_requested = False
        self.in_pit_lane = False
        self.pit_stop_timer = 0.0
        self.new_compound = None  # Compound to use after pit stop

        # Statistics
        self.laps_since_pit = 0
        self.total_pit_stops = 0
        self.last_lap_number = 0

        # Pit entry detection (hysteresis to avoid double triggering)
        self._crossed_pit_entry = False
        self._crossed_finish_line = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and pit stop state.

        Args:
            seed: Random seed
            options: Reset options (can include 'pit_compound' to start with specific tyres)

        Returns:
            observation: Initial observation
            info: Info dict with pit status added
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset pit stop state
        self.pit_requested = False
        self.in_pit_lane = False
        self.pit_stop_timer = 0.0
        self.new_compound = None
        self.laps_since_pit = 0
        self.total_pit_stops = 0
        self.last_lap_number = info.get('lap', 1)
        self._crossed_pit_entry = False
        self._crossed_finish_line = False

        # Add pit status to info
        info['pit_status'] = self.get_pit_status()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with pit stop logic.

        Process:
        1. If in pit lane and timer > 0: keep car stopped, count down timer
        2. If timer reaches 0: change tyres, move to pit exit
        3. Otherwise: normal step, check for pit entry crossing
        4. Update lap counter and pit statistics

        Args:
            action: [steering, throttle, brake]

        Returns:
            observation: Current observation
            reward: Reward (with pit time penalty if applicable)
            terminated: Episode termination flag
            truncated: Episode truncation flag
            info: Info dict with pit status
        """
        # Get current state before step
        old_distance = self.env.last_distance_along

        # === PIT STOP ACTIVE ===
        if self.in_pit_lane and self.pit_stop_timer > 0:
            # Car is stopped in pit box
            # Don't update physics, just count down timer
            self.pit_stop_timer -= self.env.dt

            # Keep car stationary (override action)
            obs, reward, terminated, truncated, info = self.env.step(
                np.array([0.0, 0.0, 1.0])  # Full brake, no movement
            )

            # Apply pit stop time penalty to reward
            reward -= 1.0  # Penalty for being stopped

            # Check if pit stop completed
            if self.pit_stop_timer <= 0:
                self._complete_pit_stop()

        # === NORMAL DRIVING ===
        else:
            # Execute normal step
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Get current distance
            current_distance = info['distance_along']

            # === DETECT PIT ENTRY ===
            if self.pit_requested and not self.in_pit_lane:
                # Check if car crossed pit entry zone
                if self._is_crossing_zone(old_distance, current_distance, self.pit_entry_distance):
                    if not self._crossed_pit_entry:
                        self._enter_pit_lane()
                        self._crossed_pit_entry = True
                        reward -= 0.5  # Small penalty for entering pits
                else:
                    self._crossed_pit_entry = False

        # === UPDATE LAP COUNTER ===
        current_lap = info.get('lap', 1)
        if current_lap > self.last_lap_number:
            self.laps_since_pit += 1
            self.last_lap_number = current_lap

        # === DETECT FINISH LINE (reset pit entry detection) ===
        current_distance = info['distance_along']
        if current_distance < 50.0 and old_distance > self.env.track.total_length - 50.0:
            # Crossed finish line
            self._crossed_finish_line = True
            self._crossed_pit_entry = False  # Reset pit entry detection for next lap

        # Add pit status to info
        info['pit_status'] = self.get_pit_status()
        info['laps_since_pit'] = self.laps_since_pit
        info['total_pit_stops'] = self.total_pit_stops

        return obs, reward, terminated, truncated, info

    def _is_crossing_zone(
        self,
        old_distance: float,
        new_distance: float,
        zone_distance: float,
        tolerance: float = 20.0
    ) -> bool:
        """
        Check if car crossed a specific zone on the track.

        Handles wraparound at finish line.

        Args:
            old_distance: Previous distance along track
            new_distance: Current distance along track
            zone_distance: Target zone position
            tolerance: Zone width (meters)

        Returns:
            True if zone was crossed
        """
        # Check for finish line wraparound
        if new_distance < old_distance:
            # Crossed finish line
            # Check if zone is near start or end
            if zone_distance < tolerance:
                # Zone is near start
                return new_distance <= zone_distance + tolerance
            elif zone_distance > self.env.track.total_length - tolerance:
                # Zone is near end
                return old_distance <= zone_distance + tolerance
            else:
                return False
        else:
            # Normal case: no wraparound
            return (old_distance < zone_distance - tolerance and 
                    new_distance >= zone_distance - tolerance)

    def _enter_pit_lane(self):
        """
        Enter pit lane and start pit stop.

        This is called when the car crosses the pit entry zone with
        a pit stop requested.
        """
        print(f"🔧 Entering pit lane (Lap {self.last_lap_number})")
        self.in_pit_lane = True
        self.pit_stop_timer = self.pit_stop_duration
        self.pit_requested = False  # Request fulfilled

    def _complete_pit_stop(self):
        """
        Complete the pit stop.

        This is called when the pit stop timer reaches zero.
        Changes tyres and moves car to pit exit.
        """
        # Change tyres
        old_compound = self.env.tyres.compound.name
        new_compound = self.new_compound if self.new_compound else self.env.tyres.compound

        print(f"🔧 Pit stop complete: {old_compound} → {new_compound.name}")
        self.env.tyres.reset(new_compound)

        # Move car to pit exit
        # Set position to pit exit distance
        # Calculate position from distance along track
        pit_exit_pos, pit_exit_heading = self._get_position_from_distance(
            self.pit_exit_distance
        )

        # Reset car position
        self.env.car.reset(
            x=pit_exit_pos[0],
            y=pit_exit_pos[1],
            heading=pit_exit_heading
        )

        # Update state
        self.in_pit_lane = False
        self.pit_stop_timer = 0.0
        self.laps_since_pit = 0
        self.total_pit_stops += 1
        self.new_compound = None

    def _get_position_from_distance(
        self,
        distance: float
    ) -> Tuple[np.ndarray, float]:
        """
        Get (x, y) position and heading from distance along track.

        Args:
            distance: Distance along track centerline

        Returns:
            position: (x, y) position
            heading: Heading angle in radians
        """
        # Use track's existing methods to get position and heading
        position = self.env.track.get_point_at_distance(distance)
        heading = self.env.track.get_direction_at_distance(distance)
        return position, heading

    def request_pit(self, compound: Optional[TyreCompound] = None):
        """
        Request a pit stop for the next opportunity.

        This is called by the engineer agent to signal that a pit stop
        should be performed. The pit stop will happen when the car
        crosses the pit entry zone.

        Args:
            compound: Tyre compound to use after pit stop.
                     If None, uses same compound as current.
        """
        if self.in_pit_lane:
            print("⚠️  Already in pit lane, cannot request another pit stop")
            return

        print(f"📢 Pit stop requested (will enter on next lap)")
        self.pit_requested = True
        self.new_compound = compound

    def cancel_pit(self):
        """
        Cancel a pending pit stop request.

        Only works if the car hasn't entered the pit lane yet.
        """
        if self.in_pit_lane:
            print("⚠️  Already in pit lane, cannot cancel")
            return

        if self.pit_requested:
            print("❌ Pit stop request cancelled")
            self.pit_requested = False
            self.new_compound = None

    def get_pit_status(self) -> Dict[str, Any]:
        """
        Get current pit stop status.

        Returns:
            Dictionary with:
                - pit_requested: Whether pit stop is requested
                - in_pit_lane: Whether car is in pit lane
                - pit_stop_timer: Remaining time in pit stop (seconds)
                - laps_since_pit: Laps since last pit stop
                - total_pit_stops: Total pit stops this episode
                - next_compound: Compound to use after pit (if requested)
        """
        return {
            'pit_requested': self.pit_requested,
            'in_pit_lane': self.in_pit_lane,
            'pit_stop_timer': self.pit_stop_timer,
            'laps_since_pit': self.laps_since_pit,
            'total_pit_stops': self.total_pit_stops,
            'next_compound': self.new_compound.name if self.new_compound else None,
        }
