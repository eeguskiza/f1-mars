"""Tyre degradation and temperature system for F1 Mars simulator - BALANCED."""

from enum import Enum
from typing import Dict, Tuple, Optional
import numpy as np


class TyreCompound(Enum):
    """
    F1 tire compounds with different performance characteristics.

    Balanced for realistic F1 stint lengths:
    - SOFT: ~12-18 laps, maximum grip
    - MEDIUM: ~20-28 laps, balanced
    - HARD: ~30-40 laps, durable but slower

    Each compound has:
    - grip_base: Base grip level (higher = more grip)
    - wear_rate: How quickly the tire degrades (higher = faster wear)
    - optimal_temp_min/max: Temperature range for best performance (°C)
    - temp_sensitivity: How much temperature affects grip
    """

    SOFT = {
        'grip_base': 1.0,
        'wear_rate': 2.5,  # Aggressive wear for short stints
        'optimal_temp_min': 85,
        'optimal_temp_max': 105,
        'temp_sensitivity': 1.2  # More sensitive to temp
    }

    MEDIUM = {
        'grip_base': 0.92,
        'wear_rate': 1.5,  # Balanced wear
        'optimal_temp_min': 80,
        'optimal_temp_max': 100,
        'temp_sensitivity': 1.0
    }

    HARD = {
        'grip_base': 0.85,
        'wear_rate': 0.9,  # Slow wear for long stints
        'optimal_temp_min': 75,
        'optimal_temp_max': 95,
        'temp_sensitivity': 0.8  # Less temperature sensitive
    }

    @property
    def grip_base(self) -> float:
        """Get base grip level for this compound."""
        return self.value['grip_base']

    @property
    def wear_rate(self) -> float:
        """Get wear rate multiplier for this compound."""
        return self.value['wear_rate']

    @property
    def optimal_temp_min(self) -> int:
        """Get minimum optimal temperature."""
        return self.value['optimal_temp_min']

    @property
    def optimal_temp_max(self) -> int:
        """Get maximum optimal temperature."""
        return self.value['optimal_temp_max']

    @property
    def temp_sensitivity(self) -> float:
        """Get temperature sensitivity multiplier."""
        return self.value.get('temp_sensitivity', 1.0)

    @property
    def optimal_temp_range(self) -> Tuple[int, int]:
        """Get optimal temperature range (min, max) in Celsius."""
        return (self.optimal_temp_min, self.optimal_temp_max)

    @classmethod
    def from_string(cls, name: str) -> 'TyreCompound':
        """Convert string name to TyreCompound."""
        name_upper = name.upper()
        for compound in cls:
            if compound.name == name_upper:
                return compound
        raise ValueError(f"Unknown tyre compound: {name}")


class TyreSet:
    """
    Represents a set of four tires with wear and temperature simulation.

    BALANCED for realistic F1 racing:
    - Wear accumulates realistically (3-5% per lap typical)
    - Temperature affects grip significantly
    - Cliff edge effect at ~70% wear
    - Strategic pit stops necessary every 15-25 laps
    """

    def __init__(self, compound: TyreCompound):
        """
        Initialize a fresh set of tires.

        Args:
            compound: Tire compound type (SOFT, MEDIUM, or HARD)
        """
        self.compound = compound
        self.wear = 0.0  # Wear percentage [0, 100]
        self.temperature = 70.0  # Temperature in Celsius (starts at ambient)

        # Cache compound properties
        self.grip_base = compound.grip_base
        self.wear_rate = compound.wear_rate
        self.optimal_temp_min = compound.optimal_temp_min
        self.optimal_temp_max = compound.optimal_temp_max
        self.temp_sensitivity = compound.temp_sensitivity

    def reset(self, compound: Optional[TyreCompound] = None):
        """
        Reset to fresh tires, optionally changing compound.

        Args:
            compound: New compound, or None to keep current compound
        """
        if compound is not None:
            self.compound = compound
            self.grip_base = compound.grip_base
            self.wear_rate = compound.wear_rate
            self.optimal_temp_min = compound.optimal_temp_min
            self.optimal_temp_max = compound.optimal_temp_max
            self.temp_sensitivity = compound.temp_sensitivity

        self.wear = 0.0
        self.temperature = 70.0

    def update(
        self,
        dt: float,
        speed: float,
        lateral_force: float,
        throttle: float,
        braking: float = 0.0
    ):
        """
        Update tire wear and temperature based on driving conditions.

        BALANCED PHYSICS:
        - Typical wear: 3-5% per lap on medium tires
        - Temperature fluctuates 70-120°C based on driving
        - Out-of-temp-window degrades tires faster

        Args:
            dt: Time step in seconds
            speed: Current speed (m/s, max ~97)
            lateral_force: Magnitude of lateral acceleration [0, 1] normalized
            throttle: Throttle input [0, 1]
            braking: Brake input [0, 1]
        """
        # === WEAR MODEL ===
        # Base wear per timestep (calibrated for ~4-5% per lap in mixed driving)
        # At 30s lap time and ~1800 frames (60 fps), we want ~4.5% per lap
        # IMPORTANT: Tuned for realistic mixed driving (straights + corners)
        # Not constant high-G cornering like in test_balance.py
        # Final calibration: 0.035 gives ~4.4% per 30s → ~16 laps to cliff at 30s/lap
        base_wear = dt * self.wear_rate * 0.035  # Realistic driving calibration

        # Speed factor: higher speed = more wear (quadratic)
        speed_normalized = speed / 97.0  # Normalize to F1 max speed
        speed_factor = 0.5 + 1.5 * (speed_normalized ** 2)

        # Lateral force factor: cornering wears tires significantly
        lateral_factor = 1.0 + lateral_force * 2.0  # Up to 3x in hard corners (reduced from 4x)

        # Traction factor: hard acceleration/braking wears tires
        traction_factor = 1.0 + throttle * 0.3 + braking * 0.5  # Reduced impact

        # Temperature factor: out of optimal range wears faster
        temp_factor = 1.0
        if self.temperature < self.optimal_temp_min:
            temp_deficit = self.optimal_temp_min - self.temperature
            temp_factor = 1.0 + temp_deficit * 0.01  # +1% per degree below (reduced from 2%)
        elif self.temperature > self.optimal_temp_max:
            temp_excess = self.temperature - self.optimal_temp_max
            temp_factor = 1.0 + temp_excess * 0.015  # +1.5% per degree above (reduced from 3%)

        # Total wear
        total_wear = base_wear * speed_factor * lateral_factor * traction_factor * temp_factor
        self.wear = min(100.0, self.wear + total_wear)

        # === TEMPERATURE MODEL ===
        # Base temperature (ambient/idle)
        base_temp = 70.0

        # Heat from speed (aerodynamic friction)
        speed_heat = speed_normalized * 25.0  # Up to +25°C at max speed

        # Heat from cornering (tire friction)
        lateral_heat = lateral_force * 35.0  # Up to +35°C in hard corners

        # Heat from traction (wheel spin/braking)
        traction_heat = (throttle + braking) * 15.0  # Up to +30°C combined

        # Target temperature
        target_temp = base_temp + speed_heat + lateral_heat + traction_heat

        # Thermal dynamics (gradual change)
        temp_rate = 15.0 * dt  # Degrees per second rate of change
        if self.temperature < target_temp:
            # Heat up faster
            self.temperature = min(target_temp, self.temperature + temp_rate * 1.2)
        else:
            # Cool down slower (thermal mass)
            self.temperature = max(target_temp, self.temperature - temp_rate * 0.8)

        # Clamp temperature to realistic range
        self.temperature = np.clip(self.temperature, 50.0, 150.0)

    def get_grip(self) -> float:
        """
        Calculate current grip multiplier based on wear and temperature.

        BALANCED GRIP MODEL:
        - Fresh tires (0-30%): Full grip
        - Worn tires (30-50%): Slight degradation
        - Medium wear (50-70%): Noticeable degradation
        - Cliff edge (70-85%): Rapid grip loss
        - Destroyed (85-100%): Dangerous, minimal grip

        Temperature penalties:
        - Cold: -1.5% per degree below optimal
        - Hot: -1.0% per degree above optimal

        Returns:
            Grip multiplier [0.3, 1.0]
        """
        grip = self.grip_base

        # === WEAR-BASED DEGRADATION ===
        if self.wear < 30:
            # Fresh tire: full grip
            wear_factor = 1.0
        elif self.wear < 50:
            # Slight wear: minimal loss
            # 30% -> 1.0, 50% -> 0.9
            wear_factor = 1.0 - (self.wear - 30) * 0.005
        elif self.wear < 70:
            # Medium wear: noticeable loss
            # 50% -> 0.9, 70% -> 0.7
            wear_factor = 0.9 - (self.wear - 50) * 0.01
        elif self.wear < 85:
            # CLIFF EDGE: rapid degradation
            # 70% -> 0.7, 85% -> 0.4
            wear_factor = 0.7 - (self.wear - 70) * 0.02
        else:
            # Destroyed: minimal grip
            # 85% -> 0.4, 100% -> 0.25
            wear_factor = 0.4 - (self.wear - 85) * 0.01

        grip *= max(0.3, wear_factor)

        # === TEMPERATURE-BASED DEGRADATION ===
        if self.temperature < self.optimal_temp_min:
            # Too cold: significant penalty
            temp_deficit = self.optimal_temp_min - self.temperature
            temp_factor = 1.0 - (temp_deficit * 0.015 * self.temp_sensitivity)
        elif self.temperature > self.optimal_temp_max:
            # Too hot: moderate penalty + accelerated wear (already applied)
            temp_excess = self.temperature - self.optimal_temp_max
            temp_factor = 1.0 - (temp_excess * 0.01 * self.temp_sensitivity)
        else:
            # Optimal temperature: no penalty
            temp_factor = 1.0

        grip *= max(0.7, temp_factor)

        # Final grip clamped to reasonable range
        return max(0.3, min(1.0, grip))

    def get_state(self) -> Dict:
        """
        Get current tire state for observation/logging.

        Returns:
            Dictionary with:
                - compound: Compound name as string
                - wear: Wear percentage [0, 100]
                - temperature: Temperature in Celsius
                - current_grip: Current grip multiplier
        """
        return {
            'compound': self.compound.name,
            'wear': self.wear,
            'temperature': self.temperature,
            'current_grip': self.get_grip()
        }

    def is_dead(self) -> bool:
        """
        Check if tires are too worn to continue safely.

        Tires considered "dead" at 90% wear (was 95%, now more conservative).

        Returns:
            True if wear >= 90%, indicating mandatory tire change
        """
        return self.wear >= 90.0


class TyreStrategy:
    """
    Helper class for tire strategy decisions (used by engineer agent).

    Provides balanced heuristics for realistic F1 strategy.
    """

    @staticmethod
    def estimate_laps_remaining(
        tyre_set: TyreSet,
        avg_wear_per_lap: float
    ) -> int:
        """
        Estimate laps remaining before cliff edge (70% wear).

        Conservative estimate to avoid getting caught in cliff.

        Args:
            tyre_set: Current tire set
            avg_wear_per_lap: Average wear accumulated per lap

        Returns:
            Estimated number of full laps before cliff edge
        """
        if avg_wear_per_lap <= 0:
            return 999

        # Target cliff edge at 70%, not 100%
        usable_wear = max(0, 70.0 - tyre_set.wear)
        laps = usable_wear / avg_wear_per_lap

        return max(0, int(laps))

    @staticmethod
    def should_pit(
        tyre_set: TyreSet,
        laps_remaining: int,
        avg_wear_per_lap: float,
        pit_time_cost_laps: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Decide if car should pit for fresh tires.

        BALANCED STRATEGY:
        - Mandatory pit if wear > 75% (approaching cliff)
        - Mandatory pit if grip < 60% (dangerous)
        - Strategic pit if undercut opportunity exists
        - No pit if final laps and tires will survive

        Args:
            tyre_set: Current tire set
            laps_remaining: Laps remaining in race
            avg_wear_per_lap: Typical wear per lap
            pit_time_cost_laps: Laps lost to pit stop (default 1.0)

        Returns:
            (should_pit: bool, reason: str)
        """
        estimated_laps = TyreStrategy.estimate_laps_remaining(tyre_set, avg_wear_per_lap)
        current_grip = tyre_set.get_grip()

        # === CRITICAL: Mandatory pit ===
        if tyre_set.wear >= 75:
            return True, "CRITICAL: Tyre wear above 75% (cliff edge)"

        if current_grip < 0.6:
            return True, "CRITICAL: Grip below 60% (dangerous)"

        # === FINAL LAPS: No pit if tires will survive ===
        if laps_remaining <= 2 and tyre_set.wear < 85:
            return False, "Final laps, no pit needed"

        # === SURVIVAL: Won't make it to the end ===
        if estimated_laps < laps_remaining and estimated_laps <= 3:
            return True, f"Strategic: Only {estimated_laps} laps remaining on tyres"

        # === UNDERCUT: Strategic pit opportunity ===
        if 45 < tyre_set.wear < 60 and laps_remaining > 5:
            # Estimate time gain from fresh tires
            laps_on_new = laps_remaining - pit_time_cost_laps
            grip_gain = 1.0 - current_grip
            time_gain_per_lap = grip_gain * 0.5  # ~0.5s per lap per 0.1 grip
            total_gain = time_gain_per_lap * laps_on_new
            pit_cost = pit_time_cost_laps * 30.0  # ~30s lap time

            if total_gain > pit_cost:
                return True, "Strategic undercut opportunity"

        return False, "Continue on current tyres"

    @staticmethod
    def choose_compound(
        laps_remaining: int,
        track_degradation: str = "medium"
    ) -> TyreCompound:
        """
        Choose optimal tire compound for given race situation.

        Strategy:
        - Short stint (< 10 laps): SOFT (maximum pace)
        - Medium stint (10-20 laps): MEDIUM (balanced)
        - Long stint (> 20 laps): HARD (durability)
        - Adjust for track degradation level

        Args:
            laps_remaining: Number of laps until race end or next planned stop
            track_degradation: Track characteristic ("low", "medium", "high")

        Returns:
            Recommended tire compound
        """
        # Adjust thresholds based on track degradation
        if track_degradation == "high":
            short_threshold = 8
            medium_threshold = 15
        elif track_degradation == "low":
            short_threshold = 12
            medium_threshold = 25
        else:  # medium
            short_threshold = 10
            medium_threshold = 20

        # Choose compound based on stint length
        if laps_remaining < short_threshold:
            return TyreCompound.SOFT
        elif laps_remaining < medium_threshold:
            return TyreCompound.MEDIUM
        else:
            return TyreCompound.HARD
