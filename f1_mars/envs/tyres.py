"""Tyre degradation and temperature system for F1 Mars simulator."""

from enum import Enum
from typing import Dict, Tuple, Optional


class TyreCompound(Enum):
    """
    F1 tire compounds with different performance characteristics.

    Each compound has:
    - grip_base: Base grip level (higher = more grip)
    - wear_rate: How quickly the tire degrades (higher = faster wear)
    - optimal_temp_range: Temperature range for best performance (min, max) in °C
    """

    SOFT = {
        'grip_base': 1.0,
        'wear_rate': 1.5,
        'optimal_temp_range': (85, 105)
    }

    MEDIUM = {
        'grip_base': 0.9,
        'wear_rate': 1.0,
        'optimal_temp_range': (80, 100)
    }

    HARD = {
        'grip_base': 0.8,
        'wear_rate': 0.6,
        'optimal_temp_range': (75, 95)
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
    def optimal_temp_range(self) -> Tuple[int, int]:
        """Get optimal temperature range (min, max) in Celsius."""
        return self.value['optimal_temp_range']

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

    Models:
    - Progressive wear based on speed, cornering, and throttle usage
    - Temperature dynamics affected by driving conditions
    - Grip degradation with "cliff edge" effect at high wear
    - Temperature-dependent performance
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
        self.optimal_temp_min, self.optimal_temp_max = compound.optimal_temp_range

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
            self.optimal_temp_min, self.optimal_temp_max = compound.optimal_temp_range

        self.wear = 0.0
        self.temperature = 70.0

    def update(
        self,
        dt: float,
        speed: float,
        lateral_force: float,
        throttle: float
    ):
        """
        Update tire wear and temperature based on driving conditions.

        Wear Model:
        - Base wear increases with time
        - Speed increases wear (normalized to ~200 units)
        - Lateral forces (cornering) significantly increase wear
        - Hard throttle application increases wear

        Temperature Model:
        - Target temperature depends on speed and cornering
        - Temperature moves smoothly toward target
        - Higher activity = higher temperature

        Args:
            dt: Time step in seconds
            speed: Current speed (m/s or game units)
            lateral_force: Magnitude of lateral acceleration (0-1 normalized)
            throttle: Throttle input [0, 1]
        """
        # === Wear Model ===
        # Base wear per timestep
        base_wear = dt * self.wear_rate * 0.01

        # Speed factor: higher speed = more wear
        # Normalize to typical max speed of 200
        speed_factor = speed / 200.0

        # Lateral force factor: cornering wears tires significantly
        lateral_factor = abs(lateral_force) * 0.5

        # Throttle factor: hard acceleration wears tires
        throttle_factor = 1.0 + throttle * 0.3

        # Combine all factors
        total_wear = base_wear * speed_factor * (1.0 + lateral_factor) * throttle_factor

        # Apply wear
        self.wear = min(100.0, self.wear + total_wear)

        # === Temperature Model ===
        # Calculate target temperature based on activity
        # Base temperature: 70°C (ambient/idle)
        # Speed contribution: hotter at high speed
        # Lateral force contribution: cornering generates heat
        target_temp = 70.0 + speed * 0.1 + abs(lateral_force) * 20.0

        # Smooth temperature change (thermal inertia)
        # Temperature moves 10% toward target each second
        temp_change_rate = 10.0 * dt  # 10% per second
        self.temperature += (target_temp - self.temperature) * temp_change_rate

        # Clamp temperature to reasonable range
        self.temperature = max(20.0, min(150.0, self.temperature))

    def get_grip(self) -> float:
        """
        Calculate current grip multiplier based on wear and temperature.

        Grip Degradation Model:
        - Wear 0-50%: No degradation (tires still good)
        - Wear 50-80%: Linear degradation (performance drops)
        - Wear 80-100%: "Cliff edge" - rapid performance loss

        Temperature Effect:
        - Below optimal: 20% grip penalty (cold tires)
        - Above optimal: 10% grip penalty (overheating)
        - Within optimal: No penalty

        Returns:
            Grip multiplier [0.3, 1.0] where 1.0 is maximum grip
            Minimum is clamped to 0.3 (tires never completely lose grip)
        """
        grip = self.grip_base

        # === Wear-based degradation ===
        if self.wear < 50.0:
            # Fresh tires: no degradation
            wear_penalty = 0.0
        elif self.wear <= 80.0:
            # Moderate wear: linear degradation
            # At 50% wear: 0% loss
            # At 80% wear: 30% loss
            wear_penalty = (self.wear - 50.0) / 100.0
        else:
            # High wear: cliff edge
            # At 80% wear: 30% loss
            # At 100% wear: 70% loss (only 30% grip remaining)
            wear_penalty = 0.3 + (self.wear - 80.0) / 50.0

        grip *= (1.0 - wear_penalty)

        # === Temperature-based penalty ===
        if self.temperature < self.optimal_temp_min:
            # Cold tires: 20% penalty
            grip *= 0.8
        elif self.temperature > self.optimal_temp_max:
            # Overheating: 10% penalty
            grip *= 0.9
        # else: optimal temperature, no penalty

        # Clamp minimum grip to 0.3 (never completely lose all grip)
        return max(0.3, grip)

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

        Tires are considered "dead" at 95% wear, requiring a pit stop.

        Returns:
            True if wear >= 95%, indicating mandatory tire change
        """
        return self.wear >= 95.0


class TyreStrategy:
    """
    Helper class for tire strategy decisions (used by engineer agent).

    Provides heuristics for:
    - Estimating remaining tire life
    - Deciding when to pit
    - Optimal compound selection
    """

    @staticmethod
    def estimate_laps_remaining(
        tyre_set: TyreSet,
        avg_wear_per_lap: float
    ) -> int:
        """
        Estimate how many more laps the current tires can last.

        Uses linear extrapolation based on average wear per lap.
        Assumes tires are dead at 95% wear.

        Args:
            tyre_set: Current tire set
            avg_wear_per_lap: Average wear accumulated per lap

        Returns:
            Estimated number of full laps remaining (rounded down)
            Returns 0 if tires are already dead or near death
        """
        if tyre_set.is_dead():
            return 0

        if avg_wear_per_lap <= 0:
            # No wear or negative wear? Assume infinite laps
            return 999

        # Usable wear remaining (leave safety margin)
        wear_remaining = 95.0 - tyre_set.wear

        # Estimate laps
        laps_remaining = wear_remaining / avg_wear_per_lap

        return max(0, int(laps_remaining))

    @staticmethod
    def should_pit(
        tyre_set: TyreSet,
        laps_remaining: int,
        pit_time_cost: float
    ) -> bool:
        """
        Heuristic to decide if car should pit for fresh tires.

        Decision factors:
        1. Mandatory pit if tires are dead (>= 95% wear)
        2. Consider pit if tires are in "cliff edge" zone (> 80% wear)
        3. Don't pit if very few laps remaining in race
        4. Consider time cost of pit stop vs. performance gain

        Args:
            tyre_set: Current tire set
            laps_remaining: Laps remaining in the race
            pit_time_cost: Time lost in pit stop (seconds)

        Returns:
            True if car should pit for new tires
        """
        # Mandatory pit if tires are dead
        if tyre_set.is_dead():
            return True

        # If less than 3 laps remaining, avoid pitting (not worth the time)
        if laps_remaining < 3:
            return False

        # If in cliff edge zone (80%+ wear), strongly consider pitting
        if tyre_set.wear > 80.0:
            # Only pit if we have enough laps to benefit from fresh tires
            # (recover the time lost in pit stop)
            # Assume ~1 second per lap gained with fresh tires
            time_gain_per_lap = 1.0
            potential_gain = laps_remaining * time_gain_per_lap

            if potential_gain > pit_time_cost:
                return True

        # If wear is moderate (50-80%), pit only if many laps remain
        if tyre_set.wear > 50.0 and laps_remaining > 10:
            # Calculate if we can make it to the end on current tires
            # Rough estimate: assume wear increases linearly
            estimated_final_wear = tyre_set.wear + (laps_remaining * 3.0)  # ~3% per lap estimate

            if estimated_final_wear > 95.0:
                # Won't make it to the end, need to pit
                return True

        # Default: don't pit
        return False

    @staticmethod
    def choose_compound(
        laps_remaining: int,
        track_degradation: str = "medium"
    ) -> TyreCompound:
        """
        Choose optimal tire compound for given race situation.

        Strategy:
        - Short stint (< 10 laps): SOFT (maximize pace)
        - Medium stint (10-20 laps): MEDIUM (balanced)
        - Long stint (> 20 laps): HARD (durability)
        - High degradation tracks: favor harder compounds

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
