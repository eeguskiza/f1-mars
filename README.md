# F1 Mars - 2D Racing Simulator with Reinforcement Learning

A 2D Formula 1 racing simulator built with Gymnasium and PyGame, featuring dual reinforcement learning agents for pilot control and race strategy.

## Features

- **Realistic 2D Physics**:
  - Bicycle model car dynamics with balanced parameters
  - 350 km/h max speed (~16 laps to tyre cliff edge)
  - Temperature-dependent tyre degradation
  - Realistic lap times (~30 seconds)

- **Gymnasium Environment**:
  - Standard RL interface compatible with Stable-Baselines3
  - 26-dimensional observation space (LIDAR, telemetry, track info)
  - Continuous action space [steering, throttle, brake]
  - Balanced reward structure for RL training

- **Pit Stop System** ✅ NEW:
  - Realistic pit stop mechanics with timing penalties
  - Tyre compound strategy (SOFT, MEDIUM, HARD)
  - Engineer-pilot coordination
  - Strategic decision-making (when to pit, which compound)

- **Dual Agent System**:
  - **Pilot Agent**: Controls steering, throttle, and braking
  - **Engineer Agent**: Manages tyre strategy and pit stop decisions
  - Decoupled responsibilities for specialized learning

- **LIDAR Sensors**:
  - 11 rays covering -75° to +75° field of view
  - Distance measurements to track boundaries
  - Perfect for obstacle avoidance and track following

- **Custom Tracks**:
  - JSON-based track definitions
  - Procedural track generation (oval, figure-8, etc.)
  - Spline-based smooth curves

- **Comprehensive Testing**:
  - 28+ unit tests covering all components
  - Pytest-based test suite
  - Fixtures for reusable test components

- **Real-time Rendering** (WIP):
  - PyGame-based visualization with HUD overlay
  - Telemetry display (speed, tyre wear, lap times)

- **Human Playable** (WIP):
  - Manual control mode for testing and demonstration

## Project Structure

```
f1_mars/
├── f1_mars/                      # Main package
│   ├── envs/                     # Gymnasium environment ✅
│   │   ├── __init__.py          # Package exports
│   │   ├── f1_env.py            # Main environment (BALANCED)
│   │   ├── car.py               # Car physics (bicycle model)
│   │   ├── track.py             # Track system (splines)
│   │   ├── tyres.py             # Tyre degradation (strategic)
│   │   └── pit_wrapper.py       # Pit stop wrapper ✅ NEW
│   ├── rendering/                # Visualization (WIP)
│   │   ├── renderer.py          # PyGame renderer
│   │   ├── hud.py               # UI overlay
│   │   └── assets/              # Sprites and graphics
│   ├── agents/                   # RL agent wrappers (WIP)
│   │   ├── pilot_agent.py
│   │   └── engineer_agent.py
│   └── utils/                    # Utilities ✅
│       ├── config.py            # Global constants
│       └── geometry.py          # Geometric helpers (raycast, etc.)
├── scripts/                      # Executable scripts ✅
│   ├── demo_physics.py          # Physics demonstration
│   ├── example_random_agent.py  # Random agent example
│   └── demo_pit_stops.py        # Pit stop demo ✅ NEW
├── tests/                        # Test suite ✅
│   ├── __init__.py              # Test package
│   ├── conftest.py              # Pytest fixtures
│   ├── test_environment.py      # F1Env tests
│   ├── test_tyres.py            # Tyre physics tests
│   ├── test_tyres_extended.py   # Extended tyre validation
│   ├── test_integration.py      # Integration tests
│   ├── test_verification.py     # Quick verification
│   ├── test_geometry.py         # Geometry utilities tests
│   └── test_pit_wrapper.py      # Pit stop tests ✅ NEW
├── docs/                         # Documentation ✅
│   ├── IMPLEMENTATION_STATUS.md       # Implementation summary
│   ├── TYRE_WEAR_FIX_SUMMARY.md       # Tyre wear fix details
│   ├── REORGANIZATION_COMPLETE.md     # Project reorganization
│   └── PIT_STOP_IMPLEMENTATION.md     # Pit stop docs ✅ NEW
├── tracks/                       # Track definitions (JSON) ✅
├── trained_models/               # Saved RL models (empty)
├── logs/                         # TensorBoard logs (empty)
├── main.py                       # Entry point ✅
├── setup.py                      # Package configuration
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python >= 3.8
- pip (Python package manager)

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/f1-mars.git
cd f1-mars

# Install dependencies
pip install numpy scipy gymnasium pygame

# Install package in development mode
pip install -e .
```

### Development Installation (with testing tools)

```bash
pip install -e .
pip install pytest  # For running tests
```

### Verify Installation

```bash
# Test imports
python -c "from f1_mars.envs import F1Env, PitStopWrapper; print('✅ Installation successful!')"

# Run quick test
python main.py test
```

## Quick Start

### 1. Run Physics Demo

```bash
python main.py demo
# or directly:
python scripts/demo_physics.py
```

**What it demonstrates:**
- Car physics with balanced parameters (350 km/h max speed)
- Tyre degradation and temperature simulation
- LIDAR sensors (11 rays)
- Track boundary detection
- Realistic lap times (~30 seconds per lap)

### 2. Run Random Agent Example

```bash
python main.py random
# or directly:
python scripts/example_random_agent.py
```

**What it shows:**
- Simple LIDAR-based obstacle avoidance policy
- Environment interaction loop
- Observation and action spaces
- Info dict contents (telemetry)

### 3. Run Pit Stop Demo

```bash
python scripts/demo_pit_stops.py
```

**What it demonstrates:**
- Engineer-pilot coordination
- Automatic pit stop at 50% tyre wear
- Tyre compound changes
- Pit stop timing and penalties
- Strategic decision-making

### 4. Run Test Suite

```bash
# Run all tests
python main.py test

# Run specific test files
pytest tests/test_environment.py -v          # Environment tests
pytest tests/test_tyres.py -v               # Tyre physics tests
pytest tests/test_pit_wrapper.py -v         # Pit stop tests
pytest tests/test_geometry.py -v            # Geometry utilities

# Run all tests with coverage
pytest tests/ -v --tb=short
```

## Usage Examples

### Basic Gymnasium Environment

```python
from f1_mars.envs import F1Env
import numpy as np

# Create environment
env = F1Env(max_laps=3)
obs, info = env.reset()

print(f"Observation shape: {obs.shape}")  # (26,)
print(f"Action space: {env.action_space}")  # Box([-1, 0, 0], [1, 1, 1])

# Run episode
for step in range(1000):
    # Random action: [steering, throttle, brake]
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    # Access telemetry
    print(f"Lap {info['lap']}, Speed: {info['velocity_kmh']:.1f} km/h, "
          f"Wear: {info['tyre_wear']:.1f}%")

    if terminated or truncated:
        print(f"Episode ended. Total reward: {info['episode_reward']:.1f}")
        break

env.close()
```

### Using Pit Stop Wrapper

```python
from f1_mars.envs import F1Env, PitStopWrapper, TyreCompound
import numpy as np

# Create environment with pit stop functionality
base_env = F1Env(max_laps=10)
env = PitStopWrapper(
    base_env,
    pit_stop_duration=3.0,      # 3 seconds pit stop
    pit_entry_distance=50.0,     # Pit entry at 50m
    pit_exit_distance=150.0      # Pit exit at 150m
)

obs, info = env.reset()
print(f"Starting compound: {info['tyre_compound']}")

for step in range(5000):
    # Your pilot policy here
    action = np.array([0.0, 0.8, 0.0])  # [steering, throttle, brake]

    obs, reward, terminated, truncated, info = env.step(action)

    # Engineer strategy: pit when wear > 60%
    if info['tyre_wear'] > 60.0:
        pit_status = info['pit_status']
        if not pit_status['pit_requested'] and not pit_status['in_pit_lane']:
            print(f"🔧 Requesting pit stop (wear: {info['tyre_wear']:.1f}%)")
            env.request_pit(compound=TyreCompound.SOFT)

    # Monitor pit stop
    if info['pit_status']['in_pit_lane']:
        timer = info['pit_status']['pit_stop_timer']
        print(f"⏱️  In pit: {timer:.1f}s remaining")

    if terminated or truncated:
        break

env.close()
```

### Custom Pilot Policy with LIDAR

```python
from f1_mars.envs import F1Env
import numpy as np

def lidar_pilot(obs):
    """Simple LIDAR-based obstacle avoidance."""
    # Extract LIDAR readings (indices 4-14)
    lidar = obs[4:15]

    # Calculate left and right clearance
    left_clear = np.mean(lidar[:5])   # Left 5 rays
    right_clear = np.mean(lidar[6:])  # Right 5 rays
    center_clear = lidar[5]           # Center ray

    # Steering: turn away from obstacles
    if center_clear < 50:
        # Obstacle ahead, turn harder
        steering = 0.8 if left_clear > right_clear else -0.8
    else:
        # Normal steering
        steering = 0.3 if left_clear > right_clear else -0.3

    # Throttle: slow down for obstacles
    throttle = 0.5 if center_clear < 100 else 0.8
    brake = 0.0

    return np.array([steering, throttle, brake], dtype=np.float32)

# Use the policy
env = F1Env(max_laps=5)
obs, info = env.reset()

for step in range(3000):
    action = lidar_pilot(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 300 == 0:  # Print every 5 seconds
        print(f"Step {step}: Lap {info['lap']}, "
              f"Speed {info['velocity_kmh']:.0f} km/h")

    if terminated or truncated:
        break

env.close()
```

### Engineer-Pilot Coordination

```python
from f1_mars.envs import F1Env, PitStopWrapper, TyreCompound
import numpy as np

def pilot_policy(obs):
    """Pilot focuses on driving fast."""
    lidar = obs[4:15]
    left_dist = np.mean(lidar[:5])
    right_dist = np.mean(lidar[6:])

    steering = 0.4 if left_dist > right_dist else -0.4
    throttle = 0.9  # Aggressive
    brake = 0.0

    return np.array([steering, throttle, brake], dtype=np.float32)

def engineer_strategy(info, env):
    """Engineer manages tyre strategy."""
    wear = info['tyre_wear']
    pit_status = info['pit_status']
    laps_since_pit = info['laps_since_pit']

    # Strategy: Pit at 70% wear or after 8 laps
    should_pit = (wear > 70.0) or (laps_since_pit >= 8)

    if should_pit and not pit_status['in_pit_lane']:
        if not pit_status['pit_requested']:
            # Choose compound based on remaining laps
            remaining_laps = info['max_laps'] - info['lap']
            if remaining_laps > 5:
                compound = TyreCompound.MEDIUM  # Long stint
            else:
                compound = TyreCompound.SOFT    # Sprint to finish

            print(f"📊 ENGINEER: Pit requested on lap {info['lap']} "
                  f"(wear: {wear:.1f}%, compound: {compound.name})")
            env.request_pit(compound=compound)

# Run with coordination
base_env = F1Env(max_laps=15)
env = PitStopWrapper(base_env, pit_stop_duration=2.0)

obs, info = env.reset()

for step in range(10000):
    # Pilot drives
    action = pilot_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # Engineer makes strategy calls
    engineer_strategy(info, env)

    if terminated or truncated:
        print(f"\n🏁 Race finished!")
        print(f"   Laps: {info['lap']}")
        print(f"   Pit stops: {info['pit_status']['total_pit_stops']}")
        print(f"   Final reward: {info['episode_reward']:.1f}")
        break

env.close()
```

## Pit Stop Wrapper

The `PitStopWrapper` adds realistic pit stop mechanics to the F1 environment using the Gymnasium wrapper pattern.

### Features

- **Zone-based pit entry/exit**: Pit lane defined by distances along track
- **Timer-based pit stops**: Configurable duration (default 3 seconds)
- **Tyre compound changes**: Strategic compound selection (SOFT/MEDIUM/HARD)
- **Reward penalties**: Time cost for pit stops (-0.5 entry, -1.0/timestep stopped)
- **Lap tracking**: Automatic laps_since_pit counter
- **Multiple stops**: Support for multiple pit stops per race

### API Reference

```python
from f1_mars.envs import PitStopWrapper, TyreCompound

# Wrap environment
env = PitStopWrapper(
    base_env,
    pit_stop_duration=3.0,      # Time in pit (seconds)
    pit_entry_distance=50.0,     # Pit entry position (meters)
    pit_exit_distance=150.0      # Pit exit position (meters)
)

# Request pit stop
env.request_pit(compound=TyreCompound.SOFT)

# Cancel pit stop (before entry)
env.cancel_pit()

# Get pit status
pit_status = env.get_pit_status()
# Returns: {
#   'pit_requested': bool,
#   'in_pit_lane': bool,
#   'pit_stop_timer': float,
#   'laps_since_pit': int,
#   'total_pit_stops': int,
#   'next_compound': str
# }
```

### Pit Stop Flow

1. Engineer calls `env.request_pit(compound)` → `pit_requested = True`
2. Car crosses pit entry zone → Enters pit lane
3. Car held stationary for `pit_stop_duration` seconds
4. Tyres changed to new compound, wear reset to 0%
5. Car repositioned at pit exit → Returns to racing

See `docs/PIT_STOP_IMPLEMENTATION.md` for detailed documentation.

## Configuration

Global configuration is stored in `f1_mars/utils/config.py`:

- `SCREEN_WIDTH`, `SCREEN_HEIGHT`: Display resolution
- `FPS`: Target frames per second
- `PHYSICS_STEPS_PER_FRAME`: Physics simulation substeps
- `CAR_MAX_SPEED`: Maximum car velocity
- `CAR_MAX_STEERING`: Maximum steering angle

## Creating Custom Tracks

Tracks are defined as JSON files in the `tracks/` directory:

```json
{
  "name": "Custom Circuit",
  "waypoints": [
    {"x": 100, "y": 100},
    {"x": 200, "y": 150},
    ...
  ],
  "width": 80,
  "start_position": {"x": 100, "y": 100, "angle": 0}
}
```

## Environment Details

### Observation Space (26 dimensions)

1. Velocity (normalized)
2. Steering angle (normalized)
3. Heading relative to track
4. Lateral offset from centerline
5-15. **LIDAR rays** (11 rays from -75° to +75°)
16-20. Track curvature ahead (5 points)
21. Tyre wear (%)
22. Current grip multiplier
23. Engineer signal (0=continue, 1=pit, 2=change)
24. Current lap (normalized)
25. Total laps (normalized)
26. Distance along track (normalized)

### Action Space

**Pilot Agent:**
- `Box([-1, 0, 0], [1, 1, 1])`
- [steering, throttle, brake]
- Steering: -1 (left) to +1 (right)
- Throttle: 0 to 1
- Brake: 0 to 1

**Engineer Agent (via signal):**
- 0 = Continue on current tyres
- 1 = Pit stop required
- 2 = Change compound

### Rewards (Balanced for RL)

- **Progress**: +0.1 per meter forward, -0.2 per meter backward
- **Checkpoint**: +10 per checkpoint
- **Lap completion**: +100 + time bonus
- **Off-track**: -5 per timestep
- **Low speed**: -0.5 if velocity < 10 m/s
- **High wear**: -0.1 if wear > 80%
- **Dead tyres**: -50
- **Pit entry**: -0.5 (one-time, with PitStopWrapper)
- **In pit**: -1.0 per timestep (with PitStopWrapper)

### Info Dictionary

The `info` dict returned by `env.step()` contains:

**Basic telemetry:**
- `lap`, `max_laps`: Current and total laps
- `checkpoint`, `total_checkpoints`: Progress tracking
- `distance_along`, `track_length`: Position on track
- `lap_progress`: Lap completion (0.0 to 1.0)
- `velocity`, `velocity_kmh`: Speed in m/s and km/h
- `position_x`, `position_y`, `heading`: Car pose
- `tyre_wear`, `tyre_temperature`, `tyre_grip`: Tyre state
- `tyre_compound`: Current compound (SOFT/MEDIUM/HARD)
- `on_track`, `off_track_frames`: Track position status
- `total_time`, `timestep`: Episode timing

**With PitStopWrapper (additional):**
- `pit_status`: Dictionary with:
  - `pit_requested`: Pit stop requested
  - `in_pit_lane`: Currently in pit
  - `pit_stop_timer`: Time remaining in pit
  - `laps_since_pit`: Laps since last pit
  - `total_pit_stops`: Pit stops completed
  - `next_compound`: Compound after pit

## Dependencies

- Python >= 3.8
- Gymnasium >= 0.29.0
- PyGame >= 2.5.0
- NumPy >= 1.24.0
- Stable-Baselines3 >= 2.1.0
- TensorBoard >= 2.14.0
- SciPy >= 1.11.0

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Roadmap

**Completed:**
- [x] Core physics engine with balanced parameters
- [x] Gymnasium environment interface
- [x] Tyre degradation system (temperature, wear, cliff edge)
- [x] LIDAR sensors for obstacle detection
- [x] Pit stop mechanics with tyre strategy ✅ NEW
- [x] Comprehensive test suite (28+ tests)
- [x] Project reorganization and documentation

**In Progress:**
- [ ] PyGame rendering with real-time visualization
- [ ] Human playable mode with keyboard controls
- [ ] RL agent training (Stable-Baselines3 integration)

**Planned:**
- [ ] Implement weather conditions (rain, temperature)
- [ ] Add multiplayer support (multiple cars)
- [ ] Create web-based track editor
- [ ] Add car setup tuning (downforce, gear ratios)
- [ ] Implement DRS and ERS systems
- [ ] Create championship mode with points system
- [ ] Add telemetry logging and replay system

## Acknowledgments

Built with Gymnasium, Stable-Baselines3, and PyGame.
