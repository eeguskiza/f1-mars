# F1 Mars - 2D Racing Simulator with Reinforcement Learning

A 2D Formula 1 racing simulator built with Gymnasium and PyGame, featuring dual reinforcement learning agents for pilot control and race strategy.

## Features

- **Realistic 2D Physics**: Custom physics engine with tire degradation, fuel consumption, and car dynamics
- **Gymnasium Environment**: Standard RL interface compatible with Stable-Baselines3
- **Dual Agent System**:
  - **Pilot Agent**: Controls steering, throttle, and braking
  - **Engineer Agent**: Manages tire strategy and pit stop decisions
- **Custom Tracks**: JSON-based track definitions for easy circuit creation
- **Real-time Rendering**: PyGame-based visualization with HUD overlay
- **Human Playable**: Manual control mode for testing and demonstration

## Project Structure

```
f1_mars/
├── f1_mars/                      # Main package
│   ├── envs/                     # Gymnasium environment ✅
│   │   ├── f1_env.py            # Main environment (BALANCED)
│   │   ├── car.py               # Car physics (bicycle model)
│   │   ├── track.py             # Track system (splines)
│   │   └── tyres.py             # Tyre degradation (strategic)
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
│   └── example_random_agent.py  # Random agent example
├── tests/                        # Test suite ✅
│   ├── conftest.py              # Pytest fixtures
│   ├── test_environment.py      # F1Env tests
│   ├── test_tyres.py            # Tyre physics tests
│   ├── test_tyres_extended.py   # Extended tyre validation
│   ├── test_integration.py      # Integration tests
│   └── test_verification.py     # Quick verification
├── docs/                         # Documentation ✅
│   ├── IMPLEMENTATION_STATUS.md # Implementation summary
│   └── TYRE_WEAR_FIX_SUMMARY.md # Tyre wear fix details
├── tracks/                       # Track definitions (JSON) ✅
├── trained_models/               # Saved RL models (empty)
├── logs/                         # TensorBoard logs (empty)
├── main.py                       # Entry point ✅
├── setup.py                      # Package configuration
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Installation

### From source

```bash
git clone https://github.com/yourusername/f1-mars.git
cd f1-mars
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Run Physics Demo

```bash
python main.py demo
```

This demonstrates all physics components working together:
- Car physics with balanced parameters
- Tyre degradation and temperature
- LIDAR sensors
- Track boundaries

### Run Random Agent Example

```bash
python main.py random
```

Shows a simple LIDAR-based policy driving in the environment.

### Run Test Suite

```bash
python main.py test
```

Or directly with pytest:

```bash
pytest tests/ -v
```

### Use as Gymnasium Environment

```python
from f1_mars.envs import F1Env

env = F1Env(max_laps=3)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # [steering, throttle, brake]
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

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

- [ ] Implement weather conditions
- [ ] Add multiplayer support
- [ ] Create web-based track editor
- [ ] Add car setup tuning
- [ ] Implement DRS and ERS systems
- [ ] Create championship mode

## Acknowledgments

Built with Gymnasium, Stable-Baselines3, and PyGame.
