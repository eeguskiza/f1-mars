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
├── f1_mars/                    # Main package
│   ├── envs/                   # Gymnasium environment
│   │   ├── f1_env.py          # Main environment
│   │   ├── car.py             # Car physics model
│   │   ├── track.py           # Track system
│   │   └── tyres.py           # Tire degradation system
│   ├── rendering/              # Visualization
│   │   ├── renderer.py        # PyGame renderer
│   │   ├── hud.py             # UI overlay
│   │   └── assets/            # Sprites and graphics
│   ├── agents/                 # RL agent wrappers
│   │   ├── pilot_agent.py
│   │   └── engineer_agent.py
│   └── utils/                  # Utilities
│       ├── config.py          # Global constants
│       └── geometry.py        # Geometric helpers
├── tracks/                     # Track definitions (JSON)
├── trained_models/             # Saved RL models
├── logs/                       # TensorBoard logs
├── scripts/                    # Training and evaluation
│   ├── train_pilot.py
│   ├── train_engineer.py
│   ├── evaluate.py
│   └── play_human.py
└── tests/                      # Unit tests
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

### Train the Pilot Agent

```bash
python scripts/train_pilot.py --track monaco --episodes 10000
```

### Train the Engineer Agent

```bash
python scripts/train_engineer.py --track spa --episodes 5000
```

### Evaluate Trained Models

```bash
python scripts/evaluate.py --pilot-model trained_models/pilot_best.zip
```

### Play Manually

```bash
python scripts/play_human.py --track silverstone
```

Controls:
- Arrow Keys: Steer and accelerate/brake
- Space: Pit stop
- ESC: Quit

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

### Observation Space

- LIDAR distance sensors (16 rays)
- Current speed and steering angle
- Tire condition (4 values)
- Fuel level
- Distance to next checkpoint
- Lap progress

### Action Space

**Pilot Agent:**
- Continuous: [steering, throttle, brake]

**Engineer Agent:**
- Discrete: [continue, pit_soft, pit_medium, pit_hard]

### Rewards

- Checkpoint passed: +100
- Lap completed: +1000
- Collision: -100
- Reverse driving: -10
- Speed bonus: +0.1 per unit of speed

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
