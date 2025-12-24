# F1 Mars - 2D Racing Simulator with Reinforcement Learning

A 2D Formula 1 racing simulator built with Gymnasium and PyGame, featuring dual reinforcement learning agents for pilot control and race strategy.

## Features

- **Realistic 2D Physics**: Bicycle model car dynamics, 350 km/h max speed, temperature-dependent tyre degradation
- **Gymnasium Environment**: Standard RL interface compatible with Stable-Baselines3
- **Dual Agent System**: Separate pilot (driving) and engineer (strategy) agents
- **Pit Stop Mechanics**: Realistic pit stops with tyre compound strategy
- **LIDAR Sensors**: 11-ray sensor array for track boundary detection
- **Custom Tracks**: 4 difficulty-graded tracks for curriculum learning
- **Comprehensive Testing**: 28+ unit tests covering all components

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/f1-mars.git
cd f1-mars
pip install -r requirements.txt
```

### Run Demos

```bash
# Physics demonstration
python main.py demo

# Random agent example
python main.py random

# Run tests
python main.py test
```

See [Quick Start Guide](#usage) for detailed examples.

## Project Structure

```
f1_mars/
├── f1_mars/envs/          # Gymnasium environment & physics
├── tracks/                # 4 JSON track definitions
├── scripts/               # Training & demo scripts
├── tests/                 # Test suite (28+ tests)
├── docs/                  # Documentation
└── main.py               # Entry point
```

## Training

Train an RL agent to race autonomously:

```bash
# Basic training (oval track, PPO)
python scripts/train_pilot.py --total-timesteps 500000

# Curriculum learning (progressive difficulty)
python scripts/train_pilot.py --curriculum --total-timesteps 1000000

# Advanced: SAC with 16 parallel environments
python scripts/train_pilot.py --algorithm SAC --n-envs 16
```

**📖 Full Training Guide:** [scripts/TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md)
- Algorithm comparison (PPO vs SAC vs TD3)
- [Curriculum learning](docs/CURRICULUM_LEARNING.md) - Progressive difficulty training
- Hyperparameter tuning
- Multi-track training strategies

## Evaluation

Evaluate trained models with comprehensive metrics and visualizations:

```bash
# Basic evaluation
python scripts/evaluate.py --model trained_models/PPO_default_final.zip

# Compare two models
python scripts/evaluate.py \
    --model trained_models/PPO_v1.zip \
    --compare trained_models/SAC_v1.zip

# Record video
python scripts/evaluate.py --model MODEL --record
```

**📊 Evaluation Guide:** [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)
- Complete metrics (lap times, completion rate, tyre wear, etc.)
- Visualization plots (trajectory, lap time distribution, etc.)
- Model comparison
- Video recording

## Tracks

4 pre-made tracks for curriculum learning:

| Track | Difficulty | Length | Features |
|-------|------------|--------|----------|
| Oval | 0 (Beginner) | 907m | Wide, gentle turns |
| Simple | 1 (Basic) | 1432m | Hairpin corner |
| Technical | 2 (Advanced) | 1311m | Chicane section |
| Mixed | 3 (Expert) | 1873m | All corner types |

**📖 Track Guide:** [tracks/README.md](tracks/README.md)
- Track specifications
- Loading and using tracks
- Creating custom tracks

## Usage

### Basic Environment

```bash
python -c "
from f1_mars.envs import F1Env
env = F1Env(max_laps=3)
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated: break
"
```

### With Pit Stops

```bash
python scripts/demo_pit_stops.py
```

**📖 Pit Stop Documentation:** [docs/PIT_STOP_IMPLEMENTATION.md](docs/PIT_STOP_IMPLEMENTATION.md)

## Environment Specs

### Observation Space (26 dimensions)
- Velocity, steering, track position (4)
- LIDAR sensors (11 rays, -75° to +75°)
- Track curvature ahead (5)
- Tyre state (wear, grip, temperature)
- Lap/checkpoint progress
- Engineer signal

### Action Space
- **Pilot:** `[steering, throttle, brake]` continuous in `[-1,1] × [0,1] × [0,1]`
- **Engineer:** Request pit stops via wrapper API

### Reward Structure
- Progress: ±0.1 per meter
- Checkpoints: +10 each
- Lap completion: +100 + time bonus
- Off-track: -5 per timestep
- Pit stop: -0.5 entry + -1.0 per timestep stopped

Full specs in docstrings: `f1_mars/envs/f1_env.py`

## Key Scripts

| Script | Purpose | Documentation |
|--------|---------|---------------|
| `scripts/train_pilot.py` | Train RL agent | [TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md) |
| `scripts/demo_physics.py` | Physics demo | Built-in help |
| `scripts/demo_pit_stops.py` | Pit stop demo | [PIT_STOP_IMPLEMENTATION.md](docs/PIT_STOP_IMPLEMENTATION.md) |
| `scripts/test_all_tracks.py` | Analyze tracks | [tracks/README.md](tracks/README.md) |

## Testing

```bash
# All tests
pytest tests/ -v

# Specific test suites
pytest tests/test_environment.py -v   # Environment
pytest tests/test_tyres.py -v         # Tyre physics
pytest tests/test_tracks.py -v        # Track system
pytest tests/test_pit_wrapper.py -v   # Pit stops
```

## Documentation

- **[TRAINING_GUIDE.md](scripts/TRAINING_GUIDE.md)**: Complete training guide
- **[tracks/README.md](tracks/README.md)**: Track system documentation
- **[PIT_STOP_IMPLEMENTATION.md](docs/PIT_STOP_IMPLEMENTATION.md)**: Pit stop mechanics
- **[IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)**: Project status

## Dependencies

- Python >= 3.8
- Gymnasium >= 0.29.0
- Stable-Baselines3 >= 2.1.0
- PyGame >= 2.5.0
- NumPy >= 1.24.0
- SciPy >= 1.11.0

See `requirements.txt` for complete list.

## Development Status

**✅ Completed:**
- Core physics engine (car, tyres, track)
- Gymnasium environment interface
- Pit stop mechanics with strategy
- 4 curriculum learning tracks
- Comprehensive test suite (28+ tests)
- Training scripts (PPO, SAC, TD3)

**🚧 In Progress:**
- PyGame rendering
- Human playable mode
- Multi-agent training

**📋 Planned:**
- Weather conditions
- Multiplayer support
- Web track editor
- DRS/ERS systems

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

```bibtex
@software{f1mars2024,
  title={F1-MARS: 2D Racing Simulator with Reinforcement Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/f1-mars}
}
```

## Acknowledgments

Built with [Gymnasium](https://gymnasium.farama.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), and [PyGame](https://www.pygame.org/).
