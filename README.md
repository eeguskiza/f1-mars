# F1-MARS: Formula 1 Multi-Agent Racing Simulator

A reinforcement learning environment for training autonomous F1 racing agents using realistic physics and authentic F1 circuit data.

## Features

- **Authentic F1 Circuits**: 7 real F1 tracks from CSV racing line data (Budapest, Spa, Monza, Catalunya, Austin, Nürburgring, Yas Marina)
- **Realistic Physics**: Tire wear, temperature degradation, fuel consumption, pit stops
- **F1 TV-Style Visualization**: Broadcast-quality HUD with live telemetry, speed effects, dynamic camera
- **Multiple Algorithms**: PPO, SAC, TD3 support via Stable-Baselines3
- **Optimized Performance**: Douglas-Peucker path simplification, 90%+ point reduction

## Project Structure

```
f1_mars/
├── f1_mars/                    # Core package
│   ├── envs/                   # Gymnasium environments
│   │   ├── f1_env.py          # Main F1 environment
│   │   └── components/        # Physics components
│   │       ├── car.py         # Vehicle dynamics (bicycle model)
│   │       ├── track.py       # Track geometry and boundaries
│   │       ├── tyres.py       # Tire physics and degradation
│   │       └── fuel.py        # Fuel consumption system
│   ├── rendering/             # PyGame visualization
│   │   ├── renderer.py        # Main renderer with camera system
│   │   ├── hud.py            # F1 TV broadcast-style HUD
│   │   ├── sprites.py        # Car sprites and visual effects
│   │   └── colors.py         # F1 color palette
│   └── utils/                 # Utility functions
├── scripts/                   # Training and conversion tools
│   ├── train_agent.py        # Main RL training script
│   ├── watch_agent.py        # Model visualization tool
│   ├── csv_to_track.py       # Circuit CSV to JSON converter
│   └── convert_all_circuits.sh # Batch circuit conversion
├── tracks/                    # F1 circuit definitions (JSON)
│   └── csv/                  # Source CSV racing line data
├── docs/                      # Project documentation
└── tests/                     # Test suite

```

## Installation

### Requirements

- Python 3.8 or higher
- CPU recommended for training (MLP policies)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd f1_mars

# Create virtual environment
python -m venv rl
source rl/bin/activate  # Windows: rl\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Convert F1 Circuit Data

Place CSV files in `tracks/csv/` with format: `x_m,y_m,w_tr_right_m,w_tr_left_m`

```bash
# Convert all circuits at once
for csv in tracks/csv/*.csv; do
    python scripts/csv_to_track.py "$csv" --tolerance 3.5 --laps 5
done
```

The tolerance parameter controls simplification (higher = fewer points, better performance).

### 2. Train an Agent

```bash
# Train on Budapest (technical circuit, recommended for learning)
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/budapest_ppo \
    --eval-freq 10000
```

Expected training time: 1-2 hours on CPU for 500k timesteps.

### 3. Visualize Results

```bash
python scripts/watch_agent.py \
    --model trained_models/budapest_ppo/best_model.zip \
    --track tracks/budapest.json \
    --laps 5
```

**Viewer Controls:**
- `+/-` : Adjust zoom
- `H` : Toggle HUD display
- `T` : Toggle trajectory trail
- `SPACE` : Pause/Resume
- `R` : Reset episode
- `ESC` : Exit

## Available Circuits

| Circuit | Real Length | Width | Points | Characteristics |
|---------|-------------|-------|--------|-----------------|
| Austin (COTA) | 5.49 km | 14.0m | 71 | Technical corners, elevation |
| Budapest | 4.36 km | 9.8m | 59 | Tight, low-speed, technical |
| Catalunya | 4.63 km | 11.2m | 66 | Balanced, mixed speed |
| Monza | 5.78 km | 9.3m | 46 | High-speed, long straights |
| Nürburgring | 5.13 km | 12.1m | 66 | Technical, varied corners |
| Spa-Francorchamps | 6.98 km | 9.6m | 79 | Longest, high-speed, iconic |
| Yas Marina | 5.53 km | 12.7m | 59 | Modern, technical sections |

All circuits optimized using Douglas-Peucker algorithm with 3.5m tolerance, achieving 90-96% point reduction while preserving racing line accuracy.

## Training Guide

See [TRAINING.md](TRAINING.md) for comprehensive instructions:
- Algorithm selection (PPO vs SAC vs TD3)
- Single vs multi-circuit training strategies
- Hyperparameter tuning
- Performance metrics
- Troubleshooting common issues

### Recommended Training Strategy

**For beginners:**
1. Start with Budapest (4.36 km, technical)
2. Train 500k timesteps with PPO
3. Evaluate performance
4. If successful, expand to other circuits

**For generalization:**
Train sequentially on varied circuits:
- Budapest (technical)
- Monza (high-speed)
- Spa (long, varied)
- Catalunya (balanced)

## Environment Specifications

### Observation Space (26 dimensions)

- **Motion state**: velocity, steering angle, position (4 values)
- **LIDAR sensors**: 11-ray array spanning -75° to +75° (11 values)
- **Track curvature**: upcoming track curvature (5 samples ahead)
- **Tire state**: wear percentage, grip level, temperature
- **Progress**: lap number, checkpoint progress
- **Strategy**: race engineer signal (pit stop recommendation)

### Action Space

**Pilot Agent**: Continuous control
- Steering: [-1, 1] (left/right)
- Throttle: [0, 1] (acceleration)
- Brake: [0, 1] (braking force)

**Race Engineer**: Discrete strategy (Phase 5, in development)
- Pit stop timing
- Tire compound selection

### Reward Function

- **Forward progress**: +0.1 per meter
- **Checkpoint**: +10 points
- **Lap completion**: +100 + time bonus
- **Track limits violation**: -5 per timestep
- **Pit stop penalties**: -0.5 entry, -1.0 per stopped timestep

Detailed reward structure in `f1_mars/envs/f1_env.py`

## Performance Optimization

**Circuit Simplification:**
- Original: 876-1401 points per circuit
- Optimized: 46-79 points (90-96% reduction)
- Method: Douglas-Peucker algorithm with 3.5m tolerance
- Result: Preserved accuracy, 10-15x rendering speedup

**Rendering:**
- Fixed zoom camera (no automatic adjustments)
- Realistic car/track proportions (F1 car = 5m length)
- Optimized sprite caching
- Efficient track boundary rendering

**Training:**
CPU is recommended for MLP policies. GPU provides no significant speedup for small neural networks used in continuous control. The environment simulation (physics) is the bottleneck, not the neural network training.

## Physics Model

### Tire System
- **Degradation**: Speed, lateral forces, temperature dependent
- **Temperature**: Optimal range 80-105°C, affects grip
- **Compounds**: Soft (fast, degrades quickly), Medium (balanced), Hard (durable, slower)

### Vehicle Dynamics
- Bicycle model for 2D racing
- Maximum speed: 350 km/h on straights
- Acceleration/braking physics
- Steering response curves

### Track System
- Spline-based racing line
- Variable track width
- Track limits enforcement
- Pit lane entry/exit zones

### Pit Stops (Phase 5)
- Tire compound changes
- Fuel refilling
- Strategic timing
- Time penalties

Full physics documentation in component source files.

## Documentation

- [TRAINING.md](TRAINING.md) - Complete training guide
- [docs/CURRICULUM_LEARNING.md](docs/CURRICULUM_LEARNING.md) - Progressive difficulty training
- [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) - Model evaluation and metrics
- [docs/PIT_STOP_IMPLEMENTATION.md](docs/PIT_STOP_IMPLEMENTATION.md) - Pit stop mechanics
- [tracks/README.md](tracks/README.md) - Track system and CSV format
- [tracks/csv/README.md](tracks/csv/README.md) - Circuit data format

## Development Status

**Phase 1: Core Environment** - COMPLETE
- Car physics and dynamics
- Track geometry system
- Gymnasium interface

**Phase 2: Physics Components** - COMPLETE
- Tire wear and temperature
- Fuel consumption
- Pit stop mechanics

**Phase 3: Visualization** - COMPLETE
- PyGame renderer
- F1 TV-style HUD
- Real-time telemetry display

**Phase 4: Training System** - COMPLETE
- PPO, SAC, TD3 algorithms
- Single and multi-circuit training
- Performance optimization

**Phase 5: Multi-Agent System** - IN PROGRESS
- Pilot agent (driving)
- Race engineer agent (strategy)
- Coordinated decision making

## Testing

```bash
# Run all tests
pytest tests/ -v

# Specific test suites
pytest tests/test_environment.py -v   # Environment tests
pytest tests/test_tyres.py -v         # Tire physics tests
pytest tests/test_tracks.py -v        # Track system tests
pytest tests/test_car.py -v           # Vehicle dynamics tests
```

Test coverage: 28+ unit tests covering all core components.

## Requirements

```
gymnasium>=0.29.0
stable-baselines3>=2.0.0
pygame>=2.5.0
numpy>=1.24.0
scipy>=1.11.0
```

See `requirements.txt` for complete dependency list.

## Contributing

Contributions are welcome. Please:
1. Open an issue to discuss proposed changes
2. Fork the repository
3. Create a feature branch
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Citation

If you use this project in research, please cite:

```bibtex
@software{f1mars2024,
  title={F1-MARS: Formula 1 Multi-Agent Racing Simulator},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/f1_mars}
}
```

## Acknowledgments

- Real F1 circuit racing line data from open-source datasets
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) for environment framework
- [PyGame](https://www.pygame.org/) for visualization
