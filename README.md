# F1-MARS: Formula 1 Multi-Agent Racing Simulator

A reinforcement learning environment for training autonomous F1 racing agents using realistic physics and authentic F1 circuit data.

## Features

- **Authentic F1 Circuits**: 7 real F1 tracks from CSV racing line data (Budapest, Spa, Monza, Catalunya, Austin, Nürburgring, Yas Marina)
- **Realistic Physics**: Tire wear, temperature degradation, fuel consumption, pit stops
- **F1 TV-Style Visualization**: GPU-accelerated Arcade rendering, broadcast-quality HUD, live telemetry, dynamic camera
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
│   ├── rendering/             # Arcade GPU visualization
│   │   ├── game_window.py    # Main Arcade window
│   │   ├── camera.py         # Racing camera with smooth follow
│   │   ├── car_sprite.py     # F1 car sprite with effects
│   │   ├── track_renderer.py # GPU batch rendering
│   │   ├── hud.py            # F1 TV broadcast-style HUD
│   │   └── effects.py        # Particle effects system
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

### 1. Train Your First Agent

All circuits are ready to use. Start with the easiest circuit (Monza) or your favorite F1 track:

```bash
# Easy: Monza - Wide track, high-speed (recommended for first training)
python scripts/train_agent.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/monza_ppo \
    --eval-freq 10000

# Medium: Budapest - Technical, tight corners
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/budapest_ppo \
    --eval-freq 10000

# Hard: Spa - Longest circuit, varied challenges
python scripts/train_agent.py \
    --track tracks/spa.json \
    --algorithm PPO \
    --timesteps 800000 \
    --output trained_models/spa_ppo \
    --eval-freq 10000
```

**Training time:** 1-2 hours on CPU for 500k timesteps.

### 2. Watch Your Agent Race

Use the interactive viewer to select model and track:

```bash
# Interactive mode - Choose from available models
python scripts/watch_agent.py
```

Or specify directly:

```bash
# Direct mode - Specify model and track
python scripts/watch_agent.py \
    --model trained_models/monza_ppo/best_model.zip \
    --track tracks/monza.json \
    --laps 5
```

**Viewer Controls:**
- `SPACE` : Pause/Resume
- `R` : Reset episode
- `H` : Toggle HUD display
- `D` : Toggle debug info
- `+/-` : Zoom in/out
- `ESC` : Quit

### 3. Optional: Convert Custom Circuits

If you have F1 racing line CSV data (`x_m,y_m,w_tr_right_m,w_tr_left_m`):

```bash
# Place CSV in tracks/csv/ and convert
python scripts/csv_to_track.py tracks/csv/your_circuit.csv --tolerance 3.5 --laps 5
```

The tolerance parameter controls simplification (higher = fewer points, better performance).

## Available Circuits

| Circuit | Length | Difficulty | Timesteps | Characteristics |
|---------|--------|-----------|-----------|-----------------|
| Monza | 5.78 km | ⭐ Easy | 500k | Wide track, high-speed, long straights |
| Catalunya | 4.63 km | ⭐⭐ Medium | 500k | Balanced, mixed speed corners |
| Yas Marina | 5.53 km | ⭐⭐ Medium | 500k | Modern, technical sections |
| Budapest | 4.36 km | ⭐⭐ Medium | 500k | Tight, low-speed, technical |
| Austin (COTA) | 5.49 km | ⭐⭐⭐ Medium-Hard | 600k | Technical corners, elevation |
| Nürburgring | 5.13 km | ⭐⭐⭐ Medium-Hard | 600k | Technical, varied corners |
| Spa-Francorchamps | 6.98 km | ⭐⭐⭐⭐ Hard | 800k | Longest, varied, challenging |

All circuits optimized using Douglas-Peucker algorithm with 3.5m tolerance, achieving 90-96% point reduction while preserving racing line accuracy.

**Recommended training order:** Monza → Catalunya → Budapest → Spa (easy to hard)

## Training Guide

### Quick Start - Train on Specific Circuits

Each circuit has different characteristics and difficulty. Train your agent on the circuits you want to master:

#### 🏁 Budapest (Hungaroring) - Technical, Tight Corners
**Difficulty:** Medium | **Best for:** Learning precise control

```bash
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/budapest_ppo \
    --eval-freq 10000
```

#### 🏁 Monza (Temple of Speed) - High-Speed, Long Straights
**Difficulty:** Easy | **Best for:** Speed control

```bash
python scripts/train_agent.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/monza_ppo \
    --eval-freq 10000
```

#### 🏁 Spa-Francorchamps - Long, Varied, Iconic
**Difficulty:** Hard | **Best for:** Endurance and variety

```bash
python scripts/train_agent.py \
    --track tracks/spa.json \
    --algorithm PPO \
    --timesteps 800000 \
    --output trained_models/spa_ppo \
    --eval-freq 10000
```

#### 🏁 Catalunya (Barcelona) - Balanced, Mixed Speed
**Difficulty:** Medium | **Best for:** All-around skills

```bash
python scripts/train_agent.py \
    --track tracks/catalunya.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/catalunya_ppo \
    --eval-freq 10000
```

#### 🏁 Austin (COTA) - Technical with Elevation
**Difficulty:** Medium-Hard | **Best for:** Complex racing

```bash
python scripts/train_agent.py \
    --track tracks/austin.json \
    --algorithm PPO \
    --timesteps 600000 \
    --output trained_models/austin_ppo \
    --eval-freq 10000
```

#### 🏁 Nürburgring (GP-Strecke) - Varied Corners
**Difficulty:** Medium-Hard | **Best for:** Technical variety

```bash
python scripts/train_agent.py \
    --track tracks/nuerburgring.json \
    --algorithm PPO \
    --timesteps 600000 \
    --output trained_models/nuerburgring_ppo \
    --eval-freq 10000
```

#### 🏁 Yas Marina (Abu Dhabi) - Modern, Technical Sections
**Difficulty:** Medium | **Best for:** Modern circuit mastery

```bash
python scripts/train_agent.py \
    --track tracks/yasmarina.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/yasmarina_ppo \
    --eval-freq 10000
```

### Progressive Training Strategy

If your agent struggles to complete circuits, use this curriculum:

**Step 1: Start Easy - Monza (Wide, Fast)**
```bash
python scripts/train_agent.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/progressive_agent
```

**Step 2: Medium Difficulty - Catalunya (Balanced)**
```bash
python scripts/train_agent.py \
    --track tracks/catalunya.json \
    --algorithm PPO \
    --timesteps 500000 \
    --model trained_models/progressive_agent/best_model.zip \
    --output trained_models/progressive_agent
```

**Step 3: Technical - Budapest (Tight)**
```bash
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --model trained_models/progressive_agent/best_model.zip \
    --output trained_models/progressive_agent
```

**Step 4: Challenge - Spa (Long, Varied)**
```bash
python scripts/train_agent.py \
    --track tracks/spa.json \
    --algorithm PPO \
    --timesteps 800000 \
    --model trained_models/progressive_agent/best_model.zip \
    --output trained_models/progressive_agent
```

### Training All Circuits Sequentially

To create a general agent that can drive on any circuit:

```bash
# Train on all 7 circuits
for circuit in monza catalunya budapest austin nuerburgring spa yasmarina; do
    echo "Training on $circuit..."
    python scripts/train_agent.py \
        --track tracks/${circuit}.json \
        --algorithm PPO \
        --timesteps 500000 \
        --model trained_models/multi_circuit/best_model.zip \
        --output trained_models/multi_circuit \
        --eval-freq 10000
done
```

### Troubleshooting Failed Laps

If your agent can't complete circuits:

1. **Increase training time:** 500k → 1M timesteps
2. **Use curriculum learning:** Start with easier circuits (Monza → Catalunya → Budapest)
3. **Try SAC algorithm:** Better for continuous control
4. **Check with visualization:** Use `watch_agent.py` to see what's failing

See [TRAINING.md](TRAINING.md) for comprehensive instructions:
- Algorithm selection (PPO vs SAC vs TD3)
- Hyperparameter tuning
- Performance metrics
- Advanced troubleshooting

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
- Arcade GPU renderer (OpenGL 3.3+)
- F1 TV-style HUD
- Real-time telemetry display
- Dynamic camera and effects

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
arcade>=3.3.3
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
- [Python Arcade](https://api.arcade.academy/) for GPU-accelerated visualization
