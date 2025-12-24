# Curriculum Learning for F1-MARS

Curriculum learning is a training technique that progressively increases task difficulty, starting with simple scenarios and gradually advancing to complex ones. This often leads to faster learning and better final performance.

## Overview

The `CurriculumWrapper` automatically adjusts environment difficulty based on agent performance across 4 levels:

| Level | Name         | Track Difficulty | Tyre Wear | Initial Speed | Extra Rewards |
|-------|--------------|------------------|-----------|---------------|---------------|
| 0     | Basic        | Easy oval        | None (0x) | 20 m/s        | ✓ Progress bonus |
| 1     | Intermediate | Moderate curves  | Half (0.5x) | Stopped     | None |
| 2     | Advanced     | Complex tracks   | Normal (1x) | Stopped     | None |
| 3     | Expert       | Hard tracks      | High (1.5x) | Stopped     | None |

## How It Works

### Automatic Level Progression

The wrapper tracks agent performance using:
- **Success rate**: % of episodes completing at least 1 lap
- **Lap times**: Average time to complete laps

**Advancement criteria** (moving to next level):
- Success rate ≥ level threshold (60-80% depending on level)
- Minimum 20 episodes at current level

**Retreat criteria** (moving to previous level):
- Success rate < 30% sustained
- Minimum 50 episodes at current level

### Level Settings

#### Level 0 (Basic)
- **Purpose**: Learn basic controls and track following
- **Track**: Simple oval with gentle curves
- **Modifications**:
  - No tyre wear (car performance constant)
  - Starts with initial velocity (easier to maintain control)
  - Progress bonus reward (encourages forward movement)
- **Success criteria**: 60% of episodes complete 1 lap

#### Level 1 (Intermediate)
- **Purpose**: Handle moderate complexity
- **Track**: Circuits with moderate curves
- **Modifications**:
  - Half tyre wear (degradation introduced gradually)
  - Start from stop (learn acceleration)
- **Success criteria**: 70% of episodes complete 1 lap

#### Level 2 (Advanced)
- **Purpose**: Master full mechanics
- **Track**: Complex circuits
- **Modifications**:
  - Normal tyre wear
  - All standard mechanics active
- **Success criteria**: 75% of episodes complete 1 lap

#### Level 3 (Expert)
- **Purpose**: Handle challenging scenarios
- **Track**: Difficult circuits
- **Modifications**:
  - Increased tyre wear (1.5x degradation)
  - Maximum challenge
- **Success criteria**: 80% of episodes complete 1 lap

## Usage

### Training with Curriculum Learning

Use the `--curriculum` flag when training:

```bash
# Start from basic level
python scripts/train_pilot.py --curriculum --total-timesteps 500000

# Start from intermediate level (if you have a partially trained model)
python scripts/train_pilot.py --curriculum --curriculum-level 1 \
    --load-model trained_models/pilot_checkpoint.zip
```

### Complete Example

```bash
python scripts/train_pilot.py \
    --algorithm PPO \
    --curriculum \
    --curriculum-level 0 \
    --total-timesteps 1000000 \
    --n-envs 8 \
    --learning-rate 3e-4 \
    --tensorboard-log logs/curriculum/
```

### Programmatic Usage

```python
from f1_mars.envs import F1Env, CurriculumWrapper

# Create base environment
env = F1Env()

# Wrap with curriculum (starts at level 0)
env = CurriculumWrapper(env, initial_level=0)

# Or use convenience function
from f1_mars.envs import wrap_with_curriculum
env = wrap_with_curriculum(env, initial_level=0)
```

### Custom Configuration

```python
from f1_mars.envs import CurriculumWrapper

# Custom progression settings
config = {
    "window_size": 30,              # Consider last 30 episodes
    "min_episodes_advance": 30,     # Need 30 episodes before advancing
    "min_episodes_retreat": 100,    # Need 100 episodes before retreating
    "retreat_threshold": 0.2        # Retreat if success < 20%
}

env = CurriculumWrapper(base_env, config=config, initial_level=0)
```

### Manual Level Control

For testing or debugging, manually set the level:

```python
# Override to specific level
env.set_level(2)  # Jump to Advanced level

# Check current state
info = env.get_curriculum_info()
print(f"Level: {info['level']} ({info['level_name']})")
print(f"Success rate: {info['success_rate']:.1%}")
```

## Monitoring Progress

### TensorBoard Logging

Curriculum metrics are automatically logged:
- `curriculum/level`: Current level (0-3)
- `curriculum/episodes_at_level`: Episodes at current level
- `curriculum/success_rate`: Recent success rate
- `curriculum/avg_lap_time`: Average lap completion time

View during training:
```bash
tensorboard --logdir logs/curriculum/
```

### Console Output

Level changes are logged to console:
```
[Curriculum] 📈 ADVANCED: Level 0 → 1 (Intermediate) after 150 episodes (success rate: 82.5%)
```

### Curriculum Info in Episodes

Each step returns curriculum info in the `info` dict:

```python
obs, info = env.reset()
curriculum_info = info['curriculum']

# Available fields:
# - level: Current level (0-3)
# - level_name: "Basic", "Intermediate", etc.
# - episode_count: Total episodes
# - episodes_at_level: Episodes at current level
# - success_rate: Recent success rate
# - avg_lap_time: Average lap time
# - target_lap_time: Target for current level
# - success_threshold: Threshold to advance
```

## Best Practices

### When to Use Curriculum Learning

**Use curriculum when:**
- Training from scratch (no pre-trained model)
- Agent struggles to learn on complex tracks
- You want more robust generalization
- Training time is not the primary constraint

**Don't use curriculum when:**
- Fine-tuning an already trained model
- Focusing on a specific track (use that track directly)
- You need fastest possible training on a single task

### Combining with Other Techniques

Curriculum works well with:
- **Parallel environments**: Speed up data collection at each level
- **Algorithm choice**: PPO generally works well across levels
- **Multi-track training**: Let curriculum choose tracks by difficulty

Example combining techniques:
```bash
python scripts/train_pilot.py \
    --curriculum \
    --algorithm PPO \
    --n-envs 16 \
    --total-timesteps 2000000
```

### Troubleshooting

**Agent stuck at early level:**
- Check tensorboard for success rate trends
- Verify tracks are available for each difficulty
- Consider lowering success thresholds in config

**Oscillating between levels:**
- Increase `min_episodes_before_retreat`
- Check if evaluation window (`window_size`) is too small

**Too slow progression:**
- Decrease success thresholds
- Reduce `min_episodes_advance`
- Ensure `n_envs` is high enough for data collection

## Examples

See:
- `examples/curriculum_training_example.py` - Complete training example
- `tests/test_curriculum_wrapper.py` - Usage patterns and tests

## Implementation Details

The curriculum wrapper is implemented as a Gymnasium wrapper that:
1. Wraps the base F1Env
2. Intercepts `reset()` to evaluate progress and change levels
3. Modifies `step()` to add level-specific reward bonuses
4. Tracks performance metrics in a sliding window
5. Automatically adjusts environment parameters per level

The wrapper is compatible with:
- Stable-Baselines3 algorithms (PPO, SAC, TD3)
- Vectorized environments (SubprocVecEnv, DummyVecEnv)
- TensorBoard logging
- Custom callbacks

## References

- Paper: [Curriculum Learning (Bengio et al., 2009)](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)
- SB3 Docs: [Custom Wrappers](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
