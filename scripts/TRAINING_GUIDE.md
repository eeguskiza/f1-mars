# F1-MARS Training Guide

Guide for training the pilot agent using `train_pilot.py`.

## Quick Start

### Basic Training (Default Track)

```bash
python scripts/train_pilot.py --total-timesteps 100000
```

This will:
- Use PPO algorithm
- Train on the default oval track
- Use 8 parallel environments
- Save checkpoints every 50,000 steps
- Evaluate every 10,000 steps

### Training on Specific Track

```bash
python scripts/train_pilot.py \
    --track tracks/oval.json \
    --total-timesteps 500000 \
    --algorithm PPO
```

### Multi-Track Training (Curriculum Learning)

```bash
python scripts/train_pilot.py \
    --multi-track \
    --total-timesteps 1000000 \
    --n-envs 16
```

Trains on all available tracks, rotating through them across environments.

### Training by Difficulty Level

```bash
# Train on all beginner tracks (difficulty 0)
python scripts/train_pilot.py --difficulty 0 --total-timesteps 200000

# Train on intermediate tracks (difficulty 1)
python scripts/train_pilot.py --difficulty 1 --total-timesteps 300000

# Train on advanced tracks (difficulty 2)
python scripts/train_pilot.py --difficulty 2 --total-timesteps 500000

# Train on expert tracks (difficulty 3)
python scripts/train_pilot.py --difficulty 3 --total-timesteps 1000000
```

## Algorithm Comparison

### PPO (Proximal Policy Optimization)
**Best for:** General racing, stable learning

```bash
python scripts/train_pilot.py \
    --algorithm PPO \
    --learning-rate 3e-4 \
    --batch-size 64 \
    --total-timesteps 500000
```

**Pros:**
- Stable and reliable
- Good sample efficiency
- Works well for continuous control

**Cons:**
- Can be slower than SAC for some tasks

### SAC (Soft Actor-Critic)
**Best for:** Exploration, complex tracks

```bash
python scripts/train_pilot.py \
    --algorithm SAC \
    --learning-rate 3e-4 \
    --batch-size 256 \
    --total-timesteps 500000
```

**Pros:**
- Better exploration
- Off-policy (more sample efficient)
- Good for complex state spaces

**Cons:**
- Requires more hyperparameter tuning
- Higher memory usage

### TD3 (Twin Delayed DDPG)
**Best for:** Precision driving, time trials

```bash
python scripts/train_pilot.py \
    --algorithm TD3 \
    --learning-rate 1e-3 \
    --batch-size 100 \
    --total-timesteps 500000
```

**Pros:**
- Deterministic policy
- Good for fine control
- Stable learning

**Cons:**
- Less exploration than SAC
- Can get stuck in local optima

## Curriculum Learning Workflow

### Stage 1: Beginner (Oval Track)

```bash
python scripts/train_pilot.py \
    --track tracks/oval.json \
    --total-timesteps 200000 \
    --algorithm PPO \
    --model-dir models/stage1
```

**Goal:** Learn basic controls, complete consistent laps

### Stage 2: Transfer to Simple Circuit

```bash
python scripts/train_pilot.py \
    --track tracks/simple.json \
    --total-timesteps 300000 \
    --algorithm PPO \
    --load-model models/stage1/PPO_oval_final.zip \
    --model-dir models/stage2
```

**Goal:** Master braking zones, hairpin corners

### Stage 3: Technical Mastery

```bash
python scripts/train_pilot.py \
    --track tracks/technical.json \
    --total-timesteps 400000 \
    --algorithm PPO \
    --load-model models/stage2/PPO_simple_final.zip \
    --model-dir models/stage3
```

**Goal:** Precision through chicanes, consistency

### Stage 4: Multi-Track Generalization

```bash
python scripts/train_pilot.py \
    --multi-track \
    --total-timesteps 1000000 \
    --algorithm PPO \
    --load-model models/stage3/PPO_technical_final.zip \
    --model-dir models/stage4
```

**Goal:** Generalize across all track types

## Advanced Options

### High-Performance Training

```bash
python scripts/train_pilot.py \
    --algorithm SAC \
    --total-timesteps 2000000 \
    --n-envs 32 \
    --batch-size 256 \
    --learning-rate 3e-4 \
    --checkpoint-freq 100000 \
    --eval-freq 20000
```

Uses 32 parallel environments for faster training. CPU is the default device (optimal for RL with parallel environments).

### Debugging / Quick Test

```bash
python scripts/train_pilot.py \
    --total-timesteps 10000 \
    --n-envs 2 \
    --eval-freq 2000 \
    --checkpoint-freq 5000
```

Quick test with minimal resources.

### Resume Training

```bash
python scripts/train_pilot.py \
    --load-model trained_models/PPO_checkpoint_100000_steps.zip \
    --total-timesteps 500000
```

Continue training from a checkpoint.

## Monitoring Training

### View TensorBoard

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 to view:
- Reward curves
- Lap times
- Tyre wear
- On-track percentage
- Loss metrics

### Key Metrics to Watch

1. **episode_reward_mean**: Should increase over time
2. **f1/lap_time_best**: Should decrease as agent improves
3. **f1/on_track_percentage**: Should approach 100%
4. **f1/laps_completed_mean**: Should increase
5. **eval/mean_reward**: Best indicator of performance

## Hyperparameter Tuning

### Learning Rate

```bash
# Conservative (stable but slow)
--learning-rate 1e-4

# Standard (recommended)
--learning-rate 3e-4

# Aggressive (faster but less stable)
--learning-rate 1e-3
```

### Batch Size

```bash
# Small (less memory, noisier updates)
--batch-size 32

# Medium (recommended)
--batch-size 64

# Large (more stable, requires more memory)
--batch-size 256
```

### Parallel Environments

```bash
# Low resources
--n-envs 4

# Standard
--n-envs 8

# High performance
--n-envs 16

# Maximum (requires powerful CPU)
--n-envs 32
```

## Common Issues

### Training is Slow
- Reduce `--n-envs` if CPU is bottleneck
- Increase `--n-envs` if CPU has spare cores
- CPU is optimal for RL (GPU can be tested with `--device cuda` but usually slower due to transfer overhead)

### Agent Not Learning
- Reduce learning rate: `--learning-rate 1e-4`
- Increase training time: `--total-timesteps 1000000`
- Try different algorithm: `--algorithm SAC`
- Start with easier track: `--difficulty 0`

### Agent Crashes Often
- Increase penalty for going off-track (modify reward function)
- Train longer on easier tracks first
- Use curriculum learning

### Memory Issues
- Reduce `--n-envs`
- Reduce `--batch-size`
- CPU is already the default (uses less memory than GPU)

## File Outputs

After training, you'll find:

```
trained_models/
├── PPO_checkpoint_50000_steps.zip
├── PPO_checkpoint_100000_steps.zip
├── best_model.zip                      # Best performing model
└── PPO_oval_final.zip                  # Final model

logs/
└── PPO_1/
    └── events.out.tfevents.*           # TensorBoard logs
```

## Example Workflow

Complete training pipeline:

```bash
# 1. Start with oval track
python scripts/train_pilot.py \
    --track tracks/oval.json \
    --total-timesteps 200000 \
    --model-dir models/oval

# 2. Evaluate
python scripts/evaluate.py \
    --model models/oval/PPO_oval_final.zip \
    --track tracks/oval.json

# 3. Transfer to next difficulty
python scripts/train_pilot.py \
    --track tracks/simple.json \
    --total-timesteps 300000 \
    --load-model models/oval/PPO_oval_final.zip \
    --model-dir models/simple

# 4. Multi-track training
python scripts/train_pilot.py \
    --multi-track \
    --total-timesteps 1000000 \
    --load-model models/simple/PPO_simple_final.zip \
    --model-dir models/final

# 5. View results
tensorboard --logdir logs/
```

## Tips for Best Results

1. **Start Simple**: Begin with oval track (difficulty 0)
2. **Use Curriculum**: Progressively increase difficulty
3. **Monitor Metrics**: Watch TensorBoard to detect issues early
4. **Save Checkpoints**: Keep multiple checkpoints to revert if needed
5. **Tune Hyperparameters**: Adjust learning rate based on training curves
6. **Use Transfer Learning**: Load previous models when changing tracks
7. **Parallel Training**: Use multiple environments for faster training
8. **Evaluate Regularly**: Check performance on separate eval environment
9. **Be Patient**: Good policies need 500k-1M timesteps
10. **Experiment**: Try different algorithms and hyperparameters
