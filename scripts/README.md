# F1-MARS Training & Evaluation Scripts

Complete guide for training and evaluating F1-MARS agents (Pilot and Engineer) using different RL algorithms.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Training Pilot Agent](#training-pilot-agent)
- [Training Engineer Agent](#training-engineer-agent)
- [Evaluation](#evaluation)
- [Complete Workflows](#complete-workflows)
- [Monitoring & Debugging](#monitoring--debugging)

---

## Quick Start

### Train a Pilot (Basic)

```bash
# Default PPO on oval track
python scripts/train_pilot.py --total-timesteps 500000

# With curriculum learning (recommended for beginners)
python scripts/train_pilot.py --curriculum --total-timesteps 1000000

# Specific track
python scripts/train_pilot.py \
    --track tracks/monza.json \
    --total-timesteps 500000
```

### Evaluate a Model

```bash
# Basic evaluation
python scripts/evaluate.py --model trained_models/PPO_default_final.zip

# With plots and metrics
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --episodes 20 \
    --output results/
```

---

## Training Pilot Agent

The pilot agent controls steering, throttle, and braking. Train using `train_pilot.py`.

### Algorithm Comparison

#### PPO (Proximal Policy Optimization) - **RECOMMENDED**

**Best for:** Stable learning, general racing

```bash
python scripts/train_pilot.py \
    --algorithm PPO \
    --learning-rate 3e-4 \
    --batch-size 64 \
    --n-envs 8 \
    --total-timesteps 500000
```

**Pros:**
- ✅ Very stable and reliable
- ✅ Good sample efficiency
- ✅ Works well for continuous control
- ✅ Easy to tune

**Cons:**
- ⚠️ Can be slower than SAC for complex tasks

**When to use:**
- First time training
- Need stable, consistent results
- Limited computational resources

---

#### SAC (Soft Actor-Critic)

**Best for:** Exploration, complex tracks, fast convergence

```bash
python scripts/train_pilot.py \
    --algorithm SAC \
    --learning-rate 3e-4 \
    --batch-size 256 \
    --n-envs 16 \
    --total-timesteps 500000
```

**Pros:**
- ✅ Better exploration
- ✅ Off-policy (reuses past experience)
- ✅ Often converges faster
- ✅ Good for complex state spaces

**Cons:**
- ⚠️ Requires more hyperparameter tuning
- ⚠️ Higher memory usage
- ⚠️ Can be unstable with poor hyperparameters

**When to use:**
- Complex, technical tracks
- Want faster convergence
- Have computational resources
- Need good exploration

---

#### TD3 (Twin Delayed DDPG)

**Best for:** Precision driving, deterministic control

```bash
python scripts/train_pilot.py \
    --algorithm TD3 \
    --learning-rate 1e-3 \
    --batch-size 100 \
    --n-envs 8 \
    --total-timesteps 500000
```

**Pros:**
- ✅ Deterministic policy (predictable)
- ✅ Good for fine control
- ✅ Stable learning
- ✅ Lower memory than SAC

**Cons:**
- ⚠️ Less exploration than SAC
- ⚠️ Can get stuck in local optima
- ⚠️ Slower than SAC for some tasks

**When to use:**
- Need deterministic behavior
- Precision control important
- Time trials / qualifying laps

---

### Training Options

#### Basic Training

```bash
# Default settings (PPO, oval track, 8 envs)
python scripts/train_pilot.py --total-timesteps 500000
```

#### Specific Track

```bash
python scripts/train_pilot.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --total-timesteps 500000
```

#### Multi-Track Training

```bash
# Rotate through all tracks
python scripts/train_pilot.py \
    --multi-track \
    --total-timesteps 1000000 \
    --n-envs 16
```

#### Training by Difficulty

```bash
# Beginner tracks (difficulty 0)
python scripts/train_pilot.py --difficulty 0 --total-timesteps 200000

# Intermediate tracks (difficulty 1)
python scripts/train_pilot.py --difficulty 1 --total-timesteps 300000

# Advanced tracks (difficulty 2)
python scripts/train_pilot.py --difficulty 2 --total-timesteps 500000

# Expert tracks (difficulty 3)
python scripts/train_pilot.py --difficulty 3 --total-timesteps 1000000
```

#### Curriculum Learning (Automatic Progression)

```bash
# Start at basic level, automatically progress
python scripts/train_pilot.py \
    --curriculum \
    --total-timesteps 1000000 \
    --n-envs 8

# Start at intermediate level
python scripts/train_pilot.py \
    --curriculum \
    --curriculum-level 1 \
    --total-timesteps 500000
```

📖 **See:** [Curriculum Learning Guide](../docs/CURRICULUM_LEARNING.md)

#### Resume Training

```bash
# Continue from checkpoint
python scripts/train_pilot.py \
    --load-model trained_models/PPO_checkpoint_100000_steps.zip \
    --total-timesteps 500000
```

#### High-Performance Training

```bash
# Maximum speed (requires powerful CPU)
python scripts/train_pilot.py \
    --algorithm SAC \
    --n-envs 32 \
    --batch-size 256 \
    --total-timesteps 2000000 \
    --checkpoint-freq 100000
```

---

### Hyperparameters Reference

#### Learning Rate

```bash
# Conservative (stable but slow)
--learning-rate 1e-4

# Standard (recommended)
--learning-rate 3e-4

# Aggressive (faster but less stable)
--learning-rate 1e-3
```

#### Batch Size

```bash
# PPO
--batch-size 64    # Standard
--batch-size 128   # Larger batches

# SAC / TD3
--batch-size 256   # Recommended
--batch-size 512   # Large (if memory allows)
```

#### Parallel Environments

```bash
--n-envs 4    # Low resources
--n-envs 8    # Standard
--n-envs 16   # High performance
--n-envs 32   # Maximum (powerful CPU)
```

**Note:** More environments = faster data collection but higher CPU usage

#### Training Duration

```bash
--total-timesteps 100000    # Quick test
--total-timesteps 500000    # Standard training
--total-timesteps 1000000   # Full training
--total-timesteps 2000000   # Extended training
```

---

## Training Engineer Agent

The engineer agent makes strategic decisions about pit stops and tyre management. Train using `train_engineer.py`.

### Basic Training

```bash
python scripts/train_engineer.py \
    --track example_circuit \
    --timesteps 500000 \
    --learning-rate 1e-4
```

### Custom Configuration

```bash
python scripts/train_engineer.py \
    --track monza \
    --timesteps 1000000 \
    --learning-rate 1e-4 \
    --save-freq 50000 \
    --tensorboard
```

### Resume Training

```bash
python scripts/train_engineer.py \
    --track monza \
    --timesteps 500000 \
    --load-model trained_models/engineer_checkpoint_250000.zip
```

**Note:** Engineer uses DQN (discrete actions for strategy decisions)

---

## Evaluation

Evaluate trained models using `evaluate.py`.

### Basic Evaluation

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --episodes 10
```

**Outputs:**
- JSON report with metrics
- 4-panel visualization plots
- Console summary

### Evaluation on Specific Track

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_monza_final.zip \
    --track tracks/monza.json \
    --episodes 20 \
    --output results/monza/
```

### Compare Two Models

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_v1.zip \
    --compare trained_models/SAC_v1.zip \
    --episodes 10
```

**Output includes:**
- Side-by-side metrics comparison
- Winner determination per metric
- Percentage differences
- Comparison JSON report

### Record Video

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --record \
    --record-path recordings/best_lap.mp4
```

**Requires:** `pip install opencv-python`

### With Visualization

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --render \
    --episodes 5
```

### Metrics Collected

Evaluation provides comprehensive metrics:

- **Lap Times**: mean, std, best, worst
- **Completion Rate**: % of episodes completing ≥1 lap
- **On-Track Percentage**: % of time on track
- **Off-Track Incidents**: Count of track exits
- **Velocity**: max and mean speed
- **Tyre Wear**: per-lap and final wear
- **Reward**: total cumulative reward

📊 **See:** [Evaluation Guide](../docs/EVALUATION_GUIDE.md)

---

## Complete Workflows

### Workflow 1: Train from Scratch (PPO)

```bash
# 1. Train on oval track
python scripts/train_pilot.py \
    --track tracks/oval.json \
    --algorithm PPO \
    --total-timesteps 200000 \
    --model-dir models/stage1/

# 2. Evaluate
python scripts/evaluate.py \
    --model models/stage1/PPO_oval_final.zip \
    --track tracks/oval.json \
    --episodes 20

# 3. Transfer to complex track
python scripts/train_pilot.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --total-timesteps 500000 \
    --load-model models/stage1/PPO_oval_final.zip \
    --model-dir models/stage2/

# 4. Final evaluation
python scripts/evaluate.py \
    --model models/stage2/PPO_monza_final.zip \
    --track tracks/monza.json \
    --episodes 50 \
    --record
```

---

### Workflow 2: Curriculum Learning (Automatic)

```bash
# 1. Train with automatic curriculum
python scripts/train_pilot.py \
    --curriculum \
    --algorithm PPO \
    --total-timesteps 1000000 \
    --n-envs 8 \
    --tensorboard-log logs/curriculum/

# 2. Monitor progress
tensorboard --logdir logs/curriculum/

# 3. Evaluate final model
python scripts/evaluate.py \
    --model trained_models/PPO_multi_final.zip \
    --episodes 30 \
    --output results/curriculum/

# 4. Test on different tracks
for track in tracks/*.json; do
    python scripts/evaluate.py \
        --model trained_models/PPO_multi_final.zip \
        --track "$track" \
        --episodes 10
done
```

---

### Workflow 3: Algorithm Comparison

```bash
# Train with each algorithm
for algo in PPO SAC TD3; do
    python scripts/train_pilot.py \
        --algorithm $algo \
        --track tracks/monza.json \
        --total-timesteps 500000 \
        --model-dir models/${algo}/
done

# Compare all three
python scripts/evaluate.py \
    --model models/PPO/PPO_monza_final.zip \
    --compare models/SAC/SAC_monza_final.zip \
    --episodes 20

python scripts/evaluate.py \
    --model models/SAC/SAC_monza_final.zip \
    --compare models/TD3/TD3_monza_final.zip \
    --episodes 20
```

---

### Workflow 4: Hyperparameter Search

```bash
# Test different learning rates
for lr in 1e-4 3e-4 1e-3; do
    python scripts/train_pilot.py \
        --algorithm PPO \
        --learning-rate $lr \
        --total-timesteps 200000 \
        --model-dir models/lr_${lr}/
done

# Evaluate each
for lr in 1e-4 3e-4 1e-3; do
    python scripts/evaluate.py \
        --model models/lr_${lr}/PPO_default_final.zip \
        --episodes 10 \
        --output results/lr_${lr}/
done
```

---

### Workflow 5: Full Training Pipeline

Complete end-to-end training:

```bash
#!/bin/bash

# Stage 1: Basic training
echo "Stage 1: Basic training on oval..."
python scripts/train_pilot.py \
    --track tracks/oval.json \
    --total-timesteps 200000 \
    --model-dir models/stage1/

# Stage 2: Intermediate track
echo "Stage 2: Transfer to simple circuit..."
python scripts/train_pilot.py \
    --track tracks/simple.json \
    --total-timesteps 300000 \
    --load-model models/stage1/PPO_oval_final.zip \
    --model-dir models/stage2/

# Stage 3: Advanced track
echo "Stage 3: Transfer to technical circuit..."
python scripts/train_pilot.py \
    --track tracks/technical.json \
    --total-timesteps 400000 \
    --load-model models/stage2/PPO_simple_final.zip \
    --model-dir models/stage3/

# Stage 4: Multi-track generalization
echo "Stage 4: Multi-track training..."
python scripts/train_pilot.py \
    --multi-track \
    --total-timesteps 1000000 \
    --load-model models/stage3/PPO_technical_final.zip \
    --model-dir models/final/

# Evaluation
echo "Final evaluation..."
python scripts/evaluate.py \
    --model models/final/PPO_multi_final.zip \
    --episodes 50 \
    --output results/final/ \
    --record

echo "Training complete! Check results/ for evaluation."
```

---

## Monitoring & Debugging

### TensorBoard

Monitor training in real-time:

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser: http://localhost:6006
```

**Key Metrics to Watch:**

1. **rollout/ep_rew_mean**: Episode reward (should increase)
2. **f1/lap_time_best**: Best lap time (should decrease)
3. **f1/on_track_percentage**: Time on track (should approach 100%)
4. **f1/laps_completed_mean**: Laps per episode (should increase)
5. **train/loss**: Training loss (should stabilize)

### Common Issues

#### Training is Slow

```bash
# Reduce environments
--n-envs 4

# Or increase if CPU underutilized
--n-envs 16
```

**Note:** CPU is optimal for RL. GPU can be tested with `--device cuda` but usually slower.

#### Agent Not Learning

```bash
# Try lower learning rate
--learning-rate 1e-4

# More training time
--total-timesteps 1000000

# Different algorithm
--algorithm SAC

# Easier track
--difficulty 0
# Or use curriculum
--curriculum
```

#### Agent Crashes Frequently

- Train longer on easier tracks first
- Use curriculum learning
- Check reward function (may need adjustment)

#### Memory Issues

```bash
# Reduce environments
--n-envs 4

# Reduce batch size
--batch-size 32

# CPU uses less memory than GPU (already default)
```

### Quick Debug Run

```bash
# Minimal test (fast)
python scripts/train_pilot.py \
    --total-timesteps 10000 \
    --n-envs 2 \
    --checkpoint-freq 5000 \
    --eval-freq 2000
```

---

## Output Files

### Training Outputs

```
trained_models/
├── PPO_checkpoint_50000_steps.zip    # Checkpoint at 50k
├── PPO_checkpoint_100000_steps.zip   # Checkpoint at 100k
├── best_model.zip                     # Best eval performance
└── PPO_default_final.zip              # Final model

logs/
└── PPO_1/
    └── events.out.tfevents.*          # TensorBoard logs
```

### Evaluation Outputs

```
results/
├── PPO_default_evaluation.json        # Metrics JSON
├── PPO_default_plots.png              # Visualization plots
├── PPO_default_recording.mp4          # Video (if --record)
└── comparison.json                    # Model comparison (if --compare)
```

---

## Tips for Best Results

1. **Start Simple**: Begin with oval track (difficulty 0) or use `--curriculum`
2. **Use PPO First**: Most stable and reliable for beginners
3. **Monitor Training**: Watch TensorBoard to detect issues early
4. **Save Checkpoints**: Keep multiple checkpoints (default: every 50k steps)
5. **Evaluate Often**: Use eval callback to track progress
6. **Transfer Learning**: Load previous models when changing tracks
7. **Parallel Environments**: Use 8-16 envs for good speed/resource balance
8. **Be Patient**: Good policies need 500k-1M timesteps
9. **Curriculum Learning**: Use for best generalization across tracks
10. **Compare Algorithms**: Try PPO, SAC, and TD3 to see what works best

---

## Algorithm Selection Guide

| Scenario | Recommended Algorithm | Why |
|----------|----------------------|-----|
| First time training | **PPO** | Most stable, easy to tune |
| Simple/oval track | **PPO** | Efficient for simple tasks |
| Complex technical track | **SAC** | Better exploration |
| Need fast convergence | **SAC** | Off-policy learning |
| Deterministic control | **TD3** | Predictable behavior |
| Limited resources | **PPO** | Lower memory usage |
| Time trials | **TD3** or **SAC** | Precision control |
| Multi-track generalization | **PPO** + curriculum | Stable across conditions |

---

## Additional Documentation

- 📖 [Curriculum Learning Guide](../docs/CURRICULUM_LEARNING.md) - Progressive difficulty training
- 📊 [Evaluation Guide](../docs/EVALUATION_GUIDE.md) - Complete evaluation documentation
- 🏁 [Track Creation Guide](../tracks/README.md) - Create custom tracks
- 📚 [Main README](../README.md) - Project overview

---

## Quick Reference

### train_pilot.py

```bash
python scripts/train_pilot.py \
    --algorithm {PPO,SAC,TD3} \
    --total-timesteps N \
    --track PATH \
    --n-envs N \
    --learning-rate LR \
    --batch-size N \
    --curriculum \
    --load-model PATH
```

### train_engineer.py

```bash
python scripts/train_engineer.py \
    --track NAME \
    --timesteps N \
    --learning-rate LR \
    --save-freq N
```

### evaluate.py

```bash
python scripts/evaluate.py \
    --model PATH \
    --track PATH \
    --episodes N \
    --compare PATH \
    --render \
    --record
```

---

**Happy Training! 🏎️💨**
