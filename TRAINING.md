# F1-MARS Training Guide

Comprehensive guide for training reinforcement learning agents on real F1 circuits.

## Available Circuits

7 authentic F1 circuits converted from racing line data:

1. **Austin** (Circuit of The Americas) - 5.49 km
2. **Budapest** (Hungaroring) - 4.36 km
3. **Catalunya** (Barcelona) - 4.63 km
4. **Monza** (Temple of Speed) - 5.78 km
5. **Nürburgring** (GP-Strecke) - 5.13 km
6. **Spa-Francorchamps** - 6.98 km (longest)
7. **Yas Marina** (Abu Dhabi) - 5.53 km

All circuits optimized with Douglas-Peucker simplification (90-96% point reduction).

## Training Strategies

### Option 1: Single Circuit Training (Specialized Agent)

Train an agent to master a specific circuit.

**Budapest Example** (technical circuit, recommended for beginners):

```bash
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/budapest_ppo \
    --eval-freq 10000
```

**Other Circuit Examples:**

```bash
# Monza (high-speed, long straights)
python scripts/train_agent.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/monza_ppo

# Spa (long, technical, varied)
python scripts/train_agent.py \
    --track tracks/spa.json \
    --algorithm PPO \
    --timesteps 800000 \
    --output trained_models/spa_ppo
```

### Option 2: Multi-Circuit Training (Generalization)

Train an agent that can drive on multiple different circuits.

**Using the batch training script:**

```bash
chmod +x scripts/train_real_circuits.sh
bash scripts/train_real_circuits.sh PPO 500000
```

**Manual sequential training for more control:**

```bash
# 1. Start with Budapest (technical)
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/multi_circuit_ppo

# 2. Continue training on Monza
python scripts/train_agent.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --timesteps 500000 \
    --model trained_models/multi_circuit_ppo/best_model.zip \
    --output trained_models/multi_circuit_ppo

# 3. Continue on Spa
python scripts/train_agent.py \
    --track tracks/spa.json \
    --algorithm PPO \
    --timesteps 500000 \
    --model trained_models/multi_circuit_ppo/best_model.zip \
    --output trained_models/multi_circuit_ppo

# Continue with other circuits as needed
```

## Algorithm Selection

### PPO (Proximal Policy Optimization) - RECOMMENDED

**Best for:**
- General purpose training
- Stable learning
- Technical circuits

**Characteristics:**
- Most stable convergence
- Good sample efficiency
- Works well on all circuit types

**Recommended timesteps:** 500k - 1M

```bash
python scripts/train_agent.py --algorithm PPO --timesteps 500000
```

### SAC (Soft Actor-Critic)

**Best for:**
- High-speed circuits (Monza)
- Continuous control refinement
- When maximum speed is critical

**Characteristics:**
- Off-policy (more sample efficient)
- Good for exploration
- Better at high-speed scenarios

**Recommended timesteps:** 800k - 1.5M

```bash
python scripts/train_agent.py --algorithm SAC --timesteps 800000
```

### TD3 (Twin Delayed DDPG)

**Best for:**
- Fine control in technical sections
- Precise cornering
- Advanced training

**Characteristics:**
- Excellent control precision
- Good for technical circuits
- Requires more training time

**Recommended timesteps:** 1M+

```bash
python scripts/train_agent.py --algorithm TD3 --timesteps 1000000
```

## Training Monitoring

During training, monitor these metrics:

**Episode Reward:**
- Measures overall performance per episode
- Higher values indicate better driving
- Typical progression: -100 (start) to 500-1000 (good) to 1500+ (excellent)

**Mean Reward:**
- Average of last 100 episodes
- More stable metric than episode reward
- Use this to judge overall progress

**Laps Completed:**
- Number of laps finished per episode
- Goal: Consistently complete 1-2 laps

**Track Limits Violations:**
- Off-track incidents per episode
- Should decrease as training progresses
- Frequent violations indicate poor generalization

**Target Performance Metrics:**
- Reward > 500: Agent showing progress
- Reward > 1000: Strong performance
- Reward > 1500: Excellent performance
- Consistent 1-2 lap completion: Success

## Evaluating Trained Models

After training, evaluate your model:

```bash
python scripts/watch_agent.py \
    --model trained_models/budapest_ppo/best_model.zip \
    --track tracks/budapest.json \
    --laps 5
```

**Viewer Controls:**
- `+/-` : Zoom adjustment
- `H` : Toggle HUD display
- `T` : Toggle trajectory trail
- `SPACE` : Pause/Resume
- `R` : Reset episode
- `ESC` : Exit viewer

## Training Time Estimates

**On CPU (recommended for MLP policies):**
- 500k timesteps: 1-2 hours
- 1M timesteps: 2-4 hours

**Note on CPU vs GPU:**
CPU training is recommended for this project. The neural networks used (MLP policies) are small, and the bottleneck is environment simulation (physics), not neural network computation. GPU provides minimal or no speedup and may actually be slower due to transfer overhead.

Stable-Baselines3 will warn if you attempt GPU training with MLP policies.

## Recommended Training Workflow

### For Beginners:

**Step 1: Start with Budapest**
```bash
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/budapest_ppo \
    --eval-freq 10000
```

**Step 2: Evaluate Performance**
```bash
python scripts/watch_agent.py \
    --model trained_models/budapest_ppo/best_model.zip \
    --track tracks/budapest.json \
    --laps 3
```

**Step 3: Adjust if Needed**
- If agent crashes frequently: Increase timesteps to 1M
- If training too slow: Try SAC algorithm
- If performance plateaus: Check hyperparameters

**Step 4: Expand to Other Circuits**
Once Budapest performs well, train on other circuits.

### For Advanced Users:

Train on varied circuit types for better generalization:

1. **Budapest** (technical, tight corners)
2. **Monza** (high-speed, long straights)
3. **Spa** (long, varied elevation)
4. **Catalunya** (balanced, mixed speeds)

This provides exposure to different racing challenges.

## Hyperparameter Tuning

Default hyperparameters in `train_agent.py`:

```python
learning_rate = 3e-4
batch_size = 64
n_steps = 2048  # PPO
gamma = 0.99  # Discount factor
```

**If agent is not learning:**
- Decrease learning_rate to 1e-4
- Increase timesteps to 1M+
- Try different algorithm (PPO → SAC)

**If training is unstable:**
- Decrease learning_rate
- Increase batch_size to 128
- Reduce n_steps to 1024

**For faster convergence:**
- Increase learning_rate to 5e-4 (carefully)
- Use curriculum learning (easy → hard circuits)

## Troubleshooting

### Problem: Agent Does Not Complete Laps

**Solutions:**
- Increase training timesteps (500k → 1M)
- Try different algorithm (PPO → SAC)
- Check reward function is providing good signals
- Verify track start position is valid

### Problem: Training Very Slow

**Solutions:**
- Reduce timesteps for initial testing (500k → 300k)
- Ensure using CPU (not GPU for MLP policies)
- Check no other heavy processes running
- Consider using simpler circuit (Budapest vs Spa)

### Problem: Negative or Low Rewards

**Normal if:**
- Early in training (first 100k steps)
- Agent still exploring environment

**Problem if:**
- Persists after 300k steps
- Mean reward not increasing

**Solutions:**
- Check hyperparameters
- Verify environment reward function
- Try different random seed
- Reduce learning rate

### Problem: Agent Gets Stuck in Corners

**Solutions:**
- Increase training time
- Add curriculum learning (start easier)
- Adjust reward shaping
- Try TD3 for better control precision

## Advanced Topics

### Curriculum Learning

Progressively increase difficulty:

```bash
# Start easy (wide track)
python scripts/train_agent.py --track tracks/monza.json --timesteps 300000

# Continue on medium
python scripts/train_agent.py --track tracks/catalunya.json --timesteps 300000 \
    --model trained_models/monza_ppo/best_model.zip

# Finish with hard (technical)
python scripts/train_agent.py --track tracks/budapest.json --timesteps 400000 \
    --model trained_models/catalunya_ppo/best_model.zip
```

### Transfer Learning

Use a trained model as starting point for new circuit:

```bash
# Train on Budapest first
python scripts/train_agent.py --track tracks/budapest.json --timesteps 500000

# Transfer to Spa
python scripts/train_agent.py \
    --track tracks/spa.json \
    --timesteps 300000 \
    --model trained_models/budapest_ppo/best_model.zip \
    --output trained_models/spa_ppo
```

### Hyperparameter Search

Systematic hyperparameter optimization:

```bash
# Test different learning rates
for lr in 1e-4 3e-4 1e-3; do
    python scripts/train_agent.py \
        --track tracks/budapest.json \
        --timesteps 300000 \
        --learning-rate $lr \
        --output trained_models/budapest_lr_${lr}
done
```

## Performance Optimization

**Circuit Optimization:**
- All circuits pre-optimized with 3.5m tolerance
- 90-96% point reduction maintained
- No manual optimization needed

**Training Optimization:**
- Use CPU (not GPU) for MLP policies
- Close unnecessary applications
- Use eval_freq=10000 for regular checkpoints
- Monitor with TensorBoard if available

**Rendering Optimization:**
- Fixed zoom (no dynamic adjustment)
- Optimized sprite caching
- Efficient boundary rendering
- Use `--no-render` flag during training if visualization not needed

## Notes

**Important Considerations:**

- All circuits have correct start positions (no initial track limit violations)
- Car proportions realistic (F1 car ~5m length)
- Zoom fixed at 5.0 (user adjustable with +/-)
- No automatic zoom changes during visualization
- CPU training recommended (see CPU vs GPU section)

**Default Training Parameters:**

Training script uses sensible defaults for all algorithms. Explicit parameter specification optional for advanced use cases only.

**Model Checkpoints:**

- `best_model.zip`: Best performing model during training
- `final_model.zip`: Model at end of training
- Evaluation logs in model output directory

Use best_model.zip for deployment and testing.
