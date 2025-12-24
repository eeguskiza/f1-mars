# Evaluation Guide for F1-MARS

Complete guide for evaluating trained models using `scripts/evaluate.py`.

## Quick Start

```bash
# Basic evaluation
python scripts/evaluate.py --model trained_models/PPO_default_final.zip

# Evaluation on specific track
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --track tracks/monza.json \
    --episodes 20

# Compare two models
python scripts/evaluate.py \
    --model trained_models/PPO_v1.zip \
    --compare trained_models/SAC_v1.zip \
    --episodes 10
```

## Command-Line Arguments

### Required

- `--model MODEL`: Path to trained model (.zip file)

### Environment Settings

- `--track TRACK`: Path to track JSON file (default: default oval)
- `--max-laps MAX_LAPS`: Maximum laps per episode (default: 3)

### Evaluation Settings

- `--episodes N`: Number of episodes to evaluate (default: 10)
- `--deterministic`: Use deterministic actions (default: True)
- `--no-deterministic`: Use stochastic actions

### Visualization

- `--render`: Display visualization during evaluation
- `--record`: Record episode as video (MP4)
- `--record-path PATH`: Custom path for recorded video

### Output

- `--output DIR`: Directory to save results (default: `results/`)

### Model Comparison

- `--compare MODEL`: Path to second model for head-to-head comparison

## Output Files

After evaluation, the following files are generated in the output directory:

### 1. JSON Report

**File**: `{model_name}_{track_name}_evaluation.json`

Contains complete evaluation data:

```json
{
  "model": "PPO_default_final",
  "track": "monza",
  "episodes": 10,
  "metrics": {
    "lap_time_mean": 44.5,
    "lap_time_std": 1.2,
    "lap_time_best": 42.8,
    "lap_time_worst": 47.3,
    "completion_rate": 0.9,
    "laps_completed_mean": 2.8,
    "on_track_percentage": 95.3,
    "max_velocity": 85.2,
    "mean_velocity": 62.4,
    "tyre_wear_per_lap_mean": 12.3,
    "total_reward_mean": 1523.4,
    "off_track_count_total": 12
  },
  "per_episode": [...]
}
```

### 2. Visualization Plots

**File**: `{model_name}_{track_name}_plots.png`

4-panel visualization:
1. **Lap Time Distribution**: Histogram of all lap times
2. **Laps Completed**: Bar chart per episode
3. **Tyre Wear Evolution**: Line plot for episode 1
4. **Track Trajectory**: 2D scatter plot (green=on track, red=off track)

### 3. Video Recording (Optional)

**File**: `{model_name}_{track_name}_recording.mp4`

MP4 video of a complete race episode.

## Metrics Explained

### Lap Times

- **lap_time_mean**: Average lap completion time
- **lap_time_std**: Standard deviation (consistency indicator)
- **lap_time_best**: Fastest lap achieved
- **lap_time_worst**: Slowest lap completed

### Completion

- **completion_rate**: % of episodes completing at least 1 lap
- **laps_completed_mean**: Average laps per episode
- **total_laps**: Sum of all laps across episodes

### Performance

- **on_track_percentage**: % of time car stayed on track
- **off_track_count**: Number of times car went off-track
- **max_velocity**: Maximum speed reached (m/s)
- **mean_velocity**: Average speed throughout episodes

### Tyre Management

- **tyre_wear_per_lap_mean**: Average wear per lap (%)
- **final_tyre_wear_mean**: Average wear at episode end

### Reward

- **total_reward_mean**: Average cumulative reward
- **total_reward_std**: Reward consistency

## Usage Examples

### 1. Basic Evaluation

Evaluate a model on the default track:

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --episodes 10
```

**Output**:
```
EVALUATION: PPO_default_final on default
======================================================================
Episodes: 10
Deterministic: True
======================================================================

Episode 1/10... ✓ Laps: 3, Time: 125.43s, On-track: 94.2%
Episode 2/10... ✓ Laps: 3, Time: 122.18s, On-track: 96.5%
...

======================================================================
  EVALUATION SUMMARY
======================================================================

📊 Overall Performance:
  Completion rate:      100.0%
  Total laps:           30
  Laps per episode:     3.00 ± 0.00

⏱️  Lap Times:
  Best:                 40.23s
  Mean:                 41.15s ± 0.82s
  Worst:                42.44s
...
```

### 2. Evaluation on Custom Track

```bash
python scripts/evaluate.py \
    --model trained_models/SAC_monza_final.zip \
    --track tracks/monza.json \
    --episodes 20 \
    --output results/monza_eval/
```

### 3. Render Visualization

Display the race in real-time:

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --episodes 5 \
    --render
```

### 4. Record Video

Generate MP4 recording:

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --record \
    --record-path recordings/best_lap.mp4
```

**Note**: Requires OpenCV (`pip install opencv-python`)

### 5. Compare Two Models

Head-to-head comparison:

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_v2.zip \
    --compare trained_models/SAC_v2.zip \
    --episodes 10 \
    --output results/comparison/
```

**Output**:
```
======================================================================
  MODEL COMPARISON
======================================================================

📊 Model 1: PPO_v2
[PPO evaluation results...]

📊 Model 2: SAC_v2
[SAC evaluation results...]

======================================================================
  COMPARISON SUMMARY
======================================================================

Lap Time Mean:
  Model 1: 41.23
  Model 2: 39.87
  Difference: -1.36 (-3.3%)
  Winner: model_2

Completion Rate:
  Model 1: 0.90
  Model 2: 0.95
  Difference: +0.05 (+5.6%)
  Winner: model_2
...
```

### 6. Stochastic Evaluation

Test model with exploration (non-deterministic):

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --no-deterministic \
    --episodes 20
```

## Programmatic Usage

Use the `Evaluator` class in your own scripts:

```python
from scripts.evaluate import Evaluator

# Create evaluator
evaluator = Evaluator(
    model_path="trained_models/PPO_default_final.zip",
    track_path="tracks/monza.json",
    config={'max_laps': 5}
)

# Run evaluation
results = evaluator.run_evaluation(
    n_episodes=10,
    deterministic=True,
    render=False
)

# Generate report
evaluator.generate_report(results, output_dir="results/")

# Record video
evaluator.record_episode("recordings/episode.mp4")

# Compare with another model
comparison = evaluator.compare_models(
    "trained_models/SAC_default_final.zip",
    n_episodes=10
)

# Cleanup
evaluator.close()
```

## Interpreting Results

### Good Performance Indicators

✅ **Completion rate > 90%**: Model consistently completes laps
✅ **Low lap_time_std**: Consistent performance
✅ **High on_track_percentage (>95%)**: Good control
✅ **Low off_track_count**: Stable racing line

### Areas for Improvement

⚠️ **Completion rate < 70%**: Model struggles with track
⚠️ **High lap_time_std**: Inconsistent performance
⚠️ **Low on_track_percentage (<85%)**: Control issues
⚠️ **High off_track_count**: Unstable behavior

### Diagnosing Issues

**Problem: Low completion rate**
- Check lap_time_best vs target
- Review trajectory plot for crashes
- Evaluate on easier track
- Consider more training

**Problem: High lap times**
- Check mean_velocity (too cautious?)
- Review tyre_wear strategy
- Compare with different algorithm

**Problem: Many off-track incidents**
- Review trajectory plot patterns
- Check if model overfits to training track
- Consider curriculum learning

## Advanced Features

### Batch Evaluation

Evaluate multiple models:

```bash
#!/bin/bash
for model in trained_models/*.zip; do
    echo "Evaluating $model..."
    python scripts/evaluate.py --model "$model" --episodes 10
done
```

### Track Comparison

Evaluate same model on different tracks:

```bash
#!/bin/bash
MODEL="trained_models/PPO_final.zip"

for track in tracks/*.json; do
    python scripts/evaluate.py \
        --model "$MODEL" \
        --track "$track" \
        --episodes 10 \
        --output "results/$(basename $track .json)/"
done
```

### Performance Benchmarking

Create a comprehensive benchmark:

```python
from scripts.evaluate import Evaluator
import json

models = ["PPO_v1.zip", "PPO_v2.zip", "SAC_v1.zip", "TD3_v1.zip"]
tracks = ["tracks/easy.json", "tracks/medium.json", "tracks/hard.json"]

results = {}

for model_name in models:
    results[model_name] = {}

    for track_path in tracks:
        evaluator = Evaluator(
            f"trained_models/{model_name}",
            track_path
        )

        track_name = Path(track_path).stem
        results[model_name][track_name] = evaluator.run_evaluation(
            n_episodes=20,
            verbose=False
        )

        evaluator.close()

# Save benchmark
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Troubleshooting

### Model Not Loading

```
❌ Error: Model file not found: trained_models/model.zip
```

**Solution**: Check path, ensure .zip extension

### Algorithm Detection Failed

```
ValueError: Failed to load model
```

**Solution**: Rename file to include algorithm (e.g., `PPO_model.zip`)

### Video Recording Fails

```
⚠️  OpenCV not installed. Cannot record video.
```

**Solution**: Install OpenCV
```bash
pip install opencv-python
```

### No Frames Captured

```
⚠️  No frames captured
```

**Solution**: Ensure environment supports rendering

### Out of Memory

**Solution**: Reduce episodes or disable video recording
```bash
python scripts/evaluate.py --model MODEL --episodes 5
```

## Best Practices

1. **Use deterministic mode** for fair comparisons
2. **Evaluate on multiple tracks** to test generalization
3. **Run sufficient episodes** (≥10) for statistical significance
4. **Save all results** for later analysis
5. **Compare with baselines** to measure improvement
6. **Record best performances** for demonstrations

## See Also

- [Training Guide](../scripts/TRAINING_GUIDE.md)
- [Curriculum Learning](CURRICULUM_LEARNING.md)
- [Track Creation Guide](../tracks/README.md)
