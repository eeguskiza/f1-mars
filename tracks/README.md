# F1-MARS Tracks

This directory contains pre-designed racing circuits for curriculum learning.

## Available Tracks

### 1. Oval (Difficulty 0 - Tutorial)
- **Purpose**: Learn basic controls - acceleration, braking, gentle turns
- **Length**: ~907m
- **Width**: 14m (wider for beginners)
- **Characteristics**: Symmetrical, high-speed, beginner-friendly
- **Curves**: Wide radius turns (>48m), 45% straights
- **Reference lap time**: ~20s

**Best for**: Initial training, learning throttle/brake control

### 2. Simple Circuit (Difficulty 1 - Basic)
- **Purpose**: Introduce hard braking and varied corner speeds
- **Length**: ~1432m
- **Width**: 12m
- **Characteristics**: One tight hairpin, mixed speed sections
- **Curves**: 25% tight corners, 54% straights
- **Reference lap time**: ~28s

**Best for**: Learning corner entry/exit, trail braking

### 3. Technical Circuit (Difficulty 2 - Advanced)
- **Purpose**: Master precision with chicanes and linked corners
- **Length**: ~1311m
- **Width**: 11m (narrower)
- **Characteristics**: Chicane section, technical combinations
- **Curves**: 27% tight, 16% fast sweepers
- **Reference lap time**: ~35s

**Best for**: Improving racecraft, consistency

### 4. Grand Prix Circuit (Difficulty 3 - Expert)
- **Purpose**: Complete challenge with all corner types
- **Length**: ~1873m
- **Width**: 12m
- **Characteristics**: Complete variety - straights, hairpins, chicanes, fast sweepers
- **Curves**: Full spectrum from tight to fast
- **Reference lap time**: ~42s

**Best for**: Final evaluation, competitive performance

## Usage

### Loading a Track

```python
from tracks import load_track

# Load by name (without .json extension)
track = load_track("oval")
```

### Using in F1Env

```python
from f1_mars.envs import F1Env
from tracks import TRACKS_DIR

track_path = str(TRACKS_DIR / "oval.json")
env = F1Env(track_path=track_path, max_laps=3)
```

### List Available Tracks

```python
from tracks import list_available_tracks, get_track_info

# Get all track names
tracks = list_available_tracks()  # ['oval', 'simple', 'technical', 'mixed']

# Get metadata without loading full track
info = get_track_info("oval")
print(f"Difficulty: {info['difficulty']}")
print(f"Reference time: {info['reference_lap_time']}s")
```

### Filter by Difficulty

```python
from tracks import get_tracks_by_difficulty

# Get all beginner tracks
beginner_tracks = get_tracks_by_difficulty(0)  # ['oval']
```

## Curriculum Learning

Recommended training progression:

1. **Stage 1 - Basics** (oval):
   - Learn to complete laps without crashing
   - Target: Consistent lap times within 10% of reference

2. **Stage 2 - Intermediate** (simple):
   - Master corner entry and exit
   - Target: Complete 5 laps without major mistakes

3. **Stage 3 - Advanced** (technical):
   - Improve precision through chicanes
   - Target: Lap times within 5% of reference

4. **Stage 4 - Expert** (mixed):
   - Optimize racing line across all corner types
   - Target: Beat reference lap time

## File Format

Each track JSON contains:

```json
{
  "name": "Track Name",
  "author": "F1-MARS",
  "difficulty": 0-3,
  "centerline": [[x1, y1], [x2, y2], ...],
  "width": 12.0,
  "pit_entry_index": 5,
  "pit_exit_index": 15,
  "start_position": [x, y, heading],
  "checkpoints": [0, 25, 50, 75],
  "reference_lap_time": 30.0,
  "metadata": {
    "characteristics": ["high_speed", "technical"],
    "description": "Brief description"
  }
}
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_tracks.py -v
```

Test all tracks with analysis:

```bash
python scripts/test_all_tracks.py
```

## Creating Custom Tracks

See `track_definitions.py` for examples of track generation. To create a new track:

1. Define centerline points (closed loop)
2. Set width, checkpoints, metadata
3. Save as JSON in this directory
4. Track will be automatically discovered by `list_available_tracks()`

Example:

```python
import json
import numpy as np

def create_my_track():
    points = []
    # ... generate centerline points ...

    return {
        "name": "My Custom Track",
        "difficulty": 2,
        "centerline": points,
        "width": 12.0,
        # ... other properties
    }

# Save
with open("tracks/my_track.json", "w") as f:
    json.dump(create_my_track(), f, indent=2)
```
