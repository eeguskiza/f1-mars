# F1 Circuit CSV Data

This directory contains authentic F1 circuit racing line data in CSV format.

## CSV Format

```csv
# x_m,y_m,w_tr_right_m,w_tr_left_m
x,y,track_width_right,track_width_left
...
```

**Column Definitions:**
- `x_m`, `y_m`: Racing line centerline coordinates in meters
- `w_tr_right_m`: Track width to the right of centerline (meters)
- `w_tr_left_m`: Track width to the left of centerline (meters)

## Converting CSV to JSON

Use the provided conversion script:

**Basic Conversion:**
```bash
python scripts/csv_to_track.py tracks/csv/Budapest.csv
```

**Specify Output Path:**
```bash
python scripts/csv_to_track.py tracks/csv/Budapest.csv -o tracks/hungaroring.json
```

**Configure Laps and Reference Time:**
```bash
python scripts/csv_to_track.py tracks/csv/Budapest.csv --laps 10 --ref-time 79.5
```

**Adjust Simplification Tolerance:**
```bash
# Higher tolerance = fewer points, better performance
python scripts/csv_to_track.py tracks/csv/Budapest.csv --tolerance 4.0

# Lower tolerance = more points, higher precision
python scripts/csv_to_track.py tracks/csv/Budapest.csv --tolerance 2.0

# Disable simplification (keep all points)
python scripts/csv_to_track.py tracks/csv/Budapest.csv --no-simplify
```

## Batch Conversion

Convert all circuits at once:

```bash
for csv in tracks/csv/*.csv; do
    python scripts/csv_to_track.py "$csv" --tolerance 3.5 --laps 5
done
```

## Viewing Generated Circuits

After conversion, visualize circuits with a trained model:

```bash
python scripts/watch_agent.py \
    --model trained_models/best_model.zip \
    --track tracks/budapest.json \
    --laps 5
```

## Available Circuits

| Circuit | Filename | Real Length | Lap Record | Points (Original) |
|---------|----------|-------------|------------|-------------------|
| Austin (COTA) | Austin.csv | 5.49 km | ~95s | 1102 |
| Budapest (Hungaroring) | Budapest.csv | 4.36 km | 79.5s | 876 |
| Catalunya (Barcelona) | Catalunya.csv | 4.63 km | 78s | 931 |
| Monza | Monza.csv | 5.78 km | 81s | 1159 |
| Nürburgring (GP) | Nuerburgring.csv | 5.13 km | 86s | 1029 |
| Spa-Francorchamps | Spa.csv | 6.98 km | 105s | 1401 |
| Yas Marina (Abu Dhabi) | YasMarina.csv | 5.53 km | 91s | 1110 |

After optimization with default 3.5m tolerance, point counts reduce to 46-79 (90-96% reduction).

## Adding New Circuits

To add a new F1 circuit:

1. Place CSV file in this directory
2. Ensure CSV follows format: `x_m,y_m,w_tr_right_m,w_tr_left_m`
3. Run conversion script:
   ```bash
   python scripts/csv_to_track.py tracks/csv/NewCircuit.csv
   ```
4. JSON file will be generated in `tracks/` directory

## Circuit Optimization

All circuits are automatically optimized using the Douglas-Peucker algorithm during conversion:

- **Default tolerance**: 3.5 meters
- **Point reduction**: 90-96%
- **Accuracy**: Racing line shape preserved
- **Performance**: 10-15x faster rendering

The tolerance parameter controls simplification aggressiveness:
- 2.0m: High precision (~70% reduction)
- 3.5m: Balanced (90-95% reduction, recommended)
- 5.0m: Maximum performance (~97% reduction)

## Data Sources

Racing line data sourced from open F1 track datasets and racing simulations.

## Notes

- All coordinates relative to arbitrary origin
- Start position automatically set to first centerline point
- Pit lane entry/exit automatically calculated (5-15% track length)
- Track width varies along racing line (realistic modeling)
