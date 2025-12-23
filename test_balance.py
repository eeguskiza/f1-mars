#!/usr/bin/env python3
"""Quick test of balanced physics."""

import sys
import numpy as np
from pathlib import Path
import importlib.util

# Load modules directly
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = Path(__file__).parent / 'f1_mars'
car_module = load_module('car', base_path / 'envs' / 'car.py')
track_module = load_module('track', base_path / 'envs' / 'track.py')
tyres_module = load_module('tyres', base_path / 'envs' / 'tyres.py')

Car = car_module.Car
Track = track_module.Track
TrackGenerator = track_module.TrackGenerator
TyreSet = tyres_module.TyreSet
TyreCompound = tyres_module.TyreCompound

print("=" * 60)
print("  QUICK PHYSICS BALANCE TEST")
print("=" * 60)

# Create car
car = Car(x=0, y=0, heading=0)
print(f"\n✓ Car created")
print(f"  Max speed: {car.max_speed} m/s ({car.max_speed * 3.6:.1f} km/h)")
print(f"  Acceleration: {car.acceleration_power} m/s²")
print(f"  Brake: {car.brake_power} m/s²")

# Test acceleration for 5 seconds
print(f"\n  Testing acceleration (full throttle for 5s)...")
dt = 0.016
for i in range(int(5.0 / dt)):
    car.update(dt=dt, throttle=1.0, brake=0.0, steering_input=0.0)

print(f"  After 5s: {car.velocity:.1f} m/s ({car.velocity * 3.6:.1f} km/h)")

# Continue to max speed
print(f"\n  Testing max speed (full throttle for 10s more)...")
for i in range(int(10.0 / dt)):
    car.update(dt=dt, throttle=1.0, brake=0.0, steering_input=0.0)

print(f"  After 15s: {car.velocity:.1f} m/s ({car.velocity * 3.6:.1f} km/h)")
print(f"  Expected: ~{car.max_speed} m/s ({car.max_speed * 3.6:.1f} km/h)")

# Test tyres
print(f"\n✓ Tyres created")
tyres = TyreSet(TyreCompound.SOFT)
print(f"  Compound: {tyres.compound.name}")
print(f"  Grip base: {tyres.grip_base}")
print(f"  Wear rate: {tyres.wear_rate}")

# Simulate 30s of high-speed cornering
print(f"\n  Simulating 30s at max speed with hard cornering...")
initial_wear = tyres.wear
for i in range(int(30.0 / dt)):
    lateral = car.get_lateral_force()
    tyres.update(dt=dt, speed=car.velocity, lateral_force=0.8, throttle=1.0, braking=0.0)

wear_per_30s = tyres.wear - initial_wear
print(f"  Wear after 30s: {tyres.wear:.2f}%")
print(f"  Temperature: {tyres.temperature:.1f}°C")
print(f"  Grip: {tyres.get_grip():.3f}")

# Estimate laps
if wear_per_30s > 0:
    estimated_lap_time = 30.0  # Rough estimate
    wear_per_lap = wear_per_30s
    laps_to_70_percent = 70.0 / wear_per_lap
    print(f"\n  Estimated laps to 70% wear (cliff edge): {laps_to_70_percent:.1f}")
    print(f"  (Expected: 15-25 laps for balanced gameplay)")

# Test track
print(f"\n✓ Track generation")
track_data = TrackGenerator.generate_oval(length=1000, width=12.0)
track = Track.__new__(Track)
track.load_from_dict(track_data)
print(f"  Track length: {track.total_length:.1f}m")
print(f"  Track width: 12m (F1 standard)")

# Test LIDAR
print(f"\n✓ LIDAR test")
segments = track.get_boundary_segments(num_samples=50)
print(f"  Boundary segments: {len(segments)}")

print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)

# Expected vs Actual
checks = []
checks.append(("Max speed", car.max_speed * 3.6, 349, 351, "km/h"))
checks.append(("Tyre wear/30s", wear_per_30s, 2.0, 5.0, "%"))
checks.append(("Laps to cliff", laps_to_70_percent if wear_per_30s > 0 else 0, 15, 25, "laps"))

all_pass = True
for name, value, min_val, max_val, unit in checks:
    if min_val <= value <= max_val:
        print(f"✅ {name}: {value:.1f} {unit} (expected {min_val}-{max_val})")
    else:
        print(f"❌ {name}: {value:.1f} {unit} (expected {min_val}-{max_val})")
        all_pass = False

if all_pass:
    print("\n🎉 ALL BALANCE CHECKS PASSED!")
else:
    print("\n⚠️  Some values out of range - may need tuning")

print("=" * 60)
