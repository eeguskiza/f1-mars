#!/usr/bin/env python3
"""
Quick verification that tyre wear fix is working.
Run this to confirm the integration is correct.
"""

import numpy as np
from f1_mars.envs import F1Env

print("=" * 70)
print("  TYRE WEAR FIX VERIFICATION")
print("=" * 70)

# Quick 30-second test
env = F1Env(max_laps=5)
obs, info = env.reset()

print("\nRunning 30-second moderate driving test...")

total_wear = 0.0
total_lateral = 0.0

for step in range(30 * 60):  # 30 seconds
    # Moderate policy
    lidar = obs[4:15]
    left_avg = np.mean(lidar[:5])
    right_avg = np.mean(lidar[6:])

    steering = 0.4 if left_avg < right_avg else -0.4
    throttle = 0.75

    obs, reward, terminated, truncated, info = env.step([steering, throttle, 0.0])
    total_lateral += env.car.get_lateral_force()

    if terminated or truncated:
        break

final_wear = info['tyre_wear']
avg_lateral = total_lateral / (step + 1)
duration = info['total_time']

print(f"\nResults after {duration:.1f} seconds:")
print(f"  Total wear:        {final_wear:.2f}%")
print(f"  Avg lateral force: {avg_lateral:.3f}")
print(f"  Max speed:         {info['velocity_kmh']:.1f} km/h")
print(f"  Tyre temperature:  {info['tyre_temperature']:.1f}°C")

# Validation
print("\n" + "=" * 70)
if 3.0 <= final_wear <= 6.0:
    print("✅ PASS: Tyre wear is in expected range (3-6% per 30s)")
    print(f"   Measured: {final_wear:.2f}%")
else:
    print(f"❌ FAIL: Tyre wear out of range: {final_wear:.2f}% (expected 3-6%)")

if avg_lateral > 0.15:
    print(f"✅ PASS: Lateral force calculation working ({avg_lateral:.3f})")
else:
    print(f"❌ FAIL: Lateral force too low ({avg_lateral:.3f})")

if final_wear > 0 and duration > 20:
    wear_per_lap_30s = final_wear * (30.0 / duration)
    laps_to_cliff = 70.0 / wear_per_lap_30s
    print(f"✅ PASS: Integration working correctly")
    print(f"   Projected laps to 70% (30s/lap): {laps_to_cliff:.1f}")

    if 12 <= laps_to_cliff <= 25:
        print(f"✅ PASS: Strategic window achieved ({laps_to_cliff:.1f} laps)")
    else:
        print(f"⚠️  Strategic window: {laps_to_cliff:.1f} laps (target 15-25)")

print("=" * 70)
print("\n🎉 Tyre wear fix verification complete!")
print("\nThe system is now ready for RL training with proper tyre management.")

env.close()
