#!/usr/bin/env python3
"""
Gentle tyre wear test - stays on track longer for accurate measurement.
"""

import numpy as np
from f1_mars.envs import F1Env

def test_gentle_laps():
    """
    Test with gentler steering to complete full laps and measure real wear.
    """
    env = F1Env(max_laps=3)
    obs, info = env.reset()

    print("=" * 70)
    print("  GENTLE LAPS TYRE WEAR TEST")
    print("=" * 70)
    print("\nUsing moderate steering to stay on track and complete full laps...")

    total_steps = 0
    max_steps = 60 * 60  # 60 seconds max
    lap_wear = {}
    initial_wear = 0.0

    while total_steps < max_steps:
        # Moderate policy - balance between speed and staying on track
        lidar = obs[4:15]
        left_avg = np.mean(lidar[:5])
        right_avg = np.mean(lidar[6:])

        # Gentler steering based on LIDAR
        if left_avg < right_avg:
            steering = 0.4  # Turn left (towards wider side)
        else:
            steering = -0.4  # Turn right

        # Moderate throttle
        throttle = 0.75

        action = np.array([steering, throttle, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        # Track wear per lap
        current_lap = info['lap']
        if current_lap not in lap_wear:
            if current_lap == 1:
                initial_wear = info['tyre_wear']
            lap_wear[current_lap] = {
                'start_wear': info['tyre_wear'] if current_lap == 1 else lap_wear[current_lap - 1]['end_wear'],
                'end_wear': info['tyre_wear'],
                'lap_time': 0.0
            }

        # Print status every 10 seconds
        if total_steps % 600 == 0:
            print(f"t={info['total_time']:5.1f}s | "
                  f"Lap {info['lap']} ({info['lap_progress']*100:5.1f}%) | "
                  f"Speed: {info['velocity_kmh']:6.1f} km/h | "
                  f"Wear: {info['tyre_wear']:6.2f}% | "
                  f"Temp: {info['tyre_temperature']:6.1f}°C")

        if terminated or truncated:
            reason = "off track" if not info['on_track'] else "completed laps" if info['lap'] > info['max_laps'] else "other"
            print(f"\nEpisode ended: {reason}")
            break

    # Calculate results
    final_wear = info['tyre_wear']
    total_wear = final_wear - initial_wear
    total_time = info['total_time']
    laps_completed = info['lap'] - 1 + info['lap_progress']

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"Total time:         {total_time:.1f} seconds")
    print(f"Laps completed:     {laps_completed:.2f}")
    print(f"Total distance:     {info['distance_along']:.1f}m")
    print(f"Total wear:         {total_wear:.2f}%")
    print(f"Avg speed:          {info['distance_along']/total_time * 3.6:.1f} km/h")

    if total_time > 0:
        wear_per_second = total_wear / total_time
        wear_per_30s = wear_per_second * 30
        print(f"\nWear rate:          {wear_per_second:.4f}% per second")
        print(f"Projected 30s wear: {wear_per_30s:.2f}%")

        if laps_completed > 0:
            wear_per_lap = total_wear / laps_completed
            lap_time = total_time / laps_completed
            laps_to_70_percent = 70.0 / wear_per_lap if wear_per_lap > 0 else 999
            print(f"\nWear per lap:       {wear_per_lap:.2f}%")
            print(f"Avg lap time:       {lap_time:.1f}s")
            print(f"Laps to 70% (cliff):{laps_to_70_percent:.1f} laps")

            # Validation
            print("\n" + "=" * 70)
            if 3.0 <= wear_per_30s <= 6.0:
                print("✅ PASS: Wear rate in expected range (3-6% per 30s)")
            else:
                print(f"⚠️  Wear rate: {wear_per_30s:.2f}% per 30s (expected 3-6%)")

            if 15 <= laps_to_70_percent <= 25:
                print(f"✅ PASS: Strategic window achieved ({laps_to_70_percent:.1f} laps to cliff)")
            else:
                print(f"⚠️  Laps to cliff: {laps_to_70_percent:.1f} (expected 15-25)")
    print("=" * 70)

    env.close()

if __name__ == "__main__":
    test_gentle_laps()
