#!/usr/bin/env python3
"""
Example: Random Agent in F1 Mars Environment

Demonstrates how to use the F1Env with a simple random policy.
This serves as a template for implementing RL agents.
"""

import numpy as np
from f1_mars.envs import F1Env

def random_policy(obs):
    """
    Simple random policy for demonstration.

    In practice, this would be replaced by a trained RL agent
    (PPO, SAC, etc.) from stable-baselines3 or similar.

    Args:
        obs: Observation vector (26 dimensions)

    Returns:
        action: [steering, throttle, brake]
    """
    # Extract LIDAR observations (indices 4-14)
    lidar = obs[4:15]

    # Simple rule-based steering: avoid obstacles
    left_distance = np.mean(lidar[:5])   # Left 5 rays
    right_distance = np.mean(lidar[6:])   # Right 5 rays
    center_distance = lidar[5]            # Center ray

    # Steering: turn away from closer obstacle
    if left_distance < right_distance:
        steering = 0.3  # Turn right
    else:
        steering = -0.3  # Turn left

    # Add some randomness
    steering += np.random.uniform(-0.1, 0.1)
    steering = np.clip(steering, -1.0, 1.0)

    # Throttle: slow down if obstacle ahead
    if center_distance < 0.3:  # Normalized distance
        throttle = 0.3
        brake = 0.2
    else:
        throttle = 0.8
        brake = 0.0

    return np.array([steering, throttle, brake], dtype=np.float32)


def main():
    """Run a simple episode with random policy."""
    print("=" * 70)
    print("  F1 MARS - RANDOM AGENT EXAMPLE")
    print("=" * 70)

    # Create environment
    print("\nInitializing environment...")
    env = F1Env(max_laps=2)
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Run episode
    print("\nRunning episode with random policy...")
    obs, info = env.reset()

    episode_reward = 0.0
    steps = 0
    max_steps = 1800  # 30 seconds at 60 FPS

    print(f"\nStarting position: [{info['position_x']:.1f}, {info['position_y']:.1f}]")
    print(f"Starting lap: {info['lap']}/{info['max_laps']}\n")

    while steps < max_steps:
        # Get action from policy
        action = random_policy(obs)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps += 1

        # Print status every 5 seconds
        if steps % 300 == 0:
            print(f"t={info['total_time']:5.1f}s | "
                  f"Lap {info['lap']} | "
                  f"Progress: {info['lap_progress']*100:5.1f}% | "
                  f"Speed: {info['velocity_kmh']:6.1f} km/h | "
                  f"Tyres: {info['tyre_wear']:5.2f}% | "
                  f"Reward: {episode_reward:8.1f}")

        # Check if episode ended
        if terminated or truncated:
            print(f"\nEpisode ended at step {steps}")
            if terminated:
                if not info['on_track']:
                    print("  Reason: Off track")
                elif info['tyre_wear'] >= 90:
                    print("  Reason: Dead tyres")
                elif info['lap'] > info['max_laps']:
                    print("  Reason: Completed laps!")
            else:
                print("  Reason: Truncated (max steps)")
            break

    # Print summary
    print("\n" + "=" * 70)
    print("  EPISODE SUMMARY")
    print("=" * 70)
    print(f"Total steps:        {steps}")
    print(f"Total time:         {info['total_time']:.2f}s")
    print(f"Total reward:       {episode_reward:.2f}")
    print(f"Final lap:          {info['lap']}/{info['max_laps']}")
    print(f"Lap progress:       {info['lap_progress']*100:.1f}%")
    print(f"Distance traveled:  {info['distance_along']:.1f}m / {info['track_length']:.1f}m")
    print(f"Final speed:        {info['velocity_kmh']:.1f} km/h")
    print(f"Tyre wear:          {info['tyre_wear']:.2f}%")
    print(f"Tyre compound:      {info['tyre_compound']}")
    print(f"Final grip:         {info['tyre_grip']:.3f}")
    print(f"On track:           {'Yes' if info['on_track'] else 'No'}")
    print("=" * 70)

    # Clean up
    env.close()

    print("\n✓ Example completed successfully!")
    print("\nNext steps:")
    print("  1. Replace random_policy() with a trained RL agent (PPO, SAC, etc.)")
    print("  2. Use stable-baselines3 for training")
    print("  3. Add rendering for visualization")
    print("  4. Implement engineer agent for pit strategy")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
