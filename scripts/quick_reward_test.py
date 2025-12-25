#!/usr/bin/env python3
"""Quick test to verify the reward function fix."""

from f1_mars.envs import F1Env
import numpy as np

print("\n" + "="*70)
print("  QUICK REWARD FUNCTION TEST")
print("="*70)

# Test 1: Random actions
print("\n[Test 1] Random actions (100 steps)...")
env = F1Env(max_laps=1)
obs, info = env.reset()

total_reward = 0
for i in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    if term or trunc:
        break

env.close()
print(f"   Total reward: {total_reward:.2f}")
print(f"   Average reward per step: {total_reward/100:.3f}")

# Test 2: Constant acceleration
print("\n[Test 2] Constant acceleration straight (300 steps / 5 seconds)...")
env = F1Env(max_laps=1)
obs, info = env.reset()

accelerate_action = np.array([0.0, 1.0, 0.0])  # steering=0, throttle=1, brake=0
total_reward = 0
max_velocity = 0
step_rewards = []
velocities = []

for i in range(300):
    obs, reward, term, trunc, info = env.step(accelerate_action)
    total_reward += reward
    step_rewards.append(reward)
    max_velocity = max(max_velocity, info.get('velocity', 0))
    velocities.append(info.get('velocity', 0))
    if term or trunc:
        break

env.close()

print(f"   Total reward: {total_reward:.2f}")
print(f"   Average reward per step: {total_reward/len(step_rewards):.3f}")
print(f"   Max velocity: {max_velocity:.1f} m/s ({max_velocity*3.6:.1f} km/h)")

# Analyze reward progression
first_half_avg = sum(step_rewards[:len(step_rewards)//2]) / (len(step_rewards)//2)
second_half_avg = sum(step_rewards[len(step_rewards)//2:]) / (len(step_rewards) - len(step_rewards)//2)

print(f"   Reward first half: {first_half_avg:.3f}")
print(f"   Reward second half: {second_half_avg:.3f}")
print(f"   Velocity first half: {sum(velocities[:len(velocities)//2]) / (len(velocities)//2):.1f} m/s")
print(f"   Velocity second half: {sum(velocities[len(velocities)//2:]) / (len(velocities) - len(velocities)//2):.1f} m/s")

# Test 3: Standing still (very slow speed)
print("\n[Test 3] Standing still / very slow (100 steps)...")
env = F1Env(max_laps=1)
obs, info = env.reset()

still_action = np.array([0.0, 0.0, 0.0])  # no throttle, no brake
total_reward = 0
step_rewards = []

for i in range(100):
    obs, reward, term, trunc, info = env.step(still_action)
    total_reward += reward
    step_rewards.append(reward)
    if term or trunc:
        break

env.close()

print(f"   Total reward: {total_reward:.2f}")
print(f"   Average reward per step: {total_reward/len(step_rewards):.3f}")

# Summary
print("\n" + "="*70)
print("  ANALYSIS")
print("="*70)

if total_reward < -100:
    print("   ✅ Standing still is heavily penalized (as expected)")
else:
    print("   ❌ Standing still should be more penalized")

# Reload Test 2 results
env = F1Env(max_laps=1)
obs, info = env.reset()
accelerate_action = np.array([0.0, 1.0, 0.0])
accel_total = 0
for i in range(300):
    obs, reward, term, trunc, info = env.step(accelerate_action)
    accel_total += reward
    if term or trunc:
        break
env.close()

if accel_total > 0:
    print("   ✅ Acceleration produces POSITIVE total reward")
    print("   ✅ Fix is working!")
elif accel_total > -500:
    print("   ⚠️  Acceleration produces slightly negative reward")
    print("   ⚠️  This may improve with longer episodes")
else:
    print("   ❌ Acceleration produces very negative reward")
    print("   ❌ Fix may not be working correctly")

print("\n" + "="*70)
print("  EXPECTED BEHAVIOR:")
print("  - Random actions: ~0 or slightly negative")
print("  - Acceleration: POSITIVE (agent learns to accelerate)")
print("  - Standing still: VERY NEGATIVE (agent avoids this)")
print("="*70 + "\n")
