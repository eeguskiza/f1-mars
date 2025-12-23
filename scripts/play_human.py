"""Human-playable mode for F1 Mars simulator."""

import argparse
import gymnasium as gym
import pygame
import numpy as np

from f1_mars.utils.config import FPS


class HumanController:
    """
    Keyboard controller for human players.
    """

    def __init__(self):
        """Initialize the controller."""
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.pit_requested = False

        # Control parameters
        self.steering_speed = 3.0  # How fast steering responds
        self.throttle_speed = 2.0
        self.brake_speed = 3.0

    def get_action(self, dt: float) -> np.ndarray:
        """
        Get action based on current keyboard input.

        Args:
            dt: Time delta in seconds

        Returns:
            Action array [steering, throttle, brake]
        """
        keys = pygame.key.get_pressed()

        # Steering (left/right arrows)
        target_steering = 0.0
        if keys[pygame.K_LEFT]:
            target_steering = -1.0
        elif keys[pygame.K_RIGHT]:
            target_steering = 1.0

        # Smooth steering interpolation
        steering_delta = (target_steering - self.steering) * self.steering_speed * dt
        self.steering += steering_delta
        self.steering = np.clip(self.steering, -1.0, 1.0)

        # Throttle (up arrow)
        if keys[pygame.K_UP]:
            self.throttle = min(1.0, self.throttle + self.throttle_speed * dt)
            self.brake = 0.0
        else:
            self.throttle = max(0.0, self.throttle - self.throttle_speed * dt * 2)

        # Brake (down arrow)
        if keys[pygame.K_DOWN]:
            self.brake = min(1.0, self.brake + self.brake_speed * dt)
            self.throttle = 0.0
        else:
            self.brake = max(0.0, self.brake - self.brake_speed * dt * 2)

        # Pit stop (space bar)
        if keys[pygame.K_SPACE]:
            if not self.pit_requested:
                self.pit_requested = True
                print("Pit stop requested!")
        else:
            self.pit_requested = False

        return np.array([self.steering, self.throttle, self.brake], dtype=np.float32)


def main():
    """Run human-playable mode."""
    parser = argparse.ArgumentParser(description="Play F1 Mars manually")
    parser.add_argument(
        "--track",
        type=str,
        default="example_circuit",
        help="Track name to play on"
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=3,
        help="Number of laps to complete"
    )

    args = parser.parse_args()

    # Create environment
    env = gym.make("F1Mars-v0", track_name=args.track, render_mode="human")

    # Create controller
    controller = HumanController()

    print(f"\n{'='*60}")
    print(f"F1 Mars - Human Play Mode")
    print(f"{'='*60}")
    print(f"Track: {args.track}")
    print(f"Laps to complete: {args.laps}")
    print(f"\nControls:")
    print(f"  Arrow Keys: Steer, Accelerate, Brake")
    print(f"  Space: Pit Stop")
    print(f"  ESC: Quit")
    print(f"{'='*60}\n")

    obs, info = env.reset()
    clock = pygame.time.Clock()
    running = True
    total_reward = 0
    step_count = 0

    dt = 1.0 / FPS

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset
                    print("\nResetting...")
                    obs, info = env.reset()
                    total_reward = 0
                    step_count = 0

        # Get action from keyboard
        action = controller.get_action(dt)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Render
        env.render()

        # Print info periodically
        if step_count % 60 == 0:  # Every second at 60 FPS
            print(f"Lap {info.get('lap', 0)}, "
                  f"Speed: {info.get('speed', 0):.1f}, "
                  f"Fuel: {info.get('fuel', 0):.1f}%, "
                  f"Tire Wear: {info.get('tire_wear', 0):.1%}")

        # Check if episode ended
        if terminated or truncated:
            print(f"\n{'='*60}")
            print(f"Episode Finished!")
            print(f"{'='*60}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Laps Completed: {info.get('lap', 0)}")
            print(f"Steps: {step_count}")
            print(f"\nPress 'R' to restart or ESC to quit")
            print(f"{'='*60}\n")

            # Wait for user input
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        elif event.key == pygame.K_r:
                            obs, info = env.reset()
                            total_reward = 0
                            step_count = 0
                            waiting = False

                clock.tick(FPS)

        # Maintain frame rate
        clock.tick(FPS)

    env.close()
    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
