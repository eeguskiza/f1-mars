"""Training script for the pilot agent."""

import argparse
import gymnasium as gym
from pathlib import Path

from f1_mars.agents import PilotAgent, PilotTrainingCallback
from f1_mars.utils.config import MODELS_DIR, LOGS_DIR


def main():
    """Train the pilot agent."""
    parser = argparse.ArgumentParser(description="Train F1 Mars pilot agent")
    parser.add_argument(
        "--track",
        type=str,
        default="example_circuit",
        help="Track name to train on"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during training (slower)"
    )

    args = parser.parse_args()

    # Create directories
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make("F1Mars-v0", track_name=args.track, render_mode=render_mode)

    print(f"\n{'='*60}")
    print(f"Training Pilot Agent")
    print(f"{'='*60}")
    print(f"Track: {args.track}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")

    # Create agent
    tensorboard_log = LOGS_DIR if args.tensorboard else None
    agent = PilotAgent(
        env,
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        tensorboard_log=tensorboard_log,
    )

    # Create callback
    callback = PilotTrainingCallback(
        save_freq=args.save_freq,
        save_path=MODELS_DIR,
        verbose=1
    )

    # Train
    try:
        agent.train(
            total_timesteps=args.timesteps,
            callback=callback,
            log_interval=10
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_path = f"{MODELS_DIR}/pilot_final_{args.track}.zip"
    agent.save(final_path)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
