"""Training script for the race engineer agent."""

import argparse
import gymnasium as gym
from pathlib import Path

from f1_mars.agents import EngineerAgent, EngineerTrainingCallback
from f1_mars.utils.config import MODELS_DIR, LOGS_DIR


def main():
    """Train the engineer agent."""
    parser = argparse.ArgumentParser(description="Train F1 Mars race engineer agent")
    parser.add_argument(
        "--track",
        type=str,
        default="example_circuit",
        help="Track name to train on"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=25000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging"
    )

    args = parser.parse_args()

    # Create directories
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # Create environment (strategy-focused variant would go here)
    # For now, using the same environment
    env = gym.make("F1Mars-v0", track_name=args.track)

    print(f"\n{'='*60}")
    print(f"Training Race Engineer Agent")
    print(f"{'='*60}")
    print(f"Track: {args.track}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")

    # Create agent
    tensorboard_log = LOGS_DIR if args.tensorboard else None
    agent = EngineerAgent(
        env,
        learning_rate=args.learning_rate,
        tensorboard_log=tensorboard_log,
    )

    # Create callback
    callback = EngineerTrainingCallback(
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
    final_path = f"{MODELS_DIR}/engineer_final_{args.track}.zip"
    agent.save(final_path)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
