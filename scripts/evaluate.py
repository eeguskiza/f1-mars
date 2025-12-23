"""Evaluation script for trained agents."""

import argparse
import gymnasium as gym
import numpy as np
from pathlib import Path

from f1_mars.agents import PilotAgent
from f1_mars.utils.config import MODELS_DIR


def evaluate_agent(env, agent, num_episodes: int = 10, render: bool = True):
    """
    Evaluate a trained agent.

    Args:
        env: Gymnasium environment
        agent: Trained agent
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment

    Returns:
        Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    completed_laps = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)

        while not done:
            # Get action from agent
            action = agent.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

            # Print progress
            if episode_length % 100 == 0:
                print(f"  Step {episode_length}: Reward={episode_reward:.2f}, "
                      f"Lap={info.get('lap', 0)}, Speed={info.get('speed', 0):.1f}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        completed_laps.append(info.get('lap', 0))

        print(f"  Episode finished: Reward={episode_reward:.2f}, "
              f"Laps={info.get('lap', 0)}, Steps={episode_length}")

    # Calculate statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_laps": np.mean(completed_laps),
        "max_laps": np.max(completed_laps),
    }

    return stats


def main():
    """Evaluate trained pilot agent."""
    parser = argparse.ArgumentParser(description="Evaluate trained F1 Mars agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (relative to trained_models/)"
    )
    parser.add_argument(
        "--track",
        type=str,
        default="example_circuit",
        help="Track name to evaluate on"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering"
    )

    args = parser.parse_args()

    # Construct model path
    if not args.model.endswith('.zip'):
        args.model += '.zip'

    model_path = Path(MODELS_DIR) / args.model
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Create environment
    render_mode = None if args.no_render else "human"
    env = gym.make("F1Mars-v0", track_name=args.track, render_mode=render_mode)

    print(f"\n{'='*60}")
    print(f"Evaluating Pilot Agent")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Track: {args.track}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*60}\n")

    # Load agent
    agent = PilotAgent.create_from_checkpoint(str(model_path), env)

    # Evaluate
    stats = evaluate_agent(
        env,
        agent,
        num_episodes=args.episodes,
        render=not args.no_render
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean Episode Length: {stats['mean_length']:.1f}")
    print(f"Mean Laps Completed: {stats['mean_laps']:.2f}")
    print(f"Max Laps Completed: {stats['max_laps']}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
