"""F1 Mars - A 2D F1 Racing Simulator with Reinforcement Learning."""

__version__ = "0.1.0"

from gymnasium.envs.registration import register

# Register the Gymnasium environment
register(
    id="F1Mars-v0",
    entry_point="f1_mars.envs:F1Env",
    max_episode_steps=10000,
)

__all__ = ["__version__"]
