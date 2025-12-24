#!/usr/bin/env python3
"""
Training script for the F1-MARS pilot agent.

Trains an RL agent to race autonomously using PPO, SAC, or TD3.
Supports multi-track curriculum learning and parallel environments.
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List, Callable
import numpy as np

# Stable-Baselines3 imports
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor

# F1-MARS imports
sys.path.insert(0, '.')
from f1_mars.envs import F1Env, CurriculumWrapper
from tracks import list_available_tracks, get_tracks_by_difficulty, TRACKS_DIR


class VerboseEvalCallback(BaseCallback):
    """
    Callback that prints status messages during evaluation.
    Makes it clear when eval is running (so it doesn't look frozen).
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.is_evaluating = False
        self.eval_episode = 0

    def _on_step(self) -> bool:
        # Check if we're in evaluation mode
        if hasattr(self, 'locals') and 'dones' in self.locals:
            dones = self.locals.get('dones', [])
            if any(dones):
                self.eval_episode += 1
                if self.verbose > 0:
                    print(f"  Eval episode {self.eval_episode} completed", flush=True)
        return True


class F1MetricsCallback(BaseCallback):
    """
    Custom callback for logging F1-specific metrics to TensorBoard.

    Tracks:
    - Lap times (mean, best)
    - Tyre wear
    - On-track percentage
    - Laps completed
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_lap_times = []
        self.episode_tyre_wear = []
        self.episode_on_track_time = []
        self.episode_total_time = []
        self.episode_laps_completed = []

        # Best metrics
        self.best_lap_time = float('inf')
        self.total_laps_completed = 0

    def _on_step(self) -> bool:
        # Get info from all environments
        infos = self.locals.get('infos', [])

        for info in infos:
            if 'episode' in info:
                # Episode finished
                episode_info = info.get('episode', {})

                # Extract F1-specific metrics from info dict
                if 'lap_time' in info:
                    lap_time = info['lap_time']
                    self.episode_lap_times.append(lap_time)

                    if lap_time < self.best_lap_time:
                        self.best_lap_time = lap_time

                if 'tyre_wear' in info:
                    self.episode_tyre_wear.append(info['tyre_wear'])

                if 'laps_completed' in info:
                    laps = info['laps_completed']
                    self.episode_laps_completed.append(laps)
                    self.total_laps_completed += laps

                # Calculate on-track percentage
                if 'on_track_time' in info and 'total_time' in info:
                    on_track_pct = (info['on_track_time'] / info['total_time']) * 100
                    self.episode_on_track_time.append(on_track_pct)

        # Log metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        """Log accumulated metrics to TensorBoard."""
        if len(self.episode_lap_times) > 0:
            self.logger.record("f1/lap_time_mean", np.mean(self.episode_lap_times))
            self.logger.record("f1/lap_time_best", self.best_lap_time)
            self.episode_lap_times.clear()

        if len(self.episode_tyre_wear) > 0:
            self.logger.record("f1/tyre_wear_mean", np.mean(self.episode_tyre_wear))
            self.episode_tyre_wear.clear()

        if len(self.episode_on_track_time) > 0:
            self.logger.record("f1/on_track_percentage", np.mean(self.episode_on_track_time))
            self.episode_on_track_time.clear()

        if len(self.episode_laps_completed) > 0:
            self.logger.record("f1/laps_completed_mean", np.mean(self.episode_laps_completed))
            self.logger.record("f1/total_laps_completed", self.total_laps_completed)
            self.episode_laps_completed.clear()


def make_env(
    track_path: Optional[str] = None,
    rank: int = 0,
    seed: int = 0,
    max_laps: int = 3,
    use_curriculum: bool = False,
    curriculum_level: int = 0
) -> Callable:
    """
    Create a function that returns a monitored F1Env.

    Args:
        track_path: Path to track JSON file (None for default)
        rank: Index of the environment
        seed: Random seed
        max_laps: Maximum laps per episode
        use_curriculum: Whether to wrap with curriculum learning
        curriculum_level: Initial curriculum level (0-3)

    Returns:
        Function that creates the environment
    """
    def _init():
        env = F1Env(track_path=track_path, max_laps=max_laps)

        # Wrap with curriculum if enabled
        if use_curriculum:
            env = CurriculumWrapper(
                env,
                initial_level=curriculum_level,
                enable_logging=(rank == 0)  # Only log in first env
            )

        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vec_env(
    n_envs: int,
    track_paths: Optional[List[str]] = None,
    seed: int = 0,
    max_laps: int = 3,
    use_curriculum: bool = False,
    curriculum_level: int = 0,
    force_dummy: bool = False
) -> SubprocVecEnv:
    """
    Create vectorized environments for parallel training.

    Args:
        n_envs: Number of parallel environments
        track_paths: List of track paths (cycles through if multi-track)
        seed: Random seed
        max_laps: Maximum laps per episode
        use_curriculum: Whether to use curriculum learning
        curriculum_level: Initial curriculum level (0-3)
        force_dummy: Force DummyVecEnv (for eval envs to match type)

    Returns:
        Vectorized environment
    """
    if track_paths is None or len(track_paths) == 0:
        track_paths = [None]  # Use default track

    # Create environment factory for each process
    env_fns = []
    for i in range(n_envs):
        # Cycle through tracks if multi-track training
        track_path = track_paths[i % len(track_paths)]
        env_fns.append(make_env(
            track_path,
            rank=i,
            seed=seed,
            max_laps=max_laps,
            use_curriculum=use_curriculum,
            curriculum_level=curriculum_level
        ))

    # Use SubprocVecEnv for true parallelization, DummyVecEnv for single env or when forced
    if n_envs > 1 and not force_dummy:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    return vec_env


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train F1-MARS pilot agent with Stable-Baselines3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Algorithm settings
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="PPO",
        choices=["PPO", "SAC", "TD3"],
        help="RL algorithm to use"
    )

    # Training settings
    parser.add_argument(
        "--total-timesteps", "-t",
        type=int,
        default=500000,
        help="Total timesteps for training"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", "-bs",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--n-envs", "-e",
        type=int,
        default=8,
        help="Number of parallel environments"
    )

    # Track settings
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Path to track JSON file (default: use default oval)"
    )
    parser.add_argument(
        "--multi-track",
        action="store_true",
        help="Train on multiple tracks (rotate each environment)"
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Use tracks of specific difficulty (0-3)"
    )

    # Curriculum learning settings
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning (progressive difficulty)"
    )
    parser.add_argument(
        "--curriculum-level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Initial curriculum level (0=Basic, 3=Expert)"
    )

    # Callbacks and logging
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50000,
        help="Save checkpoint every N timesteps"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50000,
        help="Evaluate every N timesteps (default: 50000, less frequent = faster training)"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Number of episodes for evaluation (default: 3, fewer = faster)"
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs/",
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="trained_models/",
        help="Directory to save trained models"
    )

    # Model management
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to existing model to continue training"
    )

    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (CPU recommended for vectorial obs + parallel envs)"
    )
    parser.add_argument(
        "--max-laps",
        type=int,
        default=3,
        help="Maximum laps per episode"
    )

    return parser.parse_args()


def get_track_paths(args) -> List[str]:
    """
    Get list of track paths based on arguments.

    Returns:
        List of track paths (or [None] for default)
    """
    if args.track:
        # Single specific track
        return [args.track]
    elif args.multi_track:
        # All tracks in tracks/ directory
        track_names = list_available_tracks()
        return [str(TRACKS_DIR / f"{name}.json") for name in track_names]
    elif args.difficulty is not None:
        # Tracks of specific difficulty
        track_names = get_tracks_by_difficulty(args.difficulty)
        if not track_names:
            print(f"⚠️  No tracks found with difficulty {args.difficulty}")
            return [None]
        return [str(TRACKS_DIR / f"{name}.json") for name in track_names]
    else:
        # Default track
        return [None]


def create_model(algorithm: str, env, args):
    """
    Create RL model based on algorithm choice.

    Args:
        algorithm: Algorithm name (PPO, SAC, TD3)
        env: Vectorized environment
        args: Command-line arguments

    Returns:
        Initialized model
    """
    common_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": args.learning_rate,
        "verbose": 1,
        "tensorboard_log": args.tensorboard_log,
        "device": args.device,
        "seed": args.seed
    }

    if algorithm == "PPO":
        model = PPO(
            **common_kwargs,
            n_steps=2048,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
    elif algorithm == "SAC":
        model = SAC(
            **common_kwargs,
            buffer_size=100000,
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
        )
    elif algorithm == "TD3":
        model = TD3(
            **common_kwargs,
            buffer_size=100000,
            batch_size=args.batch_size,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model


def main():
    """Main training loop."""
    args = parse_args()

    # Create directories
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_log).mkdir(parents=True, exist_ok=True)

    # Print training configuration
    print("\n" + "=" * 70)
    print("  F1-MARS PILOT AGENT TRAINING")
    print("=" * 70)
    print(f"Algorithm:        {args.algorithm}")
    print(f"Total timesteps:  {args.total_timesteps:,}")
    print(f"Learning rate:    {args.learning_rate}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Parallel envs:    {args.n_envs}")
    print(f"Device:           {args.device}")
    print(f"Seed:             {args.seed}")
    print(f"Max laps:         {args.max_laps}")

    # Get track configuration
    track_paths = get_track_paths(args)
    if args.multi_track:
        print(f"Multi-track:      Enabled ({len(track_paths)} tracks)")
        for path in track_paths:
            track_name = Path(path).stem if path else "default"
            print(f"  - {track_name}")
    elif args.track:
        track_name = Path(args.track).stem
        print(f"Track:            {track_name}")
    elif args.difficulty is not None:
        print(f"Difficulty:       {args.difficulty} ({len(track_paths)} tracks)")
    else:
        print(f"Track:            Default oval")

    # Curriculum learning info
    if args.curriculum:
        level_names = ["Basic", "Intermediate", "Advanced", "Expert"]
        print(f"\nCurriculum:       ENABLED (starting at level {args.curriculum_level}: {level_names[args.curriculum_level]})")
        print(f"                  Will automatically progress through difficulty levels")
    else:
        print(f"\nCurriculum:       Disabled")

    print(f"\nCheckpoint freq:  Every {args.checkpoint_freq:,} steps")
    print(f"Eval freq:        Every {args.eval_freq:,} steps")
    print(f"Model directory:  {args.model_dir}")
    print(f"TensorBoard log:  {args.tensorboard_log}")
    print("=" * 70 + "\n")

    # Create training environments
    print("Creating training environments...")
    train_env = create_vec_env(
        n_envs=args.n_envs,
        track_paths=track_paths,
        seed=args.seed,
        max_laps=args.max_laps,
        use_curriculum=args.curriculum,
        curriculum_level=args.curriculum_level
    )
    print(f"✓ Created {args.n_envs} parallel environments")
    if args.curriculum:
        print(f"  Curriculum learning enabled (level {args.curriculum_level})")

    # Create evaluation environment (single env, first track)
    print("Creating evaluation environment...")
    eval_track = track_paths[0] if track_paths else None
    eval_env = create_vec_env(
        n_envs=1,
        track_paths=[eval_track],
        seed=args.seed + 1000,
        max_laps=args.max_laps,
        use_curriculum=False,  # Don't use curriculum for evaluation
        curriculum_level=0,
        force_dummy=True  # Always use DummyVecEnv for eval (avoids type mismatch warning)
    )
    print(f"✓ Created evaluation environment")

    # Create or load model
    if args.load_model:
        print(f"\nLoading existing model from {args.load_model}...")
        if args.algorithm == "PPO":
            model = PPO.load(args.load_model, env=train_env)
        elif args.algorithm == "SAC":
            model = SAC.load(args.load_model, env=train_env)
        elif args.algorithm == "TD3":
            model = TD3.load(args.load_model, env=train_env)
        print("✓ Model loaded successfully")
    else:
        print(f"\nCreating new {args.algorithm} model...")
        model = create_model(args.algorithm, train_env, args)
        print("✓ Model created successfully")

    # Create callbacks
    print("\nSetting up callbacks...")

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,  # Adjust for parallel envs
        save_path=args.model_dir,
        name_prefix=f"{args.algorithm}_checkpoint",
        verbose=1
    )

    # Evaluation callback with custom wrapper for progress
    class VerboseEvalWrapper(EvalCallback):
        """Wrapper that prints when evaluation starts/ends."""

        def _on_step(self) -> bool:
            # Check if it's time to evaluate
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                print(f"\n{'='*50}")
                print(f"  EVALUATION at step {self.num_timesteps}")
                print(f"  Running {self.n_eval_episodes} episodes...")
                print(f"{'='*50}\n")

            result = super()._on_step()

            # Print results after evaluation
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                if len(self.evaluations_results) > 0:
                    last_mean = self.last_mean_reward
                    print(f"\n{'='*50}")
                    print(f"  EVALUATION COMPLETE")
                    print(f"  Mean reward: {last_mean:.2f}")
                    print(f"  Resuming training...")
                    print(f"{'='*50}\n")

            return result

    eval_callback = VerboseEvalWrapper(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.tensorboard_log,
        eval_freq=args.eval_freq // args.n_envs,  # Adjust for parallel envs
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # F1 metrics callback
    f1_callback = F1MetricsCallback(verbose=1)

    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, f1_callback])
    print("✓ Callbacks configured")

    # Train
    print("\n" + "=" * 70)
    print("  STARTING TRAINING")
    print("=" * 70 + "\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    training_time = time.time() - start_time

    # Save final model
    track_name = "multi" if args.multi_track else (
        Path(args.track).stem if args.track else "default"
    )
    final_model_path = f"{args.model_dir}/{args.algorithm}_{track_name}_final"
    model.save(final_model_path)

    # Print summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTotal training time:  {training_time/60:.1f} minutes")
    print(f"Final model saved:    {final_model_path}.zip")

    if hasattr(f1_callback, 'best_lap_time') and f1_callback.best_lap_time < float('inf'):
        print(f"Best lap time:        {f1_callback.best_lap_time:.2f}s")

    if hasattr(f1_callback, 'total_laps_completed'):
        print(f"Total laps completed: {f1_callback.total_laps_completed}")

    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {args.tensorboard_log}")

    print("\nTo evaluate the model:")
    print(f"  python scripts/evaluate.py --model {final_model_path}.zip")

    print("=" * 70 + "\n")

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
