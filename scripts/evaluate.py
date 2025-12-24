#!/usr/bin/env python3
"""
Evaluation script for F1-MARS trained models.

Evaluates RL agents on race tracks, collects metrics, generates reports,
and creates visualizations.
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Stable-Baselines3 imports
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

# F1-MARS imports
sys.path.insert(0, '.')
from f1_mars.envs import F1Env


class Evaluator:
    """
    Evaluator for trained F1-MARS agents.

    Runs evaluation episodes, collects metrics, and generates reports.
    """

    def __init__(
        self,
        model_path: str,
        track_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model (.zip file)
            track_path: Path to track JSON (None for default)
            config: Optional configuration dict
        """
        self.model_path = model_path
        self.track_path = track_path
        self.config = config or {}

        # Load model and create environment
        print(f"\nLoading model from: {model_path}")
        self.model = self._load_model(model_path)

        print(f"Creating environment...")
        max_laps = self.config.get('max_laps', 3)
        self.env = F1Env(track_path=track_path, max_laps=max_laps)

        # Extract model info
        self.model_name = Path(model_path).stem
        self.track_name = Path(track_path).stem if track_path else "default"

        # Results storage
        self.episode_results: List[Dict] = []

        print(f"✓ Model loaded: {self.model_name}")
        print(f"✓ Track: {self.track_name}")

    def _load_model(self, model_path: str) -> BaseAlgorithm:
        """
        Load model from file (auto-detect algorithm).

        Args:
            model_path: Path to model file

        Returns:
            Loaded model
        """
        # Try to detect algorithm from filename
        model_name = Path(model_path).stem.upper()

        try:
            if 'PPO' in model_name:
                return PPO.load(model_path)
            elif 'SAC' in model_name:
                return SAC.load(model_path)
            elif 'TD3' in model_name:
                return TD3.load(model_path)
            else:
                # Try PPO first (most common)
                try:
                    return PPO.load(model_path)
                except:
                    try:
                        return SAC.load(model_path)
                    except:
                        return TD3.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def run_evaluation(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation for multiple episodes.

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic actions
            render: Display visualization
            verbose: Print progress

        Returns:
            Dictionary with aggregated results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"  EVALUATION: {self.model_name} on {self.track_name}")
            print(f"{'='*70}")
            print(f"Episodes: {n_episodes}")
            print(f"Deterministic: {deterministic}")
            print(f"{'='*70}\n")

        self.episode_results = []

        for episode in range(n_episodes):
            if verbose:
                print(f"Episode {episode + 1}/{n_episodes}...", end=" ", flush=True)

            result = self._run_single_episode(
                deterministic=deterministic,
                render=render
            )

            self.episode_results.append(result)

            if verbose:
                print(f"✓ Laps: {result['laps_completed']}, "
                      f"Time: {result['total_time']:.2f}s, "
                      f"On-track: {result['on_track_percentage']:.1f}%")

        # Aggregate results
        aggregated = self._aggregate_results()

        if verbose:
            self._print_summary(aggregated)

        return aggregated

    def _run_single_episode(
        self,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single evaluation episode.

        Args:
            deterministic: Use deterministic actions
            render: Display visualization

        Returns:
            Episode results dictionary
        """
        obs, info = self.env.reset()

        # Episode tracking
        episode_data = {
            'actions': [],
            'rewards': [],
            'velocities': [],
            'positions': [],
            'on_track': [],
            'tyre_wear': [],
            'lap_times': [],
        }

        done = False
        step_count = 0
        total_reward = 0.0
        on_track_steps = 0
        off_track_count = 0
        last_on_track = True

        start_time = time.time()

        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Track metrics
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['velocities'].append(info.get('velocity', 0))
            episode_data['positions'].append(info.get('position', (0, 0)))

            on_track = info.get('on_track', True)
            episode_data['on_track'].append(on_track)
            episode_data['tyre_wear'].append(info.get('tyre_wear', 0))

            if on_track:
                on_track_steps += 1

            # Count off-track incidents
            if last_on_track and not on_track:
                off_track_count += 1
            last_on_track = on_track

            # Track lap times
            if 'lap_time' in info and info['lap_time'] > 0:
                episode_data['lap_times'].append(info['lap_time'])

            total_reward += reward
            step_count += 1

            if render:
                self.env.render()

        elapsed_time = time.time() - start_time

        # Compile results
        result = {
            'episode_length': step_count,
            'total_reward': total_reward,
            'total_time': elapsed_time,
            'laps_completed': info.get('lap', 0),
            'checkpoints_reached': info.get('checkpoint', 0),
            'on_track_percentage': (on_track_steps / step_count * 100) if step_count > 0 else 0,
            'off_track_count': off_track_count,
            'max_velocity': max(episode_data['velocities']) if episode_data['velocities'] else 0,
            'mean_velocity': np.mean(episode_data['velocities']) if episode_data['velocities'] else 0,
            'final_tyre_wear': episode_data['tyre_wear'][-1] if episode_data['tyre_wear'] else 0,
            'lap_times': episode_data['lap_times'],
            'episode_data': episode_data,
        }

        # Calculate per-lap tyre wear
        if result['laps_completed'] > 0 and result['final_tyre_wear'] > 0:
            result['tyre_wear_per_lap'] = result['final_tyre_wear'] / result['laps_completed']
        else:
            result['tyre_wear_per_lap'] = 0

        return result

    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results from all episodes.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.episode_results:
            return {}

        # Collect all lap times
        all_lap_times = []
        for result in self.episode_results:
            all_lap_times.extend(result['lap_times'])

        # Calculate completion rate
        completed_episodes = sum(
            1 for r in self.episode_results if r['laps_completed'] >= 1
        )
        completion_rate = completed_episodes / len(self.episode_results)

        # Aggregate metrics
        aggregated = {
            'model': self.model_name,
            'track': self.track_name,
            'episodes': len(self.episode_results),
            'metrics': {
                # Lap times
                'lap_time_mean': np.mean(all_lap_times) if all_lap_times else 0,
                'lap_time_std': np.std(all_lap_times) if all_lap_times else 0,
                'lap_time_best': np.min(all_lap_times) if all_lap_times else 0,
                'lap_time_worst': np.max(all_lap_times) if all_lap_times else 0,
                'total_laps': sum(r['laps_completed'] for r in self.episode_results),

                # Completion
                'completion_rate': completion_rate,
                'laps_completed_mean': np.mean([r['laps_completed'] for r in self.episode_results]),
                'laps_completed_std': np.std([r['laps_completed'] for r in self.episode_results]),

                # Performance
                'total_reward_mean': np.mean([r['total_reward'] for r in self.episode_results]),
                'total_reward_std': np.std([r['total_reward'] for r in self.episode_results]),
                'on_track_percentage': np.mean([r['on_track_percentage'] for r in self.episode_results]),

                # Velocity
                'max_velocity': np.max([r['max_velocity'] for r in self.episode_results]),
                'mean_velocity': np.mean([r['mean_velocity'] for r in self.episode_results]),

                # Tyre wear
                'tyre_wear_per_lap_mean': np.mean([r['tyre_wear_per_lap'] for r in self.episode_results]),
                'final_tyre_wear_mean': np.mean([r['final_tyre_wear'] for r in self.episode_results]),

                # Incidents
                'off_track_count_mean': np.mean([r['off_track_count'] for r in self.episode_results]),
                'off_track_count_total': sum(r['off_track_count'] for r in self.episode_results),
            },
            'per_episode': [
                {
                    'episode': i,
                    'laps_completed': r['laps_completed'],
                    'total_time': r['total_time'],
                    'total_reward': r['total_reward'],
                    'on_track_percentage': r['on_track_percentage'],
                    'lap_times': r['lap_times'],
                }
                for i, r in enumerate(self.episode_results)
            ]
        }

        return aggregated

    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        metrics = results['metrics']

        print(f"\n{'='*70}")
        print(f"  EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"\n📊 Overall Performance:")
        print(f"  Completion rate:      {metrics['completion_rate']:.1%}")
        print(f"  Total laps:           {metrics['total_laps']}")
        print(f"  Laps per episode:     {metrics['laps_completed_mean']:.2f} ± {metrics['laps_completed_std']:.2f}")

        if metrics['lap_time_best'] > 0:
            print(f"\n⏱️  Lap Times:")
            print(f"  Best:                 {metrics['lap_time_best']:.2f}s")
            print(f"  Mean:                 {metrics['lap_time_mean']:.2f}s ± {metrics['lap_time_std']:.2f}s")
            print(f"  Worst:                {metrics['lap_time_worst']:.2f}s")

        print(f"\n🏎️  Performance:")
        print(f"  On-track percentage:  {metrics['on_track_percentage']:.1f}%")
        print(f"  Off-track incidents:  {metrics['off_track_count_total']} ({metrics['off_track_count_mean']:.1f} per episode)")
        print(f"  Max velocity:         {metrics['max_velocity']:.1f} m/s")
        print(f"  Mean velocity:        {metrics['mean_velocity']:.1f} m/s")

        print(f"\n🔧 Tyre Management:")
        print(f"  Wear per lap:         {metrics['tyre_wear_per_lap_mean']:.1f}%")
        print(f"  Final wear:           {metrics['final_tyre_wear_mean']:.1f}%")

        print(f"\n🎯 Reward:")
        print(f"  Mean total reward:    {metrics['total_reward_mean']:.2f} ± {metrics['total_reward_std']:.2f}")

        print(f"{'='*70}\n")

    def generate_report(
        self,
        results: Dict[str, Any],
        output_dir: str = "results/"
    ):
        """
        Generate evaluation report with JSON and plots.

        Args:
            results: Aggregated evaluation results
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_path = output_path / f"{self.model_name}_{self.track_name}_evaluation.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Report saved: {json_path}")

        # Generate plots
        self._generate_plots(results, output_path)

    def _generate_plots(self, results: Dict[str, Any], output_path: Path):
        """Generate visualization plots."""
        # Collect data
        all_lap_times = []
        for episode in self.episode_results:
            all_lap_times.extend(episode['lap_times'])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Evaluation: {self.model_name} on {self.track_name}", fontsize=16)

        # 1. Lap times histogram
        if all_lap_times:
            axes[0, 0].hist(all_lap_times, bins=20, color='steelblue', edgecolor='black')
            axes[0, 0].axvline(np.mean(all_lap_times), color='red', linestyle='--',
                              label=f'Mean: {np.mean(all_lap_times):.2f}s')
            axes[0, 0].set_xlabel('Lap Time (s)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Lap Time Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Laps completed per episode
        episode_nums = [r['episode'] for r in results['per_episode']]
        laps_completed = [r['laps_completed'] for r in results['per_episode']]

        axes[0, 1].bar(episode_nums, laps_completed, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Laps Completed')
        axes[0, 1].set_title('Laps Completed per Episode')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. Tyre wear evolution (first episode)
        if self.episode_results:
            tyre_wear = self.episode_results[0]['episode_data']['tyre_wear']
            steps = range(len(tyre_wear))
            axes[1, 0].plot(steps, tyre_wear, color='orange', linewidth=2)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Tyre Wear (%)')
            axes[1, 0].set_title('Tyre Wear Evolution (Episode 1)')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Trajectory on track (first episode)
        if self.episode_results:
            positions = self.episode_results[0]['episode_data']['positions']
            on_track = self.episode_results[0]['episode_data']['on_track']

            if positions:
                xs, ys = zip(*positions)

                # Color by on-track status
                colors = ['green' if on else 'red' for on in on_track]
                axes[1, 1].scatter(xs, ys, c=colors, s=1, alpha=0.5)
                axes[1, 1].set_xlabel('X Position')
                axes[1, 1].set_ylabel('Y Position')
                axes[1, 1].set_title('Trajectory (Episode 1)')
                axes[1, 1].set_aspect('equal')
                axes[1, 1].grid(True, alpha=0.3)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', label='On track'),
                    Patch(facecolor='red', label='Off track')
                ]
                axes[1, 1].legend(handles=legend_elements)

        plt.tight_layout()

        # Save figure
        plot_path = output_path / f"{self.model_name}_{self.track_name}_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plots saved: {plot_path}")
        plt.close()

    def record_episode(
        self,
        output_path: str,
        fps: int = 30,
        deterministic: bool = True
    ):
        """
        Record a single episode as video.

        Args:
            output_path: Path to save video file
            fps: Frames per second
            deterministic: Use deterministic actions
        """
        try:
            import cv2
        except ImportError:
            print("⚠️  OpenCV not installed. Cannot record video.")
            print("   Install with: pip install opencv-python")
            return

        print(f"\nRecording episode to: {output_path}")

        # Run episode and collect frames
        obs, info = self.env.reset()
        frames = []
        done = False

        while not done:
            # Render frame
            frame = self.env.render()
            if frame is not None:
                frames.append(frame)

            # Get action and step
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

        # Save video
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in frames:
                out.write(frame)

            out.release()
            print(f"✓ Video saved: {output_path} ({len(frames)} frames)")
        else:
            print("⚠️  No frames captured")

    def compare_models(
        self,
        other_model_path: str,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Compare this model with another model.

        Args:
            other_model_path: Path to second model
            n_episodes: Number of episodes for each model
            deterministic: Use deterministic actions

        Returns:
            Comparison results dictionary
        """
        print(f"\n{'='*70}")
        print(f"  MODEL COMPARISON")
        print(f"{'='*70}")

        # Evaluate this model
        print(f"\n📊 Model 1: {self.model_name}")
        results_1 = self.run_evaluation(n_episodes, deterministic, verbose=False)
        self._print_summary(results_1)

        # Create evaluator for second model
        print(f"\n📊 Model 2: {Path(other_model_path).stem}")
        evaluator_2 = Evaluator(other_model_path, self.track_path, self.config)
        results_2 = evaluator_2.run_evaluation(n_episodes, deterministic, verbose=False)
        evaluator_2._print_summary(results_2)

        # Generate comparison
        comparison = {
            'model_1': results_1,
            'model_2': results_2,
            'comparison': self._compute_comparison(results_1, results_2)
        }

        # Print comparison summary
        self._print_comparison(comparison['comparison'])

        return comparison

    def _compute_comparison(
        self,
        results_1: Dict[str, Any],
        results_2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute comparison metrics between two models."""
        m1 = results_1['metrics']
        m2 = results_2['metrics']

        comparison = {}

        # Compare key metrics
        metrics_to_compare = [
            'completion_rate',
            'lap_time_mean',
            'lap_time_best',
            'on_track_percentage',
            'laps_completed_mean',
            'total_reward_mean',
            'off_track_count_mean'
        ]

        for metric in metrics_to_compare:
            v1 = m1.get(metric, 0)
            v2 = m2.get(metric, 0)

            # Calculate difference and percentage
            diff = v2 - v1
            pct_change = (diff / v1 * 100) if v1 != 0 else 0

            # Determine winner (lower is better for lap_time and off_track_count)
            if metric in ['lap_time_mean', 'lap_time_best', 'off_track_count_mean']:
                winner = 'model_1' if v1 < v2 else 'model_2'
            else:
                winner = 'model_1' if v1 > v2 else 'model_2'

            comparison[metric] = {
                'model_1': v1,
                'model_2': v2,
                'difference': diff,
                'percent_change': pct_change,
                'winner': winner
            }

        return comparison

    def _print_comparison(self, comparison: Dict[str, Any]):
        """Print comparison summary."""
        print(f"\n{'='*70}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*70}\n")

        for metric, data in comparison.items():
            v1 = data['model_1']
            v2 = data['model_2']
            diff = data['difference']
            pct = data['percent_change']
            winner = data['winner']

            # Format metric name
            metric_name = metric.replace('_', ' ').title()

            # Print comparison
            print(f"{metric_name}:")
            print(f"  Model 1: {v1:.2f}")
            print(f"  Model 2: {v2:.2f}")
            print(f"  Difference: {diff:+.2f} ({pct:+.1f}%)")
            print(f"  Winner: {winner}")
            print()

        print(f"{'='*70}\n")

    def close(self):
        """Clean up resources."""
        self.env.close()


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained F1-MARS models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)"
    )

    # Environment settings
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Path to track JSON (default: same as training)"
    )
    parser.add_argument(
        "--max-laps",
        type=int,
        default=3,
        help="Maximum laps per episode"
    )

    # Evaluation settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)"
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        help="Use stochastic actions"
    )

    # Visualization
    parser.add_argument(
        "--render",
        action="store_true",
        help="Display visualization during evaluation"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record episode as video"
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help="Path to save recorded video (default: auto-generate)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Directory to save evaluation results"
    )

    # Comparison
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to second model for comparison"
    )

    args = parser.parse_args()

    # Validate model path
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        return 1

    # Create configuration
    config = {
        'max_laps': args.max_laps
    }

    # Create evaluator
    evaluator = Evaluator(args.model, args.track, config)

    try:
        # Run comparison if specified
        if args.compare:
            if not Path(args.compare).exists():
                print(f"❌ Error: Comparison model file not found: {args.compare}")
                return 1

            comparison = evaluator.compare_models(
                args.compare,
                n_episodes=args.episodes,
                deterministic=args.deterministic
            )

            # Save comparison report
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)

            comparison_path = output_path / "comparison.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"✓ Comparison saved: {comparison_path}")

        # Run standard evaluation
        else:
            results = evaluator.run_evaluation(
                n_episodes=args.episodes,
                deterministic=args.deterministic,
                render=args.render
            )

            # Generate report
            evaluator.generate_report(results, args.output)

        # Record video if requested
        if args.record:
            record_path = args.record_path
            if record_path is None:
                model_name = Path(args.model).stem
                track_name = Path(args.track).stem if args.track else "default"
                record_path = f"{args.output}/{model_name}_{track_name}_recording.mp4"

            evaluator.record_episode(
                record_path,
                deterministic=args.deterministic
            )

    finally:
        evaluator.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
