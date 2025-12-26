#!/usr/bin/env python3
"""
Simplified training interface for F1-MARS agents.

This is a wrapper around train_pilot.py that provides a cleaner interface
matching the documentation in README.md and TRAINING.md.

Example usage:
    python scripts/train_agent.py \\
        --track tracks/budapest.json \\
        --algorithm PPO \\
        --timesteps 500000 \\
        --output trained_models/budapest_ppo \\
        --eval-freq 10000
"""

import sys
import subprocess
from pathlib import Path


def main():
    """
    Wrapper script that translates simplified arguments to train_pilot.py format.

    Argument mapping:
        --timesteps     → --total-timesteps
        --output        → --model-dir
        --eval-freq     → --eval-freq (same)
        --algorithm     → --algorithm (same)
        --track         → --track (same)

    All other arguments are passed through unchanged.
    """
    # Get script directory
    script_dir = Path(__file__).parent
    train_pilot_path = script_dir / "train_pilot.py"

    if not train_pilot_path.exists():
        print(f"Error: train_pilot.py not found at {train_pilot_path}")
        sys.exit(1)

    # Build command with argument mapping
    cmd = [sys.executable, str(train_pilot_path)]

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        # Map simplified arguments to train_pilot.py arguments
        if arg == "--timesteps":
            cmd.append("--total-timesteps")
            if i + 1 < len(sys.argv):
                cmd.append(sys.argv[i + 1])
                i += 2
            else:
                print("Error: --timesteps requires a value")
                sys.exit(1)

        elif arg == "--output":
            cmd.append("--model-dir")
            if i + 1 < len(sys.argv):
                cmd.append(sys.argv[i + 1])
                i += 2
            else:
                print("Error: --output requires a value")
                sys.exit(1)

        elif arg == "--model":
            # Map --model to --load-model for continuing training
            cmd.append("--load-model")
            if i + 1 < len(sys.argv):
                cmd.append(sys.argv[i + 1])
                i += 2
            else:
                print("Error: --model requires a value")
                sys.exit(1)

        else:
            # Pass through all other arguments unchanged
            cmd.append(arg)
            i += 1

    # Execute train_pilot.py with transformed arguments
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running train_pilot.py: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
