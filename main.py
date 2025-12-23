#!/usr/bin/env python3
"""
F1-MARS: Formula 1 Multi-Agent Race Simulator

Entry point for running demos, training, and evaluation.

Usage:
    python main.py demo          # Run physics demo
    python main.py random        # Run random agent example
    python main.py train         # Train pilot agent (requires implementation)
    python main.py play          # Human playable mode (requires implementation)
    python main.py test          # Run test suite
"""

import sys
import subprocess


def print_usage():
    print(__doc__)
    print("\nAvailable commands:")
    print("  demo    - Run physics demonstration")
    print("  random  - Run random agent in environment")
    print("  test    - Run pytest test suite")
    print("  help    - Show this message")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "demo":
        print("Running physics demo...")
        exec(open('scripts/demo_physics.py').read())
    
    elif command == "random":
        print("Running random agent example...")
        exec(open('scripts/example_random_agent.py').read())
    
    elif command == "test":
        print("Running test suite...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    
    elif command in ("help", "-h", "--help"):
        print_usage()
    
    else:
        print(f"Unknown command: {command}")
        print("\nRun 'python main.py help' for usage information.")


if __name__ == "__main__":
    main()
