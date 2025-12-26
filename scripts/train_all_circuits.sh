#!/bin/bash
# Train an agent on all F1 circuits sequentially
# Creates a general-purpose agent that can drive on any track
#
# Usage:
#   bash scripts/train_all_circuits.sh [algorithm] [timesteps_per_circuit]
#
# Examples:
#   bash scripts/train_all_circuits.sh PPO 500000
#   bash scripts/train_all_circuits.sh SAC 800000

set -e  # Exit on error

# Default values
ALGORITHM=${1:-PPO}
TIMESTEPS=${2:-500000}
OUTPUT_DIR="trained_models/multi_circuit_${ALGORITHM,,}"

# Circuit order: easy -> medium -> hard
CIRCUITS=(
    "monza"         # Easy: Wide, high-speed
    "catalunya"     # Medium: Balanced
    "yasmarina"     # Medium: Technical sections
    "budapest"      # Medium: Tight, technical
    "austin"        # Medium-Hard: Elevation changes
    "nuerburgring"  # Medium-Hard: Varied corners
    "spa"           # Hard: Long, varied, challenging
)

echo "========================================"
echo "  F1-MARS Multi-Circuit Training"
echo "========================================"
echo "Algorithm:    $ALGORITHM"
echo "Timesteps:    $TIMESTEPS per circuit"
echo "Circuits:     ${#CIRCUITS[@]} circuits"
echo "Output:       $OUTPUT_DIR"
echo "========================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Train on each circuit
for i in "${!CIRCUITS[@]}"; do
    circuit="${CIRCUITS[$i]}"
    circuit_num=$((i + 1))

    echo ""
    echo "[$circuit_num/${#CIRCUITS[@]}] Training on $circuit..."
    echo "----------------------------------------"

    # Build command
    cmd="python scripts/train_agent.py \
        --track tracks/${circuit}.json \
        --algorithm $ALGORITHM \
        --timesteps $TIMESTEPS \
        --output $OUTPUT_DIR \
        --eval-freq 10000"

    # Add --model flag if not first circuit
    if [ $i -gt 0 ]; then
        cmd="$cmd --model $OUTPUT_DIR/best_model.zip"
    fi

    # Execute training
    echo "Command: $cmd"
    eval $cmd

    echo "✓ Completed $circuit"
done

echo ""
echo "========================================"
echo "  Training Complete!"
echo "========================================"
echo "Trained on ${#CIRCUITS[@]} circuits:"
for circuit in "${CIRCUITS[@]}"; do
    echo "  ✓ $circuit"
done
echo ""
echo "Model saved to: $OUTPUT_DIR/best_model.zip"
echo ""
echo "Test your agent with:"
echo "  python scripts/watch_agent.py"
echo ""
