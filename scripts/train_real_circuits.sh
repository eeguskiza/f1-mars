#!/bin/bash
#
# Entrena agentes en circuitos F1 reales
# Uso: bash scripts/train_real_circuits.sh [algoritmo] [timesteps]
#

set -e

# Parámetros
ALGORITHM=${1:-PPO}  # PPO, SAC, o TD3
TIMESTEPS=${2:-500000}
OUTPUT_DIR="trained_models/real_circuits_${ALGORITHM}_$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "  ENTRENAMIENTO EN CIRCUITOS F1 REALES"
echo "============================================"
echo ""
echo "Algoritmo:  $ALGORITHM"
echo "Timesteps:  $TIMESTEPS"
echo "Output:     $OUTPUT_DIR"
echo ""

# Circuitos disponibles
TRACKS=(
    "tracks/budapest.json"
    "tracks/catalunya.json"
    "tracks/monza.json"
    "tracks/spa.json"
    "tracks/austin.json"
    "tracks/nuerburgring.json"
    "tracks/yasmarina.json"
)

echo "Circuitos de entrenamiento:"
for track in "${TRACKS[@]}"; do
    name=$(basename "$track" .json)
    echo "  - ${name^}"
done
echo ""

# Crear directorio de salida
mkdir -p "$OUTPUT_DIR"

# Entrenar en cada circuito
echo "Iniciando entrenamiento..."
echo ""

for i in "${!TRACKS[@]}"; do
    track="${TRACKS[$i]}"
    track_name=$(basename "$track" .json)
    step=$((i + 1))

    echo "[$step/${#TRACKS[@]}] Entrenando en ${track_name^}..."

    # Entrenar
    python3 scripts/train_agent.py \
        --track "$track" \
        --algorithm "$ALGORITHM" \
        --timesteps "$TIMESTEPS" \
        --output "$OUTPUT_DIR/${track_name}_$ALGORITHM" \
        --eval-freq 10000 \
        --n-eval-episodes 3

    echo "  ✓ Completado"
    echo ""
done

echo "============================================"
echo "✓ ENTRENAMIENTO COMPLETADO"
echo "============================================"
echo ""
echo "Modelos guardados en: $OUTPUT_DIR/"
echo ""
echo "Para evaluar un modelo:"
echo "  python scripts/watch_agent.py \\"
echo "    --model $OUTPUT_DIR/<circuito>_$ALGORITHM/<model>.zip \\"
echo "    --track tracks/<circuito>.json"
echo ""
