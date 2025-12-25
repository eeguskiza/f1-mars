#!/bin/bash
#
# Convierte todos los CSVs de circuitos F1 a JSON
# Uso: bash scripts/convert_all_circuits.sh
#

set -e

echo "============================================"
echo "  CONVERSIÓN DE CIRCUITOS F1 (CSV → JSON)"
echo "============================================"
echo ""

# Directorio de CSVs
CSV_DIR="tracks/csv"
OUTPUT_DIR="tracks"

# Verificar que existe el directorio
if [ ! -d "$CSV_DIR" ]; then
    echo "❌ Error: No existe el directorio $CSV_DIR"
    exit 1
fi

# Contar CSVs
num_csvs=$(ls -1 "$CSV_DIR"/*.csv 2>/dev/null | wc -l)

if [ "$num_csvs" -eq 0 ]; then
    echo "❌ Error: No hay archivos CSV en $CSV_DIR"
    echo ""
    echo "Coloca los CSVs de circuitos F1 en $CSV_DIR/"
    echo "Formato esperado: x_m,y_m,w_tr_right_m,w_tr_left_m"
    exit 1
fi

echo "Encontrados $num_csvs circuitos en $CSV_DIR/"
echo ""

# Convertir cada CSV
count=0
for csv_file in "$CSV_DIR"/*.csv; do
    count=$((count + 1))
    circuit_name=$(basename "$csv_file" .csv)

    echo "[$count/$num_csvs] Convirtiendo $circuit_name..."

    # Tiempos de referencia de circuitos F1 reales (lap records aproximados)
    ref_time=""
    case "$circuit_name" in
        "Budapest"|"Hungaroring")
            ref_time="--ref-time 79.5"
            ;;
        "Spa"|"Spa-Francorchamps")
            ref_time="--ref-time 105.0"
            ;;
        "Monza")
            ref_time="--ref-time 81.0"
            ;;
        "Monaco")
            ref_time="--ref-time 72.0"
            ;;
        "Silverstone")
            ref_time="--ref-time 86.0"
            ;;
        "Suzuka")
            ref_time="--ref-time 90.0"
            ;;
        "Barcelona"|"Catalunya")
            ref_time="--ref-time 78.0"
            ;;
        "Bahrain"|"Sakhir")
            ref_time="--ref-time 91.0"
            ;;
        "Austin"|"COTA")
            ref_time="--ref-time 95.0"
            ;;
        "Interlagos"|"Brazil")
            ref_time="--ref-time 70.0"
            ;;
        *)
            # Auto-calcular si no conocemos el circuito
            ref_time=""
            ;;
    esac

    # Convertir (tolerancia 3.5m para buen balance rendimiento/precisión)
    python3 scripts/csv_to_track.py "$csv_file" \
        --tolerance 3.5 \
        --laps 5 \
        $ref_time

    echo ""
done

echo "============================================"
echo "✓ CONVERSIÓN COMPLETADA"
echo "============================================"
echo ""
echo "Circuitos generados en $OUTPUT_DIR/:"
ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | xargs -n1 basename
echo ""
echo "Para ver un circuito:"
echo "  python scripts/watch_agent.py --track tracks/<circuito>.json"
echo ""
