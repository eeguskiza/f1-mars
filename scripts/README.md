# F1-MARS Training & Evaluation Scripts

Scripts para entrenar y evaluar agentes de F1-MARS.

## 📁 Scripts Disponibles

| Script | Descripción | Agente | Algoritmos | Documentación |
|--------|-------------|--------|------------|---------------|
| **train_pilot.py** | Entrena el piloto autónomo (control del coche) | Pilot | PPO, SAC, TD3 | [→ Guía Completa](TRAIN_PILOT.md) |
| **train_engineer.py** | Entrena el ingeniero de carrera (estrategia) | Engineer | DQN | [→ Guía Completa](TRAIN_ENGINEER.md) |
| **evaluate.py** | Evalúa modelos entrenados con métricas detalladas | Ambos | Todos | [→ Guía de Evaluación](../docs/EVALUATION_GUIDE.md) |

---

## 🚀 Quick Start

### Entrenar Piloto

```bash
# PPO (recomendado para principiantes)
python scripts/train_pilot.py --algorithm PPO --total-timesteps 500000

# Con curriculum learning automático
python scripts/train_pilot.py --curriculum --total-timesteps 1000000
```

**→ [Guía completa de entrenamiento del piloto](TRAIN_PILOT.md)**

### Entrenar Ingeniero

```bash
# Estrategia con DQN
python scripts/train_engineer.py --track monza --timesteps 500000
```

**→ [Guía completa de entrenamiento del ingeniero](TRAIN_ENGINEER.md)**

### Evaluar Modelo

```bash
# Evaluación básica
python scripts/evaluate.py --model trained_models/PPO_default_final.zip

# Con visualización y grabación
python scripts/evaluate.py \
    --model trained_models/PPO_final.zip \
    --episodes 20 \
    --record \
    --output results/
```

**→ [Guía completa de evaluación](../docs/EVALUATION_GUIDE.md)**

---

## 🎯 Casos de Uso

### 1. Quiero entrenar un piloto desde cero

```bash
# Opción A: Training tradicional (PPO)
python scripts/train_pilot.py \
    --algorithm PPO \
    --total-timesteps 500000

# Opción B: Con curriculum learning (recomendado)
python scripts/train_pilot.py \
    --curriculum \
    --total-timesteps 1000000
```

**Ver:** [TRAIN_PILOT.md](TRAIN_PILOT.md) - Sección "Algoritmos Disponibles"

### 2. Quiero comparar diferentes algoritmos

```bash
# Entrenar con cada algoritmo
python scripts/train_pilot.py --algorithm PPO --total-timesteps 500000
python scripts/train_pilot.py --algorithm SAC --total-timesteps 500000
python scripts/train_pilot.py --algorithm TD3 --total-timesteps 500000

# Comparar dos modelos
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --compare trained_models/SAC_default_final.zip
```

**Ver:** [TRAIN_PILOT.md](TRAIN_PILOT.md) - Sección "Comparación de Algoritmos"

### 3. Quiero entrenar en un circuito específico

```bash
python scripts/train_pilot.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --total-timesteps 500000
```

**Ver:** [TRAIN_PILOT.md](TRAIN_PILOT.md) - Sección "Training Options"

### 4. Quiero entrenar estrategia de pit stops

```bash
python scripts/train_engineer.py \
    --track monza \
    --timesteps 500000 \
    --tensorboard
```

**Ver:** [TRAIN_ENGINEER.md](TRAIN_ENGINEER.md)

### 5. Quiero evaluar y comparar todos mis modelos

```bash
# Evaluar todos los modelos en trained_models/
for model in trained_models/*.zip; do
    python scripts/evaluate.py --model "$model" --episodes 10
done

# Comparación directa de dos modelos
python scripts/evaluate.py \
    --model trained_models/PPO_v1.zip \
    --compare trained_models/SAC_v1.zip \
    --episodes 20
```

**Ver:** Sección "Evaluación Batch" más abajo

---

## 📊 Comparación de Algoritmos

| Algoritmo | Tipo | Velocidad | Estabilidad | Exploración | Mejor Para |
|-----------|------|-----------|-------------|-------------|------------|
| **PPO** | On-policy | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Principiantes, entrenamiento estable |
| **SAC** | Off-policy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Circuitos complejos, convergencia rápida |
| **TD3** | Off-policy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Control preciso, time trials |
| **DQN** | Off-policy | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Decisiones discretas (estrategia) |

**→ Detalles completos:** [TRAIN_PILOT.md - Comparación de Algoritmos](TRAIN_PILOT.md#comparación-de-algoritmos)

---

## 🔄 Workflow Completo

### Pipeline End-to-End

```bash
# 1. Entrenar piloto con curriculum
python scripts/train_pilot.py \
    --curriculum \
    --algorithm PPO \
    --total-timesteps 1000000 \
    --model-dir models/pilot/

# 2. Entrenar ingeniero de estrategia
python scripts/train_engineer.py \
    --track monza \
    --timesteps 500000 \
    --model-dir models/engineer/

# 3. Evaluar piloto
python scripts/evaluate.py \
    --model models/pilot/PPO_multi_final.zip \
    --episodes 20 \
    --output results/pilot/

# 4. Evaluar ingeniero
python scripts/evaluate.py \
    --model models/engineer/engineer_final_monza.zip \
    --episodes 20 \
    --output results/engineer/

# 5. Ver resultados en TensorBoard
tensorboard --logdir logs/
```

---

## 📈 Evaluación Batch de Múltiples Modelos

### Script para Evaluar Todos los Modelos

Crear archivo `evaluate_all.sh`:

```bash
#!/bin/bash

# Configuración
EPISODES=20
OUTPUT_BASE="results/batch_eval"
TRACKS=("tracks/oval.json" "tracks/monza.json" "tracks/technical.json")

# Crear directorio de salida
mkdir -p "$OUTPUT_BASE"

# Evaluar cada modelo
for model in trained_models/*.zip; do
    model_name=$(basename "$model" .zip)
    echo "Evaluando: $model_name"

    # Evaluar en cada circuito
    for track in "${TRACKS[@]}"; do
        track_name=$(basename "$track" .json)
        output_dir="$OUTPUT_BASE/${model_name}/${track_name}"

        python scripts/evaluate.py \
            --model "$model" \
            --track "$track" \
            --episodes "$EPISODES" \
            --output "$output_dir"
    done
done

echo "Evaluación completa! Resultados en: $OUTPUT_BASE"
```

**Ejecutar:**
```bash
chmod +x evaluate_all.sh
./evaluate_all.sh
```

### Comparar Todos los Algoritmos

```bash
#!/bin/bash

# Comparar PPO vs SAC vs TD3
TRACK="tracks/monza.json"
EPISODES=20

# Entrenar cada algoritmo
for algo in PPO SAC TD3; do
    python scripts/train_pilot.py \
        --algorithm "$algo" \
        --track "$TRACK" \
        --total-timesteps 500000 \
        --model-dir "models/${algo}/"
done

# Evaluar cada uno
for algo in PPO SAC TD3; do
    python scripts/evaluate.py \
        --model "models/${algo}/${algo}_monza_final.zip" \
        --track "$TRACK" \
        --episodes "$EPISODES" \
        --output "results/${algo}/"
done

# Comparaciones directas
python scripts/evaluate.py \
    --model "models/PPO/PPO_monza_final.zip" \
    --compare "models/SAC/SAC_monza_final.zip" \
    --episodes "$EPISODES" \
    --output "results/comparison_PPO_vs_SAC/"

python scripts/evaluate.py \
    --model "models/SAC/SAC_monza_final.zip" \
    --compare "models/TD3/TD3_monza_final.zip" \
    --episodes "$EPISODES" \
    --output "results/comparison_SAC_vs_TD3/"
```

### Generar Reporte de Comparación

Python script `compare_all.py`:

```python
#!/usr/bin/env python3
"""
Compara todos los modelos y genera un reporte CSV.
"""

import json
import csv
from pathlib import Path
from glob import glob

results_dir = Path("results/batch_eval")
output_file = "comparison_report.csv"

# Recopilar métricas de todos los modelos
all_results = []

for json_file in glob(str(results_dir / "*/*/*.json")):
    with open(json_file, 'r') as f:
        data = json.load(f)

        metrics = data['metrics']
        all_results.append({
            'model': data['model'],
            'track': data['track'],
            'completion_rate': metrics['completion_rate'],
            'lap_time_mean': metrics['lap_time_mean'],
            'lap_time_best': metrics['lap_time_best'],
            'on_track_percentage': metrics['on_track_percentage'],
            'off_track_count': metrics['off_track_count_total'],
        })

# Guardar a CSV
with open(output_file, 'w', newline='') as f:
    if all_results:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

print(f"Reporte guardado: {output_file}")
print(f"Total de evaluaciones: {len(all_results)}")
```

**Ejecutar:**
```bash
python compare_all.py
```

---

## 📚 Documentación Completa

### Guías de Entrenamiento

- **[TRAIN_PILOT.md](TRAIN_PILOT.md)** - Entrenamiento completo del piloto
  - Comparación detallada PPO vs SAC vs TD3
  - Hyperparámetros y configuración
  - Curriculum learning
  - Transfer learning
  - Workflows completos

- **[TRAIN_ENGINEER.md](TRAIN_ENGINEER.md)** - Entrenamiento del ingeniero
  - Estrategia de pit stops
  - Gestión de neumáticos
  - DQN para decisiones discretas

### Guías de Evaluación

- **[../docs/EVALUATION_GUIDE.md](../docs/EVALUATION_GUIDE.md)** - Evaluación detallada
  - Métricas completas
  - Visualizaciones
  - Comparación de modelos
  - Grabación de videos

### Documentación Adicional

- **[../docs/CURRICULUM_LEARNING.md](../docs/CURRICULUM_LEARNING.md)** - Curriculum learning
- **[../tracks/README.md](../tracks/README.md)** - Creación de circuitos
- **[../README.md](../README.md)** - Documentación principal del proyecto

---

## 💡 Tips

1. **Empieza con PPO** - Es el más estable y fácil de usar
2. **Usa curriculum learning** - Para mejor generalización (`--curriculum`)
3. **Monitorea con TensorBoard** - `tensorboard --logdir logs/`
4. **Guarda checkpoints frecuentes** - Ya configurado por defecto
5. **Evalúa regularmente** - Usa `evaluate.py` para tracking
6. **Compara algoritmos** - Prueba PPO, SAC y TD3 para encontrar el mejor
7. **CPU es suficiente** - Óptimo para RL con entornos paralelos

---

## ⚡ Comandos Rápidos

```bash
# Entrenar piloto (PPO, curriculum)
python scripts/train_pilot.py --curriculum --total-timesteps 1000000

# Entrenar piloto (SAC, circuito específico)
python scripts/train_pilot.py --algorithm SAC --track tracks/monza.json --total-timesteps 500000

# Entrenar ingeniero
python scripts/train_engineer.py --track monza --timesteps 500000

# Evaluar modelo
python scripts/evaluate.py --model trained_models/PPO_final.zip --episodes 20

# Comparar modelos
python scripts/evaluate.py --model MODEL1.zip --compare MODEL2.zip --episodes 10

# Ver TensorBoard
tensorboard --logdir logs/
```

---

**Para más detalles, consulta las guías específicas de cada script.**
