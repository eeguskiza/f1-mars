# Entrenamiento del Piloto Autónomo

Guía completa para entrenar el agente piloto que controla el coche (steering, throttle, brake).

---

## 📋 Índice

- [Algoritmos Disponibles](#algoritmos-disponibles)
- [Cómo Entrenar](#cómo-entrenar)
- [Cómo Evaluar](#cómo-evaluar)
- [Comparación de Todos los Modelos](#comparación-de-todos-los-modelos)

---

## Algoritmos Disponibles

El piloto soporta **3 algoritmos de RL** para control continuo.

### PPO (Proximal Policy Optimization)

**Recomendado para:** Principiantes, entrenamiento estable, primera vez

#### Características

| Aspecto | Valoración | Detalles |
|---------|-----------|----------|
| **Estabilidad** | ⭐⭐⭐⭐⭐ | Muy estable, converge de forma confiable |
| **Velocidad** | ⭐⭐⭐ | Moderada, requiere más pasos que SAC |
| **Facilidad de uso** | ⭐⭐⭐⭐⭐ | Fácil de configurar, pocos hiperparámetros |
| **Sample efficiency** | ⭐⭐⭐ | Requiere bastantes muestras (on-policy) |
| **Exploración** | ⭐⭐⭐ | Exploración moderada |

#### Pros
- ✅ **Muy estable**: Rara vez diverge o falla
- ✅ **Pocos hiperparámetros**: Fácil de tunear
- ✅ **Predecible**: Comportamiento consistente
- ✅ **Bajo uso de memoria**: No necesita replay buffer grande
- ✅ **Funciona bien out-of-the-box**: Configuración por defecto suele ser buena

#### Cons
- ⚠️ **Más lento que SAC**: Requiere más timesteps para convergencia
- ⚠️ **Exploración limitada**: Puede quedarse en óptimos locales
- ⚠️ **On-policy**: No puede reutilizar experiencias antiguas

#### Cuándo Usar PPO

- ✓ Primera vez entrenando RL
- ✓ Necesitas resultados confiables
- ✓ Recursos computacionales limitados
- ✓ Circuitos sencillos a moderados
- ✓ Prefieres estabilidad sobre velocidad

#### Ejemplo de Entrenamiento

```bash
# Básico (recomendado para empezar)
python scripts/train_pilot.py \
    --algorithm PPO \
    --total-timesteps 500000

# Con curriculum learning
python scripts/train_pilot.py \
    --algorithm PPO \
    --curriculum \
    --total-timesteps 1000000

# Circuito específico
python scripts/train_pilot.py \
    --algorithm PPO \
    --track tracks/monza.json \
    --total-timesteps 500000 \
    --n-envs 8

# Alta performance
python scripts/train_pilot.py \
    --algorithm PPO \
    --n-envs 16 \
    --total-timesteps 1000000 \
    --learning-rate 3e-4 \
    --batch-size 64
```

#### Hiperparámetros Recomendados

```bash
--learning-rate 3e-4      # Default, funciona bien
--batch-size 64           # Equilibrado
--n-envs 8               # Standard (ajustar según CPU)
--total-timesteps 500000  # Mínimo para convergencia
```

---

### SAC (Soft Actor-Critic)

**Recomendado para:** Circuitos complejos, convergencia rápida, exploración

#### Características

| Aspecto | Valoración | Detalles |
|---------|-----------|----------|
| **Estabilidad** | ⭐⭐⭐ | Generalmente estable, sensible a hiperparámetros |
| **Velocidad** | ⭐⭐⭐⭐⭐ | Muy rápida convergencia |
| **Facilidad de uso** | ⭐⭐⭐ | Requiere más tuning que PPO |
| **Sample efficiency** | ⭐⭐⭐⭐⭐ | Excelente (off-policy con replay buffer) |
| **Exploración** | ⭐⭐⭐⭐⭐ | Exploración máxima (entropy regularization) |

#### Pros
- ✅ **Convergencia rápida**: Suele aprender más rápido que PPO
- ✅ **Excelente exploración**: Entropy bonus ayuda a descubrir estrategias
- ✅ **Sample efficient**: Reutiliza experiencia pasada (replay buffer)
- ✅ **Bueno para tareas complejas**: Maneja bien circuitos técnicos
- ✅ **Off-policy**: Puede aprender mientras explora

#### Cons
- ⚠️ **Sensible a hiperparámetros**: Requiere tuning cuidadoso
- ⚠️ **Mayor uso de memoria**: Replay buffer grande
- ⚠️ **Puede ser inestable**: Con mal tuning puede divergir
- ⚠️ **Requiere más compute**: Buffer + double Q-networks

#### Cuándo Usar SAC

- ✓ Circuitos técnicos y complejos
- ✓ Quieres convergencia rápida
- ✓ Tienes recursos computacionales
- ✓ Necesitas buena exploración
- ✓ Estás dispuesto a tunear hiperparámetros

#### Ejemplo de Entrenamiento

```bash
# Básico
python scripts/train_pilot.py \
    --algorithm SAC \
    --total-timesteps 500000

# Circuito complejo
python scripts/train_pilot.py \
    --algorithm SAC \
    --track tracks/technical.json \
    --total-timesteps 500000 \
    --n-envs 16

# Alta performance
python scripts/train_pilot.py \
    --algorithm SAC \
    --n-envs 32 \
    --batch-size 256 \
    --learning-rate 3e-4 \
    --total-timesteps 1000000

# Multi-track
python scripts/train_pilot.py \
    --algorithm SAC \
    --multi-track \
    --n-envs 16 \
    --total-timesteps 1000000
```

#### Hiperparámetros Recomendados

```bash
--learning-rate 3e-4      # Standard para SAC
--batch-size 256          # Más grande que PPO
--n-envs 16              # Beneficia de más paralelización
--total-timesteps 500000  # Suele converger más rápido que PPO
```

---

### TD3 (Twin Delayed DDPG)

**Recomendado para:** Control preciso, time trials, comportamiento determinístico

#### Características

| Aspecto | Valoración | Detalles |
|---------|-----------|----------|
| **Estabilidad** | ⭐⭐⭐⭐ | Estable, más que SAC |
| **Velocidad** | ⭐⭐⭐⭐ | Convergencia rápida |
| **Facilidad de uso** | ⭐⭐⭐ | Moderada, menos sensible que SAC |
| **Sample efficiency** | ⭐⭐⭐⭐ | Muy buena (off-policy) |
| **Exploración** | ⭐⭐⭐ | Exploración moderada |

#### Pros
- ✅ **Política determinística**: Comportamiento predecible
- ✅ **Control preciso**: Excelente para maniobras finas
- ✅ **Más estable que DDPG**: Twin critics reducen overestimation
- ✅ **Off-policy**: Reutiliza experiencia pasada
- ✅ **Buen equilibrio**: Entre estabilidad y velocidad

#### Cons
- ⚠️ **Menos exploración que SAC**: Puede quedarse en óptimos locales
- ⚠️ **Requiere replay buffer**: Mayor uso de memoria
- ⚠️ **Sensible a noise**: Necesita configurar noise adecuadamente

#### Cuándo Usar TD3

- ✓ Necesitas control determinístico
- ✓ Time trials / qualifying laps
- ✓ Maniobras de precisión
- ✓ Quieres algo entre PPO y SAC
- ✓ Circuitos donde la precisión importa más que la exploración

#### Ejemplo de Entrenamiento

```bash
# Básico
python scripts/train_pilot.py \
    --algorithm TD3 \
    --total-timesteps 500000

# Time trial en circuito específico
python scripts/train_pilot.py \
    --algorithm TD3 \
    --track tracks/monza.json \
    --total-timesteps 500000 \
    --n-envs 8

# Precision training
python scripts/train_pilot.py \
    --algorithm TD3 \
    --learning-rate 1e-3 \
    --batch-size 100 \
    --n-envs 8 \
    --total-timesteps 500000
```

#### Hiperparámetros Recomendados

```bash
--learning-rate 1e-3      # Puede ser más alto que PPO/SAC
--batch-size 100          # Moderado
--n-envs 8               # Standard
--total-timesteps 500000  # Similar a SAC
```

---

## Cómo Entrenar

### Entrenamiento Básico

#### PPO (Recomendado para empezar)

```bash
python scripts/train_pilot.py \
    --algorithm PPO \
    --total-timesteps 500000
```

#### SAC (Para circuitos complejos)

```bash
python scripts/train_pilot.py \
    --algorithm SAC \
    --total-timesteps 500000 \
    --n-envs 16
```

#### TD3 (Para control preciso)

```bash
python scripts/train_pilot.py \
    --algorithm TD3 \
    --total-timesteps 500000
```

---

### Opciones de Entrenamiento

#### 1. Circuito Específico

```bash
python scripts/train_pilot.py \
    --algorithm PPO \
    --track tracks/monza.json \
    --total-timesteps 500000
```

#### 2. Multi-Track (Generalización)

```bash
python scripts/train_pilot.py \
    --algorithm SAC \
    --multi-track \
    --total-timesteps 1000000 \
    --n-envs 16
```

#### 3. Por Dificultad

```bash
# Beginner
python scripts/train_pilot.py --difficulty 0 --total-timesteps 200000

# Intermediate
python scripts/train_pilot.py --difficulty 1 --total-timesteps 300000

# Advanced
python scripts/train_pilot.py --difficulty 2 --total-timesteps 500000

# Expert
python scripts/train_pilot.py --difficulty 3 --total-timesteps 1000000
```

#### 4. Curriculum Learning (Recomendado)

```bash
# Progresión automática de dificultad
python scripts/train_pilot.py \
    --curriculum \
    --algorithm PPO \
    --total-timesteps 1000000

# Empezar desde nivel intermedio
python scripts/train_pilot.py \
    --curriculum \
    --curriculum-level 1 \
    --total-timesteps 500000
```

📖 **Ver:** [Curriculum Learning Guide](../docs/CURRICULUM_LEARNING.md)

#### 5. Transfer Learning

```bash
# Entrenar en óvalo
python scripts/train_pilot.py \
    --track tracks/oval.json \
    --total-timesteps 200000 \
    --model-dir models/stage1/

# Transferir a circuito complejo
python scripts/train_pilot.py \
    --track tracks/monza.json \
    --load-model models/stage1/PPO_oval_final.zip \
    --total-timesteps 500000 \
    --model-dir models/stage2/
```

#### 6. Alta Performance

```bash
python scripts/train_pilot.py \
    --algorithm SAC \
    --n-envs 32 \
    --batch-size 256 \
    --total-timesteps 2000000 \
    --checkpoint-freq 100000
```

---

### Configuración de Hiperparámetros

#### Learning Rate

```bash
# Conservador (más estable, más lento)
--learning-rate 1e-4

# Standard (recomendado)
--learning-rate 3e-4

# Agresivo (más rápido, menos estable)
--learning-rate 1e-3
```

#### Batch Size

**PPO:**
```bash
--batch-size 64   # Standard
--batch-size 128  # Más estable
```

**SAC/TD3:**
```bash
--batch-size 256  # Recomendado
--batch-size 512  # Si hay memoria suficiente
```

#### Parallel Environments

```bash
--n-envs 4    # Recursos limitados
--n-envs 8    # Standard
--n-envs 16   # Alta performance
--n-envs 32   # Máximo (CPU potente)
```

**Nota:** Más entornos = más datos/segundo pero más CPU

#### Training Duration

```bash
--total-timesteps 100000     # Test rápido
--total-timesteps 500000     # Training standard
--total-timesteps 1000000    # Training completo
--total-timesteps 2000000    # Training extendido
```

---

## Cómo Evaluar

### Evaluación Básica

Después de entrenar, evalúa el modelo:

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_default_final.zip \
    --episodes 10
```

**Salida:**
- Métricas en consola
- JSON report: `results/PPO_default_evaluation.json`
- Plots: `results/PPO_default_plots.png`

### Evaluación en Circuito Específico

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_monza_final.zip \
    --track tracks/monza.json \
    --episodes 20 \
    --output results/monza/
```

### Evaluación con Visualización

```bash
# Render en tiempo real
python scripts/evaluate.py \
    --model trained_models/PPO_final.zip \
    --render \
    --episodes 5

# Grabar video
python scripts/evaluate.py \
    --model trained_models/PPO_final.zip \
    --record \
    --record-path recordings/best_lap.mp4
```

**Requiere:** `pip install opencv-python`

### Comparar Dos Modelos

```bash
python scripts/evaluate.py \
    --model trained_models/PPO_v1.zip \
    --compare trained_models/SAC_v1.zip \
    --episodes 20
```

**Salida:**
- Comparación lado a lado
- Ganador por métrica
- Diferencias en %
- `results/comparison.json`

📊 **Ver:** [Evaluation Guide](../docs/EVALUATION_GUIDE.md)

---

## Comparación de Todos los Modelos

### Script: Entrenar Todos los Algoritmos

```bash
#!/bin/bash
# train_all.sh

TRACK="tracks/monza.json"
TIMESTEPS=500000

# Entrenar con cada algoritmo
for algo in PPO SAC TD3; do
    echo "Entrenando $algo..."
    python scripts/train_pilot.py \
        --algorithm "$algo" \
        --track "$TRACK" \
        --total-timesteps "$TIMESTEPS" \
        --model-dir "models/${algo}/"
done

echo "Entrenamiento completo!"
```

**Ejecutar:**
```bash
chmod +x train_all.sh
./train_all.sh
```

---

### Script: Evaluar Todos los Modelos

```bash
#!/bin/bash
# evaluate_all.sh

TRACK="tracks/monza.json"
EPISODES=20

# Evaluar cada algoritmo
for algo in PPO SAC TD3; do
    echo "Evaluando $algo..."
    python scripts/evaluate.py \
        --model "models/${algo}/${algo}_monza_final.zip" \
        --track "$TRACK" \
        --episodes "$EPISODES" \
        --output "results/${algo}/"
done

echo "Evaluación completa!"
echo "Resultados en: results/"
```

**Ejecutar:**
```bash
chmod +x evaluate_all.sh
./evaluate_all.sh
```

---

### Script: Comparaciones Directas

```bash
#!/bin/bash
# compare_algorithms.sh

TRACK="tracks/monza.json"
EPISODES=20

# PPO vs SAC
echo "Comparando PPO vs SAC..."
python scripts/evaluate.py \
    --model "models/PPO/PPO_monza_final.zip" \
    --compare "models/SAC/SAC_monza_final.zip" \
    --track "$TRACK" \
    --episodes "$EPISODES" \
    --output "results/comparison_PPO_vs_SAC/"

# SAC vs TD3
echo "Comparando SAC vs TD3..."
python scripts/evaluate.py \
    --model "models/SAC/SAC_monza_final.zip" \
    --compare "models/TD3/TD3_monza_final.zip" \
    --track "$TRACK" \
    --episodes "$EPISODES" \
    --output "results/comparison_SAC_vs_TD3/"

# PPO vs TD3
echo "Comparando PPO vs TD3..."
python scripts/evaluate.py \
    --model "models/PPO/PPO_monza_final.zip" \
    --compare "models/TD3/TD3_monza_final.zip" \
    --track "$TRACK" \
    --episodes "$EPISODES" \
    --output "results/comparison_PPO_vs_TD3/"

echo "Comparaciones completas!"
echo "Ver resultados en: results/comparison_*/"
```

---

### Python Script: Reporte Comparativo

```python
#!/usr/bin/env python3
"""
generate_comparison_report.py

Genera un reporte CSV comparando todos los modelos.
"""

import json
import csv
from pathlib import Path
from glob import glob

# Configuración
results_dir = Path("results")
output_file = "algorithm_comparison.csv"

# Recopilar resultados de todos los algoritmos
all_results = []

for algo in ["PPO", "SAC", "TD3"]:
    json_file = results_dir / algo / f"{algo}_monza_evaluation.json"

    if json_file.exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
            metrics = data['metrics']

            all_results.append({
                'Algorithm': algo,
                'Completion Rate': f"{metrics['completion_rate']:.2%}",
                'Best Lap Time': f"{metrics['lap_time_best']:.2f}s",
                'Mean Lap Time': f"{metrics['lap_time_mean']:.2f}s",
                'Lap Time Std': f"{metrics['lap_time_std']:.2f}s",
                'On Track %': f"{metrics['on_track_percentage']:.1f}%",
                'Off Track Count': int(metrics['off_track_count_total']),
                'Max Velocity': f"{metrics['max_velocity']:.1f} m/s",
                'Mean Reward': f"{metrics['total_reward_mean']:.2f}",
            })

# Guardar a CSV
with open(output_file, 'w', newline='') as f:
    if all_results:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

print(f"✓ Reporte guardado: {output_file}")
print(f"Total de algoritmos evaluados: {len(all_results)}")

# Imprimir tabla en consola
print("\n" + "="*80)
print("COMPARACIÓN DE ALGORITMOS")
print("="*80)

for result in all_results:
    print(f"\n{result['Algorithm']}:")
    for key, value in result.items():
        if key != 'Algorithm':
            print(f"  {key:20s}: {value}")
```

**Ejecutar:**
```bash
python generate_comparison_report.py
```

---

### Evaluación Multi-Track

Evaluar cada algoritmo en todos los circuitos:

```bash
#!/bin/bash
# evaluate_multitrack.sh

EPISODES=10
TRACKS=("tracks/oval.json" "tracks/simple.json" "tracks/technical.json")

for algo in PPO SAC TD3; do
    for track in "${TRACKS[@]}"; do
        track_name=$(basename "$track" .json)

        python scripts/evaluate.py \
            --model "models/${algo}/${algo}_multi_final.zip" \
            --track "$track" \
            --episodes "$EPISODES" \
            --output "results/${algo}/${track_name}/"
    done
done
```

---

## 📊 Tabla Resumen: Qué Algoritmo Usar

| Escenario | Algoritmo Recomendado | Por Qué |
|-----------|----------------------|---------|
| Primera vez entrenando | **PPO** | Más estable y fácil |
| Circuito simple (óvalo) | **PPO** | Suficiente y eficiente |
| Circuito técnico complejo | **SAC** | Mejor exploración |
| Quiero convergencia rápida | **SAC** | Off-policy, más eficiente |
| Time trials / qualifying | **TD3** | Control determinístico preciso |
| Recursos limitados | **PPO** | Menor uso de memoria |
| Multi-track generalización | **PPO** + curriculum | Estable en diferentes entornos |
| Exploración importante | **SAC** | Entropy regularization |
| Necesito precisión | **TD3** | Política determinística |

---

## 💡 Tips Finales

1. **Empieza con PPO** - Aprende el proceso con el algoritmo más estable
2. **Usa curriculum learning** - `--curriculum` para mejor generalización
3. **Monitorea TensorBoard** - `tensorboard --logdir logs/`
4. **Guarda checkpoints** - Configurado por defecto cada 50k steps
5. **Evalúa frecuentemente** - Ver progreso en eval callback
6. **Compara algoritmos** - Prueba los 3 para ver cuál funciona mejor
7. **Ajusta n-envs** - Según tu CPU (8-16 suele ser óptimo)
8. **Sé paciente** - Buenas políticas necesitan 500k-1M timesteps

---

## 📚 Referencias

- [Evaluation Guide](../docs/EVALUATION_GUIDE.md) - Cómo evaluar modelos
- [Curriculum Learning](../docs/CURRICULUM_LEARNING.md) - Entrenamiento progresivo
- [Scripts README](README.md) - Documentación principal
- [Main README](../README.md) - Visión general del proyecto
