# Entrenamiento del Ingeniero de Carrera

Guía completa para entrenar el agente ingeniero que toma decisiones estratégicas sobre pit stops y gestión de neumáticos.

---

## 📋 Índice

- [¿Qué es el Ingeniero?](#qué-es-el-ingeniero)
- [Algoritmo DQN](#algoritmo-dqn)
- [Cómo Entrenar](#cómo-entrenar)
- [Cómo Evaluar](#cómo-evaluar)
- [Comparación de Estrategias](#comparación-de-estrategias)

---

## ¿Qué es el Ingeniero?

El **Engineer Agent** es responsable de la **estrategia de carrera**:

### Decisiones que Toma

1. **Cuándo hacer pit stop**
   - Monitorea desgaste de neumáticos
   - Evalúa tiempo restante de carrera
   - Decide momento óptimo para parar

2. **Qué compuesto de neumático usar**
   - **Soft**: Rápidos pero se desgastan rápido
   - **Medium**: Equilibrados
   - **Hard**: Duraderos pero más lentos

3. **Gestión de la carrera**
   - Planifica número de pit stops
   - Optimiza tiempo total de carrera
   - Balancea velocidad vs durabilidad

### Espacio de Acciones

El ingeniero tiene **4 acciones discretas**:

```
0: Continue (no pit)
1: Pit - Soft tyres
2: Pit - Medium tyres
3: Pit - Hard tyres
```

### Observaciones

El ingeniero observa:
- Desgaste actual de neumáticos (%)
- Vuelta actual
- Vueltas totales
- Velocidad del coche
- Posición en pista
- Compuesto actual de neumático

---

## Algoritmo DQN

El ingeniero usa **DQN (Deep Q-Network)** para aprender la estrategia óptima.

### DQN (Deep Q-Network)

**Tipo:** Off-policy, Value-based, Discrete actions

#### Características

| Aspecto | Valoración | Detalles |
|---------|-----------|----------|
| **Estabilidad** | ⭐⭐⭐⭐ | Bastante estable con experience replay |
| **Velocidad** | ⭐⭐⭐ | Convergencia moderada |
| **Facilidad de uso** | ⭐⭐⭐⭐ | Sencillo de configurar |
| **Sample efficiency** | ⭐⭐⭐⭐ | Buena (off-policy con replay buffer) |
| **Mejor para** | Acciones discretas | Pit/no pit, tipo de neumático |

#### Pros
- ✅ **Ideal para decisiones discretas**: Perfecto para el rol del ingeniero
- ✅ **Sample efficient**: Reutiliza experiencia con replay buffer
- ✅ **Estable**: Experience replay reduce correlación
- ✅ **Interpretable**: Q-values muestran valor de cada acción

#### Cons
- ⚠️ **Solo acciones discretas**: No puede usarse para control continuo
- ⚠️ **Requiere exploration**: Epsilon-greedy para balance exploración/explotación
- ⚠️ **Convergencia más lenta que DQN moderno**: Rainbow DQN sería mejor pero más complejo

#### Por Qué DQN para el Ingeniero

1. **Acciones discretas naturales**: Pit/no pit es binario, tipo de neumático es categórico
2. **Horizonte largo**: Decisiones estratégicas a lo largo de toda la carrera
3. **Sample efficiency importante**: Las carreras son largas, queremos aprender rápido
4. **Estabilidad**: Decisiones críticas requieren política confiable

---

## Cómo Entrenar

### Entrenamiento Básico

```bash
python scripts/train_engineer.py \
    --track monza \
    --timesteps 500000
```

**Esto crea:**
- Modelo entrenado: `trained_models/engineer_final_monza.zip`
- Checkpoints: `trained_models/engineer_checkpoint_*.zip`
- Logs (si `--tensorboard`): `logs/`

### Opciones de Entrenamiento

#### 1. Circuito Específico

```bash
python scripts/train_engineer.py \
    --track monza \
    --timesteps 500000
```

#### 2. Con TensorBoard

```bash
python scripts/train_engineer.py \
    --track monza \
    --timesteps 500000 \
    --tensorboard
```

**Monitorear:**
```bash
tensorboard --logdir logs/
```

#### 3. Configuración Personalizada

```bash
python scripts/train_engineer.py \
    --track monza \
    --timesteps 1000000 \
    --learning-rate 1e-4 \
    --save-freq 50000
```

#### 4. Continuar Entrenamiento

```bash
# Entrenar primero
python scripts/train_engineer.py \
    --track monza \
    --timesteps 250000 \
    --model-dir models/stage1/

# Continuar desde checkpoint
python scripts/train_engineer.py \
    --track monza \
    --timesteps 500000 \
    --load-model models/stage1/engineer_checkpoint_250000.zip \
    --model-dir models/stage2/
```

---

### Argumentos Disponibles

```bash
python scripts/train_engineer.py \
    --track NAME \              # Nombre del circuito
    --timesteps N \             # Timesteps totales de entrenamiento
    --learning-rate LR \        # Learning rate (default: 1e-4)
    --save-freq N \             # Guardar checkpoint cada N steps
    --tensorboard \             # Habilitar logging a TensorBoard
    --device {cpu,cuda,auto}    # Dispositivo (default: cpu)
```

### Hiperparámetros Recomendados

```bash
# Standard (recomendado)
--learning-rate 1e-4
--timesteps 500000

# Entrenamiento rápido
--learning-rate 3e-4
--timesteps 250000

# Entrenamiento completo
--learning-rate 1e-4
--timesteps 1000000
```

---

## Cómo Evaluar

### Evaluación Básica

```bash
python scripts/evaluate.py \
    --model trained_models/engineer_final_monza.zip \
    --track tracks/monza.json \
    --episodes 10
```

**Salida:**
- Métricas de rendimiento
- Decisiones de pit stop por episodio
- Gestión de neumáticos
- JSON report y plots

### Evaluación Detallada

```bash
python scripts/evaluate.py \
    --model trained_models/engineer_final_monza.zip \
    --track tracks/monza.json \
    --episodes 20 \
    --output results/engineer/ \
    --record
```

### Métricas Importantes para el Ingeniero

Al evaluar, presta atención a:

1. **Completion Rate** - ¿El ingeniero completa carreras?
2. **Tyre Wear per Lap** - ¿Gestiona bien el desgaste?
3. **Lap Times** - ¿Las decisiones mejoran tiempos?
4. **Pit Stop Timing** - ¿Hace pit en momentos óptimos?

---

## Comparación de Estrategias

### Entrenar Múltiples Estrategias

Puedes entrenar con diferentes configuraciones y comparar:

```bash
#!/bin/bash
# train_engineer_strategies.sh

TRACK="monza"

# Estrategia conservadora (learning rate bajo)
python scripts/train_engineer.py \
    --track "$TRACK" \
    --timesteps 500000 \
    --learning-rate 5e-5 \
    --model-dir models/engineer_conservative/

# Estrategia agresiva (learning rate alto)
python scripts/train_engineer.py \
    --track "$TRACK" \
    --timesteps 500000 \
    --learning-rate 3e-4 \
    --model-dir models/engineer_aggressive/

# Estrategia balanced (standard)
python scripts/train_engineer.py \
    --track "$TRACK" \
    --timesteps 500000 \
    --learning-rate 1e-4 \
    --model-dir models/engineer_balanced/
```

### Evaluar Estrategias

```bash
#!/bin/bash
# evaluate_engineer_strategies.sh

TRACK="tracks/monza.json"
EPISODES=20

for strategy in conservative aggressive balanced; do
    python scripts/evaluate.py \
        --model "models/engineer_${strategy}/engineer_final_monza.zip" \
        --track "$TRACK" \
        --episodes "$EPISODES" \
        --output "results/engineer_${strategy}/"
done
```

### Comparar Directamente

```bash
# Conservative vs Aggressive
python scripts/evaluate.py \
    --model models/engineer_conservative/engineer_final_monza.zip \
    --compare models/engineer_aggressive/engineer_final_monza.zip \
    --track tracks/monza.json \
    --episodes 20

# Balanced vs Aggressive
python scripts/evaluate.py \
    --model models/engineer_balanced/engineer_final_monza.zip \
    --compare models/engineer_aggressive/engineer_final_monza.zip \
    --track tracks/monza.json \
    --episodes 20
```

---

## Workflow Completo: Piloto + Ingeniero

### Entrenar Ambos Agentes

```bash
#!/bin/bash
# train_both_agents.sh

TRACK_NAME="monza"
TRACK_PATH="tracks/monza.json"

# 1. Entrenar Piloto
echo "Entrenando piloto..."
python scripts/train_pilot.py \
    --algorithm PPO \
    --track "$TRACK_PATH" \
    --total-timesteps 500000 \
    --model-dir models/pilot/

# 2. Entrenar Ingeniero
echo "Entrenando ingeniero..."
python scripts/train_engineer.py \
    --track "$TRACK_NAME" \
    --timesteps 500000 \
    --model-dir models/engineer/

echo "Entrenamiento completo!"
```

### Evaluar Ambos Agentes

```bash
#!/bin/bash
# evaluate_both_agents.sh

TRACK_PATH="tracks/monza.json"
EPISODES=20

# Evaluar piloto
echo "Evaluando piloto..."
python scripts/evaluate.py \
    --model models/pilot/PPO_monza_final.zip \
    --track "$TRACK_PATH" \
    --episodes "$EPISODES" \
    --output results/pilot/

# Evaluar ingeniero
echo "Evaluando ingeniero..."
python scripts/evaluate.py \
    --model models/engineer/engineer_final_monza.zip \
    --track "$TRACK_PATH" \
    --episodes "$EPISODES" \
    --output results/engineer/

echo "Evaluación completa!"
echo "Resultados en: results/"
```

---

## Análisis de Decisiones del Ingeniero

### Script: Analizar Pit Stops

```python
#!/usr/bin/env python3
"""
analyze_pit_strategy.py

Analiza las decisiones de pit stop del ingeniero.
"""

import json
from pathlib import Path

# Cargar resultados
results_file = Path("results/engineer/engineer_final_monza_evaluation.json")

with open(results_file, 'r') as f:
    data = json.load(f)

# Analizar por episodio
print("="*60)
print("ANÁLISIS DE ESTRATEGIA DE PIT STOPS")
print("="*60)

for ep in data['per_episode']:
    episode_num = ep['episode']
    laps = ep['laps_completed']
    lap_times = ep.get('lap_times', [])

    print(f"\nEpisodio {episode_num}:")
    print(f"  Vueltas completadas: {laps}")

    if lap_times:
        print(f"  Lap times: {[f'{t:.2f}s' for t in lap_times]}")
        print(f"  Mejor vuelta: {min(lap_times):.2f}s")
        print(f"  Promedio: {sum(lap_times)/len(lap_times):.2f}s")

# Métricas generales
metrics = data['metrics']
print(f"\n{'='*60}")
print("MÉTRICAS GENERALES")
print(f"{'='*60}")
print(f"Tasa de finalización: {metrics['completion_rate']:.1%}")
print(f"Desgaste por vuelta: {metrics['tyre_wear_per_lap_mean']:.1f}%")
print(f"Reward promedio: {metrics['total_reward_mean']:.2f}")
```

---

## Estrategias de Neumáticos

### Tipos de Compuesto

| Compuesto | Velocidad | Durabilidad | Cuándo Usar |
|-----------|-----------|-------------|-------------|
| **Soft** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Sprint final, qualifying, pocos laps restantes |
| **Medium** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Equilibrio, stint medio de carrera |
| **Hard** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Stint largo, inicio de carrera, minimizar pit stops |

### Estrategias Comunes

#### 1-Stop Strategy
```
Start: Hard → Lap 15: Soft → Finish
```
- Una sola parada
- Hard para aguantar, Soft para sprint final

#### 2-Stop Strategy
```
Start: Soft → Lap 8: Medium → Lap 16: Soft → Finish
```
- Dos paradas
- Mantiene ritmo alto toda la carrera
- Más tiempo perdido en pits

#### No-Stop Strategy
```
Start: Hard → Finish (no pit)
```
- Cero paradas
- Solo viable en carreras cortas
- Requiere gestión agresiva del desgaste

---

## 💡 Tips para el Ingeniero

1. **Entrena suficiente tiempo** - El ingeniero necesita aprender timing de pit stops (500k-1M timesteps)
2. **Monitorea decisiones** - Usa TensorBoard para ver cuándo hace pit stops
3. **Evalúa en carreras largas** - Estrategia se nota mejor con más vueltas (`--max-laps 10`)
4. **Compara con baseline** - Evalúa contra estrategia simple (e.g., pit en lap 10)
5. **Learning rate conservador** - 1e-4 funciona bien para decisiones estratégicas
6. **Paciencia** - DQN tarda más en converger que algoritmos continuos

---

## 🔍 Debugging

### El ingeniero no hace pit stops

**Posibles causas:**
- Reward function no penaliza suficiente desgaste alto
- Learning rate muy bajo
- No ha entrenado suficiente

**Solución:**
```bash
# Entrenar más tiempo con learning rate mayor
python scripts/train_engineer.py \
    --timesteps 1000000 \
    --learning-rate 3e-4
```

### Hace demasiados pit stops

**Posibles causas:**
- Reward function penaliza demasiado desgaste
- Exploration rate muy alto

**Solución:**
- Ajustar reward function en el código
- Entrenar más para que converja

### Lap times inconsistentes

**Posible causa:**
- Decisiones de neumáticos no óptimas

**Solución:**
- Entrenar más tiempo
- Evaluar que el piloto funcione bien primero

---

## 📚 Referencias

- [Evaluation Guide](../docs/EVALUATION_GUIDE.md) - Cómo evaluar modelos
- [TRAIN_PILOT.md](TRAIN_PILOT.md) - Entrenamiento del piloto
- [Scripts README](README.md) - Documentación principal
- [DQN Paper](https://arxiv.org/abs/1312.5602) - Artículo original de DQN

---

**¡Buena suerte con tu ingeniero de carrera! 🏎️🔧**
