# Training Quick Start Guide

Guía rápida para entrenar agentes en circuitos F1 reales.

## 🎯 Entrenar en un Circuito Específico

### Circuitos Disponibles

| Circuito | Dificultad | Timesteps | Características |
|----------|-----------|-----------|-----------------|
| **Monza** | ⭐ Fácil | 500k | Ancho, alta velocidad, ideal para empezar |
| **Catalunya** | ⭐⭐ Media | 500k | Balanceado, mezcla de velocidades |
| **Yas Marina** | ⭐⭐ Media | 500k | Moderno, secciones técnicas |
| **Budapest** | ⭐⭐ Media | 500k | Estrecho, técnico, curvas lentas |
| **Austin** | ⭐⭐⭐ Media-Alta | 600k | Técnico, cambios de elevación |
| **Nürburgring** | ⭐⭐⭐ Media-Alta | 600k | Curvas variadas, técnico |
| **Spa** | ⭐⭐⭐⭐ Difícil | 800k | Largo, variado, desafiante |

### Entrenar en Monza (Recomendado para empezar)

```bash
python scripts/train_agent.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/monza_ppo \
    --eval-freq 10000
```

### Entrenar en Budapest (Técnico)

```bash
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/budapest_ppo \
    --eval-freq 10000
```

### Entrenar en Spa (Desafío)

```bash
python scripts/train_agent.py \
    --track tracks/spa.json \
    --algorithm PPO \
    --timesteps 800000 \
    --output trained_models/spa_ppo \
    --eval-freq 10000
```

## 🏆 Estrategia Progresiva (Recomendado si falla)

Si tu agente no completa vueltas, usa este curriculum progresivo:

### Paso 1: Monza (Fácil, Ancho)

```bash
python scripts/train_agent.py \
    --track tracks/monza.json \
    --algorithm PPO \
    --timesteps 500000 \
    --output trained_models/progressive_agent \
    --eval-freq 10000
```

**Objetivo:** Aprender control básico de velocidad y frenado.

### Paso 2: Catalunya (Media, Balanceado)

```bash
python scripts/train_agent.py \
    --track tracks/catalunya.json \
    --algorithm PPO \
    --timesteps 500000 \
    --model trained_models/progressive_agent/best_model.zip \
    --output trained_models/progressive_agent \
    --eval-freq 10000
```

**Objetivo:** Mejorar en curvas de media velocidad.

### Paso 3: Budapest (Técnico, Estrecho)

```bash
python scripts/train_agent.py \
    --track tracks/budapest.json \
    --algorithm PPO \
    --timesteps 500000 \
    --model trained_models/progressive_agent/best_model.zip \
    --output trained_models/progressive_agent \
    --eval-freq 10000
```

**Objetivo:** Dominar curvas técnicas y precisión.

### Paso 4: Spa (Desafío Completo)

```bash
python scripts/train_agent.py \
    --track tracks/spa.json \
    --algorithm PPO \
    --timesteps 800000 \
    --model trained_models/progressive_agent/best_model.zip \
    --output trained_models/progressive_agent \
    --eval-freq 10000
```

**Objetivo:** Circuito completo con todo tipo de curvas.

## 🌍 Agente Multi-Circuito (General)

Para entrenar un agente que funcione en todos los circuitos:

### Opción 1: Script Automático

```bash
bash scripts/train_all_circuits.sh PPO 500000
```

Esto entrena secuencialmente en los 7 circuitos (fácil → difícil).

### Opción 2: Manual con Loop

```bash
for circuit in monza catalunya yasmarina budapest austin nuerburgring spa; do
    echo "Entrenando en $circuit..."
    python scripts/train_agent.py \
        --track tracks/${circuit}.json \
        --algorithm PPO \
        --timesteps 500000 \
        --model trained_models/multi_circuit/best_model.zip \
        --output trained_models/multi_circuit \
        --eval-freq 10000
done
```

## 🎮 Visualizar Resultados

### Modo Interactivo (Recomendado)

```bash
python scripts/watch_agent.py
```

Esto te permite elegir modelo y circuito de listas interactivas.

### Modo Directo

```bash
python scripts/watch_agent.py \
    --model trained_models/monza_ppo/best_model.zip \
    --track tracks/monza.json \
    --laps 5
```

## ⚙️ Opciones de Algoritmo

### PPO (Recomendado para empezar)

```bash
--algorithm PPO --timesteps 500000
```

- Más estable
- Buena eficiencia de muestras
- Funciona bien en todos los circuitos

### SAC (Para alta velocidad)

```bash
--algorithm SAC --timesteps 800000
```

- Mejor exploración
- Bueno para Monza, Spa
- Requiere más timesteps

### TD3 (Para precisión)

```bash
--algorithm TD3 --timesteps 1000000
```

- Máxima precisión de control
- Bueno para Budapest, circuitos técnicos
- Requiere más tiempo de entrenamiento

## 🐛 Solución de Problemas

### Problema: El agente no completa vueltas

**Soluciones:**

1. **Aumenta timesteps:**
   ```bash
   --timesteps 1000000  # En vez de 500000
   ```

2. **Usa curriculum learning:**
   Empieza con Monza → Catalunya → Budapest

3. **Prueba SAC:**
   ```bash
   --algorithm SAC --timesteps 800000
   ```

4. **Visualiza qué falla:**
   ```bash
   python scripts/watch_agent.py
   # Observa dónde sale de pista
   ```

### Problema: Entrenamiento muy lento

**Soluciones:**

1. **Reduce timesteps para pruebas:**
   ```bash
   --timesteps 300000  # Para probar rápido
   ```

2. **Usa circuito más simple:**
   Monza es el más rápido de entrenar

3. **Verifica que usas CPU:**
   El script ya usa CPU por defecto (mejor para MLP)

### Problema: Recompensas negativas

**Normal si:**
- Primeros 100k steps (explorando)
- Circuito nuevo/difícil

**Problema si:**
- Persiste después de 300k steps
- No mejora gradualmente

**Soluciones:**

1. Aumenta timesteps a 1M
2. Reduce learning rate (edita script)
3. Prueba circuito más fácil primero

## 📊 Métricas de Éxito

| Métrica | Inicial | Bueno | Excelente |
|---------|---------|-------|-----------|
| Episode Reward | < 0 | > 500 | > 1500 |
| Laps Completed | 0 | 1-2 | 3+ |
| Track Limits | Muchos | Pocos | Ninguno |

## ⏱️ Tiempo de Entrenamiento

En CPU (recomendado):

- **500k timesteps:** 1-2 horas
- **800k timesteps:** 2-3 horas
- **1M timesteps:** 2-4 horas

## 📁 Archivos de Salida

Después del entrenamiento encontrarás:

```
trained_models/monza_ppo/
├── best_model.zip       # ← Usa este para visualizar
├── final_model.zip      # Modelo al final del entrenamiento
└── evaluations.npz      # Métricas de evaluación
```

**Siempre usa `best_model.zip` para testing.**

## 🚀 Comandos Rápidos

```bash
# Entrenar en Monza (fácil)
python scripts/train_agent.py --track tracks/monza.json --algorithm PPO --timesteps 500000 --output trained_models/monza_ppo --eval-freq 10000

# Ver resultados
python scripts/watch_agent.py

# Entrenar en todos los circuitos
bash scripts/train_all_circuits.sh PPO 500000

# Continuar entrenamiento existente
python scripts/train_agent.py --track tracks/budapest.json --algorithm PPO --timesteps 500000 --model trained_models/monza_ppo/best_model.zip --output trained_models/budapest_ppo
```

## 📚 Más Información

- **Guía completa:** Ver `TRAINING.md`
- **Documentación del entorno:** Ver `f1_mars/envs/f1_env.py`
- **Arquitectura:** Ver `README.md`
