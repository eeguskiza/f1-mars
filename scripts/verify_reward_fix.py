#!/usr/bin/env python3
"""
Verificación rápida del fix de reward function.
Ejecuta entrenamientos cortos y verifica que el comportamiento cambió.
"""

import subprocess
import sys
import time
import json
from pathlib import Path


def run_quick_training(algorithm: str, timesteps: int = 30000) -> dict:
    """Ejecuta entrenamiento corto y retorna métricas."""
    print(f"\n{'='*60}")
    print(f"  VERIFICACIÓN: {algorithm} - {timesteps} steps")
    print(f"{'='*60}\n")

    model_name = f"verify_{algorithm}_{int(time.time())}"

    cmd = [
        sys.executable, "scripts/train_pilot.py",
        "--algorithm", algorithm,
        "--total-timesteps", str(timesteps),
        "--eval-freq", str(timesteps),  # Solo evaluar al final
        "--n-envs", "4",
        "--model-dir", "trained_models/verification",
        "--tensorboard-log", "logs/verification"
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    return {
        "algorithm": algorithm,
        "timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "return_code": result.returncode
    }


def run_evaluation(model_path: str) -> dict:
    """Evalúa un modelo y retorna métricas clave."""
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--model", model_path,
        "--episodes", "5"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parsear output para extraer métricas
    output = result.stdout
    metrics = {
        "mean_velocity": 0,
        "lap_time": 0,
        "on_track_pct": 0,
        "reward": 0
    }

    for line in output.split('\n'):
        if 'Mean velocity:' in line or 'Velocity:' in line:
            try:
                # Extract number from line (handle different formats)
                parts = line.split(':')[1].strip().split()
                metrics["mean_velocity"] = float(parts[0])
            except:
                pass
        elif 'Lap time:' in line or 'Time:' in line:
            try:
                parts = line.split(':')[1].strip()
                # Remove 's' suffix if present
                time_str = parts.replace('s', '').split()[0]
                metrics["lap_time"] = float(time_str)
            except:
                pass
        elif 'On-track' in line or 'on track' in line.lower():
            try:
                parts = line.split(':')[1].strip().replace('%', '')
                metrics["on_track_pct"] = float(parts.split()[0])
            except:
                pass
        elif 'reward:' in line.lower() or 'mean reward' in line.lower():
            try:
                parts = line.split(':')[1].strip().split()
                metrics["reward"] = float(parts[0])
            except:
                pass

    return metrics


def verify_behavior_improved(metrics: dict) -> tuple[bool, list[str]]:
    """
    Verifica que las métricas indican comportamiento mejorado.

    Criterios de éxito:
    1. Mean velocity > 20 m/s (antes era 0.6 m/s)
    2. Lap time < 120s (antes era 293s)
    3. Reward > -1000 (antes era -3600)
    """
    issues = []

    if metrics["mean_velocity"] < 20:
        issues.append(f"❌ Velocidad muy baja: {metrics['mean_velocity']:.1f} m/s (esperado > 20)")
    else:
        print(f"✅ Velocidad OK: {metrics['mean_velocity']:.1f} m/s")

    if metrics["lap_time"] > 120 and metrics["lap_time"] > 0:
        issues.append(f"❌ Lap time muy alto: {metrics['lap_time']:.1f}s (esperado < 120s)")
    elif metrics["lap_time"] > 0:
        print(f"✅ Lap time OK: {metrics['lap_time']:.1f}s")

    if metrics["reward"] < -1000:
        issues.append(f"❌ Reward muy negativo: {metrics['reward']:.1f} (esperado > -1000)")
    else:
        print(f"✅ Reward OK: {metrics['reward']:.1f}")

    return len(issues) == 0, issues


def main():
    print("\n" + "="*70)
    print("  F1-MARS REWARD FUNCTION VERIFICATION")
    print("  Verificando que el fix de reward produce comportamiento correcto")
    print("="*70)

    # Verificar que el entorno funciona
    print("\n[1/4] Verificando imports...")
    try:
        from f1_mars.envs import F1Env
        env = F1Env(max_laps=1)
        obs, info = env.reset()

        # Test rápido: ejecutar algunos steps con acción aleatoria
        total_reward = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            if term or trunc:
                break

        env.close()
        print(f"   ✅ Entorno funciona. Reward en 100 steps random: {total_reward:.2f}")

        # Verificar que el reward no es extremadamente negativo con acción random
        if total_reward < -500:
            print(f"   ⚠️  Warning: Reward muy negativo incluso con acciones random")
            print(f"      Esto puede indicar que las penalizaciones siguen dominando")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test de comportamiento básico: aceleración constante
    print("\n[2/4] Test de aceleración constante...")
    try:
        env = F1Env(max_laps=1)
        obs, info = env.reset()

        # Acelerar recto durante 5 segundos (300 steps)
        accelerate_action = [0.0, 1.0, 0.0]  # steering=0, throttle=1, brake=0
        total_reward = 0
        max_velocity = 0
        step_rewards = []

        for step in range(300):
            obs, reward, term, trunc, info = env.step(accelerate_action)
            total_reward += reward
            step_rewards.append(reward)
            max_velocity = max(max_velocity, info.get('velocity', 0))
            if term or trunc:
                break

        env.close()

        print(f"   Max velocity alcanzada: {max_velocity:.1f} m/s ({max_velocity*3.6:.1f} km/h)")
        print(f"   Reward total (5s acelerando): {total_reward:.2f}")
        print(f"   Reward promedio por step: {total_reward/len(step_rewards):.3f}")

        # Analizar tendencia de rewards
        avg_first_half = sum(step_rewards[:len(step_rewards)//2]) / (len(step_rewards)//2)
        avg_second_half = sum(step_rewards[len(step_rewards)//2:]) / (len(step_rewards) - len(step_rewards)//2)
        print(f"   Reward primera mitad: {avg_first_half:.3f}")
        print(f"   Reward segunda mitad: {avg_second_half:.3f}")

        if total_reward > 0:
            print(f"   ✅ Reward POSITIVO al acelerar - Fix funciona!")
        elif total_reward > -100:
            print(f"   ⚠️  Reward ligeramente negativo pero aceptable")
            print(f"      Esto puede mejorar con entrenamiento")
        else:
            print(f"   ❌ Reward muy negativo al acelerar - Fix NO funciona")
            print(f"      El reward debería ser positivo al ir rápido en dirección correcta")
            return 1

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Entrenamientos cortos de verificación
    print("\n[3/4] Ejecutando entrenamiento corto con PPO...")
    print("   (Esto tomará varios minutos...)")

    # Crear directorios
    Path("trained_models/verification").mkdir(parents=True, exist_ok=True)
    Path("logs/verification").mkdir(parents=True, exist_ok=True)

    try:
        training_result = run_quick_training("PPO", timesteps=20000)

        if training_result["return_code"] != 0:
            print("   ❌ Entrenamiento falló")
            return 1

        print(f"   ✅ Entrenamiento completado en {training_result['elapsed_seconds']:.1f}s")

    except Exception as e:
        print(f"   ❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Verificar modelo entrenado
    print("\n[4/4] Evaluando modelo entrenado...")

    # Buscar el modelo más reciente
    model_dir = Path("trained_models/verification")
    models = list(model_dir.glob("*.zip"))

    if not models:
        # Buscar en directorio por defecto
        model_dir = Path("trained_models")
        models = list(model_dir.glob("**/best_model.zip")) + list(model_dir.glob("**/*final*.zip"))

    if models:
        latest_model = max(models, key=lambda x: x.stat().st_mtime)
        print(f"   Evaluando: {latest_model}")

        try:
            eval_metrics = run_evaluation(str(latest_model))
            success, issues = verify_behavior_improved(eval_metrics)

            if not success:
                print("\n   Problemas detectados:")
                for issue in issues:
                    print(f"   {issue}")
        except Exception as e:
            print(f"   ⚠️  Error al evaluar: {e}")
            print(f"   Verifica manualmente con: python scripts/evaluate.py --model {latest_model}")
    else:
        print("   ⚠️  No se encontró modelo para evaluar")
        print("   Verifica manualmente con: python scripts/evaluate.py --model <path>")

    # Resumen final
    print("\n" + "="*70)
    print("  RESUMEN DE VERIFICACIÓN")
    print("="*70)

    print("""
    Si los tests pasaron:
    ✅ El fix de reward function está funcionando
    ✅ Puedes proceder con entrenamientos largos

    Si los tests fallaron:
    ❌ Revisa que _calculate_reward() fue modificado correctamente
    ❌ Verifica que no hay errores de sintaxis
    ❌ Compara con el código del prompt

    Comando para entrenamiento completo:
    python scripts/train_pilot.py --algorithm PPO --total-timesteps 500000

    Comando para ver progreso en TensorBoard:
    tensorboard --logdir logs/
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
