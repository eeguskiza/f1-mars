# Configuración de Display para WSL2

## Problema

Arcade requiere un display gráfico (OpenGL). En WSL2, necesitas configurar X11 o usar WSLg.

## Solución 1: WSLg (Recomendado - Windows 11)

Si tienes **Windows 11**, WSLg ya está incluido:

```bash
# Verificar que WSLg está disponible
echo $DISPLAY
# Debería mostrar algo como: :0

# Si no está configurado, añadir a ~/.bashrc:
export DISPLAY=:0
```

Luego simplemente ejecuta:
```bash
python scripts/watch_agent.py --model trained_models/PPO_default_final.zip
```

## Solución 2: VcXsrv (Windows 10)

1. **Instalar VcXsrv en Windows:**
   - Descargar de: https://sourceforge.net/projects/vcxsrv/
   - Instalar y ejecutar XLaunch
   - Configuración:
     - Multiple windows
     - Start no client
     - ✅ Disable access control

2. **Configurar WSL:**
```bash
# Añadir a ~/.bashrc
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1

# Recargar
source ~/.bashrc
```

3. **Ejecutar:**
```bash
python scripts/watch_agent.py --model trained_models/PPO_default_final.zip
```

## Solución 3: Renderizado Virtual (Sin Display)

Para entrenar sin display, usa el flag de headless:

```bash
# TODO: Implementar modo headless
# Por ahora, usa el viewer normal en una máquina con display
```

## Verificación

```bash
# Test simple de Arcade
python -c "
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import arcade
print('✓ Arcade imports successfully')
"

# Test de OpenGL (requiere display)
python -c "
import arcade
window = arcade.Window(100, 100)
print('✓ Window created successfully')
window.close()
"
```

## Alternativa: Ejecutar en Windows Nativo

Si tienes problemas con WSL, ejecuta directamente en Windows:

```bash
# En Windows PowerShell o CMD
cd C:\path\to\f1_mars
python scripts\watch_agent.py --model trained_models\PPO_default_final.zip
```

## Estado Actual

- ✅ Código migrado a Arcade correctamente
- ✅ Todas las importaciones funcionan
- ✅ Tests de sintaxis pasados
- ⚠️ Requiere display gráfico para ejecutar

## Próximos Pasos

1. Configurar display según tu sistema (WSLg o VcXsrv)
2. Ejecutar el visualizador
3. Disfrutar del renderizado GPU! 🚀
