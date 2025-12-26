# 🚀 Quick Start - F1-MARS Arcade Viewer

## ✅ Status: Todo Listo

**Código:** ✅ 100% funcional
**Tests:** ✅ Todos pasados
**Falta:** ⚠️ Configurar display en WSL

---

## 🎯 Configuración Rápida (3 pasos)

### 1️⃣ Verificar que todo funciona

```bash
# Test completo (sin display)
python test_watch_agent_dry_run.py
```

**Resultado esperado:**
```
✓ ALL INITIALIZATION TESTS PASSED
✅ watch_agent.py está listo para ejecutar!
```

---

### 2️⃣ Configurar Display

**Tienes 2 opciones:**

#### Opción A: Windows 11 con WSLg (Recomendado)

WSLg viene incluido en Windows 11. Solo necesitas:

```bash
# Configurar variable de entorno
export DISPLAY=:0

# Añadir a ~/.bashrc para que sea permanente
echo 'export DISPLAY=:0' >> ~/.bashrc
source ~/.bashrc
```

#### Opción B: VcXsrv (Windows 10)

1. **Instalar VcXsrv en Windows:**
   - Descargar: https://sourceforge.net/projects/vcxsrv/
   - Ejecutar XLaunch
   - Configuración:
     - ✅ Multiple windows
     - ✅ Start no client
     - ✅ **Disable access control** (importante!)

2. **Configurar WSL:**
```bash
# Añadir a ~/.bashrc
echo 'export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk "{print \$2}"):0' >> ~/.bashrc
echo 'export LIBGL_ALWAYS_INDIRECT=1' >> ~/.bashrc
source ~/.bashrc
```

#### Opción C: Ejecutar en Windows Nativo

Si tienes problemas con WSL:

```powershell
# En PowerShell o CMD de Windows
cd C:\path\to\f1_mars
python scripts\watch_agent.py --model trained_models\PPO_default_final.zip
```

---

### 3️⃣ Ejecutar el Visualizador

```bash
python scripts/watch_agent.py --model trained_models/PPO_default_final.zip
```

**Se abrirá una ventana con:**
- 🏎️ Coche F1 renderizado en GPU
- 📊 HUD estilo F1 TV
- 🎮 Controles interactivos
- ✨ Efectos visuales (humo, chispas)

---

## 🎮 Controles

| Tecla | Acción |
|-------|--------|
| **SPACE** | Pausar/Reanudar |
| **R** | Resetear episodio |
| **H** | Mostrar/Ocultar HUD |
| **D** | Mostrar/Ocultar Debug |
| **+/-** | Zoom in/out |
| **ESC** | Salir |

---

## 🔍 Troubleshooting

### Error: "No window is active"

**Solución:** Configurar display (ver paso 2)

### Error: "cannot connect to X server"

**VcXsrv (Windows 10):**
1. Asegúrate que VcXsrv esté ejecutándose
2. Verifica que "Disable access control" esté marcado
3. Reinicia XLaunch

**WSLg (Windows 11):**
```bash
echo $DISPLAY  # Debe mostrar :0
```

### Error: "AttributeError: 'Car' object has no attribute 'x'"

**Ya corregido en la última versión!** Ejecuta:
```bash
git pull  # o descarga la última versión
```

### La ventana se abre pero no se ve nada

Esto es normal en WSL sin display configurado. Ver paso 2.

---

## 📊 Verificación Final

Ejecuta todos los tests:

```bash
# Test 1: Sintaxis (sin display)
python test_arcade_syntax.py

# Test 2: Setup completo
python test_arcade_setup.py

# Test 3: Watch agent dry run
python test_watch_agent_dry_run.py
```

**Todos deben mostrar:** `✓ ALL TESTS PASSED`

---

## 🎨 Features Disponibles

✅ **Renderizado GPU** - OpenGL 3.3+ (60+ FPS)
✅ **Cámara Dinámica** - Smooth follow, zoom automático
✅ **F1 Sprite Detallado** - Alerones, ruedas, efectos
✅ **HUD F1 TV** - Velocidad, vueltas, neumáticos
✅ **Efectos Visuales** - Humo, chispas, trails
✅ **Circuito Optimizado** - Batch rendering GPU

---

## 📝 Archivos de Ayuda

- `WSL_DISPLAY_SETUP.md` - Guía detallada de configuración X11
- `MIGRATION_SUMMARY.md` - Resumen completo de la migración
- `ARCADE_MIGRATION.md` - Documentación técnica

---

## 🐛 Si Algo Falla

1. **Ejecuta los tests:**
   ```bash
   python test_watch_agent_dry_run.py
   ```

2. **Verifica el display:**
   ```bash
   echo $DISPLAY
   ```

3. **Revisa la documentación:**
   ```bash
   cat WSL_DISPLAY_SETUP.md
   ```

4. **Abre un issue:**
   Incluye la salida de los tests y el error completo.

---

## ✅ Checklist

- [ ] Tests pasados (`python test_watch_agent_dry_run.py`)
- [ ] Display configurado (`echo $DISPLAY` → `:0`)
- [ ] VcXsrv ejecutándose (si usas Windows 10)
- [ ] Modelo disponible (`trained_models/PPO_default_final.zip`)
- [ ] Ejecutar viewer: `python scripts/watch_agent.py --model ...`

---

**¡Disfruta del renderizado GPU!** 🏎️💨
