# ScoutingData v5.0 ⚽📊

Pipeline de Análisis Táctico, Detección e Inferencia Directa de Fútbol impulsado por **OpenVINO (CPU)**, **Ultralytics YOLO (CUDA/PyTorch)**, **Homografía Robusta (USAC_MAGSAC + Tracker Cinemático)** y **Arquitectura de Alto Rendimiento (Headless + Threaded I/O)**.

Diseñado para despliegue tanto en **Estaciones Locales de Trabajo** (Windows/Linux con CPU Intel) como en **Entornos Cloud/GPU (Kaggle Tesla T4)**.

---

## 🏗️ Arquitectura del Sistema (v5.0 Direct Inference Pipeline)

En v5.0 se reemplazó la arquitectura de microservicios HTTP por un **Pipeline de Inferencia Directa Monolítico de Alto Rendimiento**:

- **DirectInferenceEngine**: Carga los modelos YOLO (Keypoint, Jugadores/Árbitros, Balón) en memoria compartida (OpenVINO IR en CPU local / `.pt` nativo en GPU CUDA).
- **TemporalHomographyTracker (v5.1)**: Estimación robusta de homografía $3D \to 2D$ con USAC_MAGSAC, filtrado cinemático de deriva de centro/esquinas y análisis de residuales.
- **BallTracker**: Seguimiento cinemático y descarte de saltos anómalos del balón.
- **AsyncVideoWriter (OPT-8)**: Escritura de video MP4 en hilo de fondo con buffer desacoplado.
- **ThreadedOpenCVReader (OPT-9)**: Lectura y prefetching de frames de video en segundo plano con profiling integrado de lectura y decodificación (`read_and_decode`).
- **Modo Headless**: En entornos como Kaggle o analítica batch, omite el renderizado de video y overlays para generar exclusivamente el contrato de datos JSON (`tracking_<video>.json`).

```
                    ┌──────────────────────────────────────────────┐
                    │            Video Source (.mp4)               │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      ThreadedOpenCVReader (OPT-9)            │
                    │      Prefetching en Background + Profiling   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │         DirectInferenceEngine (v5.0)         │
                    │  (Keypoint Detect + Player Detect + Ball)    │
                    └──────────────────────┬───────────────────────┘
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 ▼                                                   ▼
┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
│ TemporalHomographyTracker       │                 │ BallTracker & ByteTrack         │
│ (USAC_MAGSAC + Rejection Audit) │                 │ (Player & Ball Trajectories)    │
└────────────────┬────────────────┘                 └────────────────┬────────────────┘
                 │                                                   │
                 └─────────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  Headless: tracking_<video>.json Contract    │
                    │  Full: AsyncVideoWriter (OPT-8) + Overlays   │
                    └──────────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto v5.0

```
scoutingData_ProcessingEventing/
│
├── app/
│   ├── config/
│   │   ├── settings.py           # Configuración centralizada por entorno (LOCAL / KAGGLE)
│   │   ├── field_dimensions.py   # Dimensiones oficiales FIFA
│   │   └── field_points.py       # Puntos de referencia 2D/3D de la cancha
│   ├── engine/
│   │   ├── direct_inference.py   # Motor de inferencia directa (CUDA / OpenVINO)
│   │   ├── async_writer.py       # AsyncVideoWriter en background (OPT-8)
│   │   └── threaded_reader.py    # ThreadedOpenCVReader + profiling de read_and_decode (OPT-9)
│   ├── homography/               # Homografía, tracking temporal y validaciones geométricas
│   ├── model/                    # Modelos OpenVINO IR y PyTorch .pt
│   ├── sports/                   # BallTracker cinemático
│   ├── test/                     # Tests unitarios e integración (Headless, Audit, Threaded)
│   ├── visualization/            # Overlays, template de cancha y minimap 2D
│   └── main.py                   # Orquestador principal y CLI
│
├── openVino/                     # Modelos OpenVINO IR (.xml / .bin)
├── data/                         # Videos de prueba locales
├── outputs/                      # Salidas generadas (JSON tracking, MP4, CSV rejections)
├── docs/                         # Documentación técnica (HEADLESS_TRACKING.md)
├── kaggle_runner.py              # Runner optimizado para ejecución autónoma en Kaggle
└── requirements.txt              # Dependencias optimizadas del proyecto
```

---

## 🚀 Instalación y Ejecución

### 1. Instalación de Dependencias
```bash
git clone https://github.com/tu-usuario/projectScoutingData.git
cd projectScoutingData/scoutingData_ProcessingEventing
pip install -r requirements.txt
```

---

### 2. Ejecución Local (Windows / Linux)

```bash
# Modo completo (Video con overlays + JSON de tracking)
python app/main.py --video data/input.mp4 --output outputs/salida.mp4 --json outputs/tracking.json

# Modo Headless de alto rendimiento (solo tracking JSON, sin renderizar video)
python app/main.py --video data/input.mp4 --headless --json outputs/tracking.json
```

---

### 3. Ejecución en Google Colab (GPU NVIDIA L4 / T4)

En una celda de Google Colab:
```bash
# 1. Entrar a la carpeta del repositorio clonado
%cd /content/scoutingData_CV

# 2. Instalar dependencias
!pip install -r requirements.txt

# 3. Ejecutar con aceleración GPU L4 (Tensor Cores TF32 + FP16)
!python colab_runner.py --video /content/garVScc_liga1max_2026_apertura_pt.mp4
```

---

### 4. Ejecución en Kaggle (GPU Tesla T4)

```bash
python kaggle_runner.py --video /kaggle/input/soccer-match/input.mp4 --output /kaggle/working/outputs/salida.mp4 --json /kaggle/working/outputs/tracking.json
```

---

## ⚙️ Opciones de Línea de Comandos (`main.py`)

| Argumento | Opción Corta | Descripción | Valor por Defecto |
| :--- | :--- | :--- | :--- |
| `--video` | `-v` | Ruta al video de entrada | `settings.DEFAULT_INPUT_VIDEO` |
| `--output` | `-o` | Ruta del video procesado (.mp4) | `outputs/scouting_output.mp4` |
| `--json` | `-j` | Ruta del JSON de tracking estructurado | Auto (`outputs/tracking_<name>.json`) |
| `--csv` | `-c` | Ruta del archivo CSV de eventos | `outputs/events_output.csv` |
| `--headless` | | Ejecutar en modo Headless (sin video/overlays) | Auto según entorno |
| `--max-frames` | `-m` | Limita la cantidad de frames a procesar | `None` (todo el video) |
| `--homography-interval` | | Recalcular H cada N frames | `1` |
| `--debug-homography` | | Logs forenses detallados de homografía | `False` |
| `--disable-threaded-reader` | | Desactivar ThreadedOpenCVReader (usa lectura secuencial) | `False` |
| `--reader-buffer` | | Tamaño del buffer de frames prefetcheados | `64` |

---

## 🧪 Ejecución de Tests

```bash
python -m unittest discover -s app/test
```
