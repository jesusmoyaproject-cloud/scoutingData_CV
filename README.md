# ScoutingData v4.0 ⚽📊

Pipeline de Análisis Táctico y Tracking de Fútbol impulsado por **OpenVINO**, **Ultralytics YOLO**, **SoccerNet Event Detection** y una **Arquitectura Orientada a Servicios (SOA)** con **FastAPI**.

Designed for seamless deployment on both **Local Workstations** (Windows/Linux) and **Kaggle GPU Environments (Tesla T4)**.

---

## 🏗️ Arquitectura del Sistema

El sistema utiliza un pipeline distribuido asíncrono con 4 microservicios FastAPI:

1. **Keypoint Service (`:8001`)**: Detección de puntos clave de calibración del terreno de juego para cálculo de homografía ($3D \to 2D$).
2. **Player Service (`:8002`)**: Detección y seguimiento multilocal (*ByteTrack*) de jugadores y árbitros.
3. **Ball Service (`:8003`)**: Detección y filtrado espacial del balón.
4. **Event Service (`:8004`)**: Clasificación temporal de eventos futbolísticos (Pases, Remates, Faltas, Tarjetas, Goles) con atribución de emisor y receptor del pase.

```
                  ┌────────────────────────────────────────┐
                  │          app/main.py Orquestador       │
                  └──────────────────┬─────────────────────┘
                                     │ Async HTTP Keep-Alive
         ┌──────────────────┬────────┴─────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐┌──────────────────┐
│ Keypoint Service ││  Player Service  ││   Ball Service   ││  Event Service   │
│   (Port 8001)    ││   (Port 8002)    ││   (Port 8003)    ││   (Port 8004)    │
└──────────────────┘└──────────────────┘└──────────────────┘└──────────────────┘
```

---

## 📁 Estructura del Proyecto v4.0

```
projectScoutingData/
│
├── app/
│   ├── config/
│   │   ├── settings.py           # Configuración por Entorno (LOCAL / KAGGLE)
│   │   ├── field_dimensions.py   # Dimensiones oficiales de cancha FIFA
│   │   ├── field_points.py       # Puntos de referencia 2D/3D
│   │   └── variables.py          # Bridge para código legado
│   ├── homography/               # Homografía e Inversa (pixel_to_field)
│   ├── model/                    # Enlace a modelos OpenVINO
│   ├── outputs/                  # Videos y CSVs de salida
│   ├── services/                 # Microservicios FastAPI y Cliente Asíncrono
│   ├── sports/                   # Ball tracker y annotators
│   ├── visualization/            # Overlays, Pitch Template y 2D Minimap
│   ├── main.py                   # Orquestador Principal CLI
│   └── run_services.py           # Lanzador portátil de Microservicios
│
├── openVino/                     # Modelos OpenVINO IR (.xml / .bin)
│   ├── modelPlayer_openvino_model/
│   ├── modelBall_openvino_model/
│   └── modelKeypoint_openvino_model/
│
├── outputs/                      # Carpeta principal de salidas
├── data/                         # Videos de entrada locales
├── requirements.txt              # Dependencias del proyecto
├── kaggle_runner.py              # Script de ejecución autónoma en Kaggle
└── README.md                     # Documentación oficial
```

---

## 🚀 Guía de Instalación y Ejecución

### 1. Instalación de Dependencias
```bash
git clone https://github.com/tu-usuario/projectScoutingData.git
cd projectScoutingData
pip install -r requirements.txt
```

---

### 2. Ejecución Local (Windows / Linux)

#### Paso 1: Iniciar los Microservicios SOA
En una terminal:
```bash
python app/run_services.py
```
*(Esperar a que los 4 health checks confirmen `:8001`, `:8002`, `:8003`, `:8004` OK)*

#### Paso 2: Ejecutar el Pipeline de Análisis
En otra terminal:
```bash
python app/main.py --video data/mi_partido.mp4 --output outputs/salida.mp4 --csv outputs/eventos.csv
```

---

### 3. Ejecución en Kaggle (GPU Tesla T4)

En una celda de Notebook de Kaggle o Terminal:

```python
# 1. Clonar el repositorio
!git clone https://github.com/tu-usuario/projectScoutingData.git
%cd projectScoutingData

# 2. Instalar requerimientos
!pip install -r requirements.txt -q

# 3. Ejecutar el Kaggle Runner Script
!python kaggle_runner.py --video /kaggle/input/mi-dataset-futbol/partido.mp4 --output /kaggle/working/outputs/resultado.mp4
```

---

## ⚙️ Opciones de Línea de Comandos (`main.py`)

| Argumento | Opción Corta | Descripción | Valor por Defecto |
| :--- | :--- | :--- | :--- |
| `--video` | `-v` | Ruta al video de entrada | `settings.DEFAULT_INPUT_VIDEO` |
| `--output` | `-o` | Ruta del video procesado (.mp4) | `outputs/scouting_output.mp4` |
| `--csv` | `-c` | Ruta del archivo CSV de eventos | `outputs/events_output.csv` |
| `--env` | `-e` | Fuerza el entorno (`LOCAL` o `KAGGLE`) | Detección Automática |
| `--max-frames` | `-m` | Limita la cantidad de frames procesados | `None` (Procesa todo el video) |

---

## 🛠️ Variables de Entorno Soporta

- `ENVIRONMENT`: `"LOCAL"` o `"KAGGLE"`
- `MODEL_DIR`: Ruta personalizada a los modelos OpenVINO.
- `OPENVINO_DEVICE`: `"CPU"`, `"GPU"`, o `"AUTO"`.
- `KEYPOINT_SERVICE_URL`, `PLAYER_SERVICE_URL`, `BALL_SERVICE_URL`, `EVENT_SERVICE_URL`: URLs para endpoints personalizados.
