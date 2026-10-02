"""
ScoutingData v4.0 - Configuración Centralizada por Entorno (LOCAL / KAGGLE)

Permite gestionar rutas dinámicas con pathlib.Path, puertos de microservicios SOA,
dispositivo OpenVINO (CPU/GPU) y parámetros por defecto sin hardcoding.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Jerarquía de Directorios:
# projectScoutingData/
#   ├── app/
#   │   └── config/
#   │       └── settings.py
#   └── openVino/

CONFIG_DIR = Path(__file__).resolve().parent
APP_DIR = CONFIG_DIR.parent
BASE_DIR = APP_DIR.parent

# Detectar Entorno de Ejecución (LOCAL vs KAGGLE)
def detect_environment() -> str:
    forced_env = os.getenv("ENVIRONMENT") or os.getenv("APP_ENV")
    if forced_env:
        return forced_env.upper()
    
    # Auto-detección en entorno Kaggle
    if os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_CONTAINER_NAME" in os.environ:
        return "KAGGLE"
    
    return "LOCAL"

ENVIRONMENT = detect_environment()

# Resolución del Directorio de Modelos OpenVINO (Prioridad: ENV > BASE_DIR/openVino > APP_DIR/model/openVino)
def resolve_model_dir() -> Path:
    env_model_dir = os.getenv("MODEL_DIR")
    if env_model_dir and Path(env_model_dir).exists():
        return Path(env_model_dir)
    
    root_openvino = BASE_DIR / "openVino"
    if root_openvino.exists():
        return root_openvino
    
    app_openvino = APP_DIR / "model" / "openVino"
    return app_openvino

MODEL_DIR = resolve_model_dir()

# Rutas de Modelos OpenVINO
MODEL_PLAYER_PATH = Path(os.getenv("MODEL_PLAYER_PATH", MODEL_DIR / "modelPlayer_openvino_model"))
MODEL_BALL_PATH = Path(os.getenv("MODEL_BALL_PATH", MODEL_DIR / "modelBall_openvino_model"))
MODEL_KEYPOINT_PATH = Path(os.getenv("MODEL_KEYPOINT_PATH", MODEL_DIR / "modelKeypoint_openvino_model"))

# Resolución del Directorio de Salidas
def resolve_output_dir() -> Path:
    if ENVIRONMENT == "KAGGLE":
        out_path = Path("/kaggle/working/outputs")
    else:
        out_path = BASE_DIR / "outputs"
    
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path

OUTPUT_DIR = resolve_output_dir()

# Valores por defecto para entradas y salidas
def resolve_default_input_video() -> str:
    env_video = os.getenv("VIDEO_INPUT")
    if env_video:
        return env_video
    
    if ENVIRONMENT == "KAGGLE":
        kaggle_candidate = Path("/kaggle/input/soccer-match-video/input.mp4")
        if kaggle_candidate.exists():
            return str(kaggle_candidate)
        return "/kaggle/input/soccer-video/garVScc_tercergol.mp4"
    else:
        sample_video = BASE_DIR / "data" / "input.mp4"
        return str(sample_video)

DEFAULT_INPUT_VIDEO = resolve_default_input_video()
DEFAULT_OUTPUT_VIDEO = str(OUTPUT_DIR / "scouting_output.mp4")
DEFAULT_OUTPUT_CSV = str(OUTPUT_DIR / "events_output.csv")

# Endpoints de Microservicios SOA
KEYPOINT_SERVICE_URL = os.getenv("KEYPOINT_SERVICE_URL", "http://127.0.0.1:8001")
PLAYER_SERVICE_URL = os.getenv("PLAYER_SERVICE_URL", "http://127.0.0.1:8002")
BALL_SERVICE_URL = os.getenv("BALL_SERVICE_URL", "http://127.0.0.1:8003")
EVENT_SERVICE_URL = os.getenv("EVENT_SERVICE_URL", "http://127.0.0.1:8004")

# Parámetros de Ejecución y Umbrales de Confianza
CONF_KEYPOINT = float(os.getenv("CONF_KEYPOINT", "0.25"))
CONF_PLAYER = float(os.getenv("CONF_PLAYER", "0.35"))
CONF_BALL = float(os.getenv("CONF_BALL", "0.15"))
CONF_EVENT = float(os.getenv("CONF_EVENT", "0.45"))
RANSAC_THRESHOLD = float(os.getenv("RANSAC_THRESHOLD", "5.0"))
MINIMAP_SCALE = float(os.getenv("MINIMAP_SCALE", "8.0"))

# Dispositivo de Inferencia OpenVINO (CPU / GPU / AUTO)
OPENVINO_DEVICE = os.getenv("OPENVINO_DEVICE", "GPU" if ENVIRONMENT == "KAGGLE" else "CPU")

def resolve_video_source(source_input: str) -> Path:
    """
    Abstracción para la resolución de orígenes de video (Local, Kaggle, URL Remota).
    Prepara la arquitectura para futuras integraciones con Frontend / API Backend.
    """
    if source_input.startswith(("http://", "https://")):
        temp_download = OUTPUT_DIR / "downloaded_input.mp4"
        return temp_download
    
    path_obj = Path(source_input).resolve()
    return path_obj

def get_summary() -> Dict[str, Any]:
    return {
        "ENVIRONMENT": ENVIRONMENT,
        "BASE_DIR": str(BASE_DIR),
        "MODEL_DIR": str(MODEL_DIR),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "DEFAULT_INPUT_VIDEO": DEFAULT_INPUT_VIDEO,
        "DEFAULT_OUTPUT_VIDEO": DEFAULT_OUTPUT_VIDEO,
        "OPENVINO_DEVICE": OPENVINO_DEVICE,
        "SERVICE_URLS": {
            "keypoint": KEYPOINT_SERVICE_URL,
            "player": PLAYER_SERVICE_URL,
            "ball": BALL_SERVICE_URL,
            "event": EVENT_SERVICE_URL
        }
    }
