"""
ScoutingData v5.0 - Configuración Centralizada por Entorno

Novedades v5.0:
  - Sección 0: selección automática de modelo (.pt CUDA vs OpenVINO CPU)
  - Sin event_service (SoccerNet eliminado)
  - MODEL_FORMAT, INFERENCE_DEVICE calculados por entorno
  - Rutas a modelos .pt originales (modeloOriginal/)
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any

CONFIG_DIR = Path(__file__).resolve().parent
APP_DIR    = CONFIG_DIR.parent
BASE_DIR   = APP_DIR.parent

# ─── Detección de Entorno ────────────────────────────────────────────────────
def detect_environment() -> str:
    forced = os.getenv("ENVIRONMENT") or os.getenv("APP_ENV")
    if forced:
        return forced.upper()
    if (os.path.exists("/kaggle/working")
            or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
            or "KAGGLE_CONTAINER_NAME" in os.environ):
        return "KAGGLE"
    return "LOCAL"

ENVIRONMENT = detect_environment()

def detect_headless_mode() -> bool:
    env_val = os.getenv("HEADLESS_MODE")
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    return ENVIRONMENT == "KAGGLE"

HEADLESS_MODE = detect_headless_mode()

# ─── Sección 0: Formato e Dispositivo de Inferencia ─────────────────────────
#
#  LOCAL  → OpenVINO IR (.xml/.bin)  en CPU Intel  → 2-4x vs .pt CPU
#  KAGGLE → PyTorch .pt              en CUDA T4    → 6-10x vs OpenVINO CPU
#
CUDA_AVAILABLE = torch.cuda.is_available()

if ENVIRONMENT == "KAGGLE" and CUDA_AVAILABLE:
    MODEL_FORMAT    = "pt"
    INFERENCE_DEVICE = "cuda"
else:
    MODEL_FORMAT    = "openvino"
    INFERENCE_DEVICE = "cpu"

# Permite override manual vía variable de entorno
MODEL_FORMAT     = os.getenv("MODEL_FORMAT",     MODEL_FORMAT)
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", INFERENCE_DEVICE)

# ─── Directorios de Modelos ──────────────────────────────────────────────────
def _resolve_model_dir(openvino_subdir: str = "openVino") -> Path:
    env_val = os.getenv("MODEL_DIR")
    if env_val and Path(env_val).exists():
        return Path(env_val)
    root_ov = BASE_DIR / openvino_subdir
    if root_ov.exists():
        return root_ov
    return APP_DIR / "model" / "openVino"

MODEL_DIR_OV = _resolve_model_dir()                        # OpenVINO IR
MODEL_DIR_PT = APP_DIR / "model" / "modeloOriginal"        # .pt originales

# ─── Rutas de Modelos: OpenVINO ──────────────────────────────────────────────
MODEL_PLAYER_PATH   = Path(os.getenv("MODEL_PLAYER_PATH",
                        str(MODEL_DIR_OV / "modelPlayer_openvino_model")))
MODEL_BALL_PATH     = Path(os.getenv("MODEL_BALL_PATH",
                        str(MODEL_DIR_OV / "modelBall_openvino_model")))
MODEL_KEYPOINT_PATH = Path(os.getenv("MODEL_KEYPOINT_PATH",
                        str(MODEL_DIR_OV / "modelKeypoint_openvino_model")))

# ─── Rutas de Modelos: .pt (CUDA / Kaggle) ───────────────────────────────────
MODEL_PLAYER_PT_PATH   = Path(os.getenv("MODEL_PLAYER_PT_PATH",
                            str(MODEL_DIR_PT / "modelPlayer.pt")))
MODEL_BALL_PT_PATH     = Path(os.getenv("MODEL_BALL_PT_PATH",
                            str(MODEL_DIR_PT / "modelBall.pt")))
MODEL_KEYPOINT_PT_PATH = Path(os.getenv("MODEL_KEYPOINT_PT_PATH",
                            str(MODEL_DIR_PT / "modelKeypoint.pt")))

# ─── Salidas ─────────────────────────────────────────────────────────────────
def _resolve_output_dir() -> Path:
    if ENVIRONMENT == "KAGGLE":
        p = Path("/kaggle/working/outputs")
    else:
        p = BASE_DIR / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p

OUTPUT_DIR = _resolve_output_dir()

def _resolve_default_input() -> str:
    env_video = os.getenv("VIDEO_INPUT")
    if env_video:
        return env_video
    if ENVIRONMENT == "KAGGLE":
        c = Path("/kaggle/input/soccer-match-video/input.mp4")
        return str(c) if c.exists() else "/kaggle/input/soccer-video/input.mp4"
    return str(BASE_DIR / "data" / "input.mp4")

DEFAULT_INPUT_VIDEO  = _resolve_default_input()
DEFAULT_OUTPUT_VIDEO = str(OUTPUT_DIR / "scouting_output.mp4")
DEFAULT_OUTPUT_CSV   = str(OUTPUT_DIR / "events_output.csv")
DEFAULT_OUTPUT_JSON  = str(OUTPUT_DIR / f"tracking_{Path(DEFAULT_INPUT_VIDEO).stem}.json")

# ─── Microservicios SOA (modo legacy) ────────────────────────────────────────
KEYPOINT_SERVICE_URL = os.getenv("KEYPOINT_SERVICE_URL", "http://127.0.0.1:8001")
PLAYER_SERVICE_URL   = os.getenv("PLAYER_SERVICE_URL",   "http://127.0.0.1:8002")
BALL_SERVICE_URL     = os.getenv("BALL_SERVICE_URL",     "http://127.0.0.1:8003")

# ─── Umbrales y Parámetros ───────────────────────────────────────────────────
CONF_KEYPOINT       = float(os.getenv("CONF_KEYPOINT",       "0.25"))
CONF_PLAYER         = float(os.getenv("CONF_PLAYER",         "0.35"))
CONF_BALL           = float(os.getenv("CONF_BALL",           "0.15"))
RANSAC_THRESHOLD    = float(os.getenv("RANSAC_THRESHOLD",    "5.0"))
MINIMAP_SCALE       = float(os.getenv("MINIMAP_SCALE",       "8.0"))
HOMOGRAPHY_INTERVAL = int(os.getenv("HOMOGRAPHY_INTERVAL",   "1"))   # 1=cada frame, N>1=cada N frames

# ─── Utilidades ──────────────────────────────────────────────────────────────
def resolve_video_source(source_input: str) -> Path:
    """Abstracción de origen de video: path local, Kaggle o URL futura."""
    if source_input.startswith(("http://", "https://")):
        return OUTPUT_DIR / "downloaded_input.mp4"
    return Path(source_input).resolve()

def get_summary() -> Dict[str, Any]:
    return {
        "ENVIRONMENT":      ENVIRONMENT,
        "HEADLESS_MODE":    HEADLESS_MODE,
        "MODEL_FORMAT":     MODEL_FORMAT,
        "INFERENCE_DEVICE": INFERENCE_DEVICE,
        "CUDA_AVAILABLE":   CUDA_AVAILABLE,
        "BASE_DIR":         str(BASE_DIR),
        "MODEL_DIR_OV":     str(MODEL_DIR_OV),
        "MODEL_DIR_PT":     str(MODEL_DIR_PT),
        "OUTPUT_DIR":       str(OUTPUT_DIR),
        "DEFAULT_INPUT":    DEFAULT_INPUT_VIDEO,
        "HOMOGRAPHY_INTERVAL": HOMOGRAPHY_INTERVAL,
    }
