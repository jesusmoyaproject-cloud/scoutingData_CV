"""
ScoutingData v5.0 - Configuración Centralizada por Entorno

Novedades v5.0:
  - Sección 0: selección automática de modelo (.pt CUDA vs OpenVINO CPU)
  - Sin event_service (SoccerNet eliminado)
  - MODEL_FORMAT, INFERENCE_DEVICE calculados por entorno
  - Rutas a modelos .pt originales (modeloOriginal/)
"""

import os
import sys
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
    # Kaggle
    if (os.path.exists("/kaggle/working")
            or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
            or "KAGGLE_CONTAINER_NAME" in os.environ):
        return "KAGGLE"
    # Google Colab
    if (os.path.exists("/content")
            or "COLAB_GPU" in os.environ
            or "COLAB_RELEASE_TAG" in os.environ
            or "google.colab" in sys.modules):
        return "COLAB"
    return "LOCAL"

ENVIRONMENT = detect_environment()

def detect_headless_mode() -> bool:
    env_val = os.getenv("HEADLESS_MODE")
    if env_val is not None:
        return env_val.lower() in ("true", "1", "yes")
    return ENVIRONMENT in ("KAGGLE", "COLAB")

HEADLESS_MODE = detect_headless_mode()

# ─── Sección 0: Formato e Dispositivo de Inferencia ─────────────────────────
#
#  LOCAL  → OpenVINO IR (.xml/.bin)  en CPU Intel  → 2-4x vs .pt CPU
#  KAGGLE → PyTorch .pt              en CUDA T4    → 6-10x vs OpenVINO CPU
#  COLAB  → PyTorch .pt              en CUDA L4/T4 → Tensor Cores TF32 + FP16
#
CUDA_AVAILABLE = torch.cuda.is_available()

# Telemetría de GPU
GPU_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else ""
GPU_IS_L4 = "L4" in GPU_DEVICE_NAME.upper() if CUDA_AVAILABLE else False
GPU_VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if CUDA_AVAILABLE else 0.0

if CUDA_AVAILABLE and (ENVIRONMENT in ("KAGGLE", "COLAB") or os.getenv("FORCE_CUDA", "").lower() in ("1", "true")):
    MODEL_FORMAT     = "pt"
    INFERENCE_DEVICE = "cuda"
elif ENVIRONMENT == "LOCAL" and not os.getenv("MODEL_FORMAT"):
    MODEL_FORMAT     = "openvino"
    INFERENCE_DEVICE = "cpu"
else:
    MODEL_FORMAT     = "pt" if CUDA_AVAILABLE else "openvino"
    INFERENCE_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# Permite override manual vía variable de entorno
MODEL_FORMAT     = os.getenv("MODEL_FORMAT",     MODEL_FORMAT)
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", INFERENCE_DEVICE)

# ─── Optimizaciones de GPU para L4 / Ampere / Ada Lovelace ───────────────────
def configure_gpu_optimizations() -> Dict[str, Any]:
    """
    Configura optimizaciones de hardware para aceleración en GPU CUDA.
    Para NVIDIA L4 (Ada Lovelace): activa TF32 en Tensor Cores y cuDNN benchmark.
    """
    global CUDA_AVAILABLE, GPU_DEVICE_NAME, GPU_IS_L4, GPU_VRAM_GB
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE and not GPU_DEVICE_NAME:
        try:
            GPU_DEVICE_NAME = torch.cuda.get_device_name(0)
            GPU_IS_L4 = "L4" in GPU_DEVICE_NAME.upper()
            GPU_VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except Exception:
            pass

    info = {
        "cuda_available": CUDA_AVAILABLE,
        "device_name": GPU_DEVICE_NAME,
        "is_l4": GPU_IS_L4,
        "vram_gb": GPU_VRAM_GB,
        "tf32_enabled": False,
        "cudnn_benchmark": False,
    }
    if CUDA_AVAILABLE and INFERENCE_DEVICE == "cuda":
        try:
            # Habilitar TensorFloat-32 (TF32) para Tensor Cores en Ada Lovelace (L4) y Ampere
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            info["tf32_enabled"] = True
        except Exception:
            pass

        try:
            # cuDNN benchmark optimiza la selección de algoritmos de convolución para frames fijos
            torch.backends.cudnn.benchmark = True
            info["cudnn_benchmark"] = True
        except Exception:
            pass

    return info

GPU_OPTIMIZATIONS = configure_gpu_optimizations()

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

def _resolve_model_pt(filename: str, env_var: str) -> Path:
    env_val = os.getenv(env_var)
    if env_val and Path(env_val).exists():
        return Path(env_val)
    candidates = [
        MODEL_DIR_PT / filename,
        BASE_DIR / "modelos" / filename,
        BASE_DIR / filename,
        Path("/content") / filename,
        Path("/kaggle/working") / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

# ─── Rutas de Modelos: OpenVINO ──────────────────────────────────────────────
MODEL_PLAYER_PATH   = Path(os.getenv("MODEL_PLAYER_PATH",
                        str(MODEL_DIR_OV / "modelPlayer_openvino_model")))
MODEL_BALL_PATH     = Path(os.getenv("MODEL_BALL_PATH",
                        str(MODEL_DIR_OV / "modelBall_openvino_model")))
MODEL_KEYPOINT_PATH = Path(os.getenv("MODEL_KEYPOINT_PATH",
                        str(MODEL_DIR_OV / "modelKeypoint_openvino_model")))

# ─── Rutas de Modelos: .pt (CUDA / Colab / Kaggle) ───────────────────────────
MODEL_PLAYER_PT_PATH   = _resolve_model_pt("modelPlayer.pt",   "MODEL_PLAYER_PT_PATH")
MODEL_BALL_PT_PATH     = _resolve_model_pt("modelBall.pt",     "MODEL_BALL_PT_PATH")
MODEL_KEYPOINT_PT_PATH = _resolve_model_pt("modelKeypoint.pt", "MODEL_KEYPOINT_PT_PATH")

# ─── Salidas ─────────────────────────────────────────────────────────────────
def _resolve_output_dir() -> Path:
    env_out = os.getenv("OUTPUT_DIR")
    if env_out:
        p = Path(env_out)
    elif ENVIRONMENT == "KAGGLE":
        p = Path("/kaggle/working/outputs")
    elif ENVIRONMENT == "COLAB":
        p = Path("/content/outputs") if Path("/content").exists() else BASE_DIR / "outputs"
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
    if ENVIRONMENT == "COLAB":
        content_path = Path("/content")
        if content_path.exists():
            mp4s = [f for f in sorted(content_path.glob("*.mp4")) if "output" not in f.name.lower()]
            if mp4s:
                return str(mp4s[0])
    return str(BASE_DIR / "data" / "input.mp4")

DEFAULT_INPUT_VIDEO  = _resolve_default_input()
DEFAULT_OUTPUT_VIDEO = str(OUTPUT_DIR / "scouting_output.mp4")
DEFAULT_OUTPUT_CSV   = str(OUTPUT_DIR / "events_output.csv")
DEFAULT_OUTPUT_JSON  = str(OUTPUT_DIR / f"tracking_{Path(DEFAULT_INPUT_VIDEO).stem}.json")
DEFAULT_KEYPOINTS_CSV = str(OUTPUT_DIR / f"keypoints_audit_{Path(DEFAULT_INPUT_VIDEO).stem}.csv")
# ─── Umbrales y Parámetros ───────────────────────────────────────────────────

CONF_KEYPOINT       = float(os.getenv("CONF_KEYPOINT",       "0.35"))
CONF_PLAYER         = float(os.getenv("CONF_PLAYER",         "0.4"))
CONF_BALL           = float(os.getenv("CONF_BALL",           "0.4"))
RANSAC_THRESHOLD    = float(os.getenv("RANSAC_THRESHOLD",    "1.5"))
MINIMAP_SCALE       = float(os.getenv("MINIMAP_SCALE",       "8.0"))
HOMOGRAPHY_INTERVAL = int(os.getenv("HOMOGRAPHY_INTERVAL",   "1"))   # 1=cada frame, N>1=cada N frames
DRAW_OUTLIERS       = os.getenv("DRAW_OUTLIERS", "false").lower() in ("true", "1", "yes")
ENABLE_CORNER_SHIFT = os.getenv("ENABLE_CORNER_SHIFT", "true").lower() in ("true", "1", "yes")

# ─── Configuración de Tracking y Equipos (BoTSORT / Team Classifier) ─────────
TRACKER_CONFIG_PATH     = CONFIG_DIR / "botsort_sports.yaml"
TRACKER_TYPE            = os.getenv("TRACKER_TYPE", "botsort")
PLAYER_INFERENCE_STRIDE = int(os.getenv("PLAYER_INFERENCE_STRIDE", "1"))  # 1=todos los frames, 2=skip adaptativo
PLAYER_NMS_IOU          = float(os.getenv("PLAYER_NMS_IOU", "0.45"))      # NMS IoU para evitar doble track en mismo jugador
ENABLE_TEAM_CLASSIFIER  = os.getenv("ENABLE_TEAM_CLASSIFIER", "true").lower() in ("true", "1", "yes")
SHOW_FRAME_NUMBER       = os.getenv("SHOW_FRAME_NUMBER", "true" if ENVIRONMENT == "LOCAL" else "false").lower() in ("true", "1", "yes")



# ─── Utilidades ──────────────────────────────────────────────────────────────
def resolve_video_source(source_input: str) -> Path:
    """Abstracción de origen de video: path local, Kaggle o URL futura."""
    if source_input.startswith(("http://", "https://")):
        return OUTPUT_DIR / "downloaded_input.mp4"
    return Path(source_input).resolve()

def get_summary() -> Dict[str, Any]:
    return {
        "ENVIRONMENT":         ENVIRONMENT,
        "HEADLESS_MODE":       HEADLESS_MODE,
        "MODEL_FORMAT":        MODEL_FORMAT,
        "INFERENCE_DEVICE":    INFERENCE_DEVICE,
        "CUDA_AVAILABLE":      CUDA_AVAILABLE,
        "GPU_DEVICE_NAME":     GPU_DEVICE_NAME or "N/A",
        "GPU_IS_L4":           GPU_IS_L4,
        "GPU_VRAM_GB":         f"{GPU_VRAM_GB} GB" if CUDA_AVAILABLE else "N/A",
        "TF32_ENABLED":        GPU_OPTIMIZATIONS.get("tf32_enabled", False),
        "CUDNN_BENCHMARK":     GPU_OPTIMIZATIONS.get("cudnn_benchmark", False),
        "BASE_DIR":            str(BASE_DIR),
        "MODEL_DIR_OV":        str(MODEL_DIR_OV),
        "MODEL_DIR_PT":        str(MODEL_DIR_PT),
        "OUTPUT_DIR":          str(OUTPUT_DIR),
        "DEFAULT_INPUT":       DEFAULT_INPUT_VIDEO,
        "HOMOGRAPHY_INTERVAL": HOMOGRAPHY_INTERVAL,
    }

