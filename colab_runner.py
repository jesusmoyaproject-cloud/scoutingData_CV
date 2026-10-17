"""
ScoutingData v5.0 - Google Colab GPU Runner (NVIDIA L4 / Tesla T4)

Optimizado específicamente para ejecución en Google Colab:
  - Detección automática del acelerador (NVIDIA L4 24GB, Tesla T4 16GB, etc.)
  - Activación de TensorFloat-32 (TF32) en Tensor Cores (L4 Ada Lovelace)
  - Inferencia FP16 (Half Precision) ultrarrápida
  - Búsqueda automática de video en /content/ si no se especifica
  - Ejecución directa en proceso o CLI para ver progreso y logs en tiempo real
"""

import os
import sys
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR  = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Asegurar entorno COLAB
os.environ["ENVIRONMENT"] = "COLAB"

import warnings
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Silenciar warnings y deprecaciones del logger interno de Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import torch
from config import settings
from main import SoccerAnalysisV5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ColabRunner")


def print_colab_diagnostic():
    """Muestra información detallada del hardware en Colab."""
    logger.info("=" * 68)
    logger.info("⚡ ScoutingData v5.0 — Google Colab GPU Runner")
    logger.info("=" * 68)

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        is_l4 = "L4" in device_name.upper()
        logger.info(f"  🎮 GPU Detectada:       {device_name} ({vram_gb} GB VRAM)")
        logger.info(f"  🚀 Arquitectura L4:     {'SÍ (Ada Lovelace - 4th Gen Tensor Cores)' if is_l4 else 'No'}")
        logger.info(f"  🔥 TensorFloat-32:      {settings.GPU_OPTIMIZATIONS.get('tf32_enabled', False)}")
        logger.info(f"  ⚡ Precisión:           FP16 (Half Precision)")
        logger.info(f"  📦 Formato Modelo:      PyTorch .pt")
    else:
        logger.warning("  ⚠ ALERTA: No se detectó GPU CUDA activa en esta sesión de Colab.")
        logger.warning("  👉 Para activar la GPU NVIDIA L4:")
        logger.warning("     1. Ve al menú superior: 'Entorno de ejecución' (Runtime)")
        logger.warning("     2. Selecciona 'Cambiar tipo de entorno de ejecución' (Change runtime type)")
        logger.warning("     3. Elige 'GPU L4' (o 'T4') y presiona 'Guardar'.")

    logger.info(f"  📁 Directorio Salidas:  {settings.OUTPUT_DIR}")
    logger.info("=" * 68)


def verify_models() -> bool:
    """Verifica que los modelos .pt existan en las rutas configuradas."""
    models = [
        ("Keypoint", settings.MODEL_KEYPOINT_PT_PATH),
        ("Player",   settings.MODEL_PLAYER_PT_PATH),
        ("Ball",     settings.MODEL_BALL_PT_PATH),
    ]
    all_ok = True
    for label, path in models:
        p = Path(path)
        if p.exists():
            size_mb = round(p.stat().st_size / (1024 * 1024), 1)
            logger.info(f"  ✓ Modelo {label:8s} : {p.name} ({size_mb} MB)")
        else:
            logger.error(f"  ✗ Modelo {label:8s} NO ENCONTRADO en: {p}")
            all_ok = False

    if not all_ok:
        logger.error("❌ Faltan archivos de modelos .pt.")
        logger.info("   Verifica que la carpeta 'app/model/modeloOriginal/' contenga los 3 archivos .pt.")
    return all_ok


def run_pipeline(
    video_path: str,
    output_path: str = None,
    csv_path: str = None,
    json_path: str = None,
    parquet_path: str = None,
    mot_path: str = None,
    max_frames: int = None,
    headless: bool = False,
    homography_interval: int = 1,
    player_stride: int = settings.PLAYER_INFERENCE_STRIDE,
    debug_homography: bool = False,
    homography_json_path: str = None,
):
    print_colab_diagnostic()

    if not verify_models():
        return False

    resolved_video = Path(video_path).resolve()
    if not resolved_video.exists():
        logger.error(f"❌ Video no encontrado en: '{video_path}'")
        logger.info("   Sube un archivo .mp4 a Google Colab (ej. en /content/) e indica su ruta con --video.")
        return False

    out_dir = Path(settings.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_output  = output_path or str(out_dir / "scouting_output.mp4")
    actual_csv     = csv_path or str(out_dir / "events_output.csv")
    actual_json    = json_path or str(out_dir / f"tracking_{resolved_video.stem}.json")
    actual_parquet = parquet_path or str(out_dir / f"tracking_{resolved_video.stem}.parquet")
    actual_homography_json = homography_json_path or str(out_dir / f"homography_{resolved_video.stem}.json")

    logger.info(f"🎬 Procesando: {resolved_video.name}")
    logger.info(f"   Salida Video:      {actual_output if not headless else '[HEADLESS - Omitido]'}")
    logger.info(f"   Salida JSON:       {actual_json}")
    logger.info(f"   Salida Homografía: {actual_homography_json}")
    logger.info(f"   Salida Parquet:    {actual_parquet}")
    logger.info(f"   Salida CSV:        {actual_csv}")

    pipeline = SoccerAnalysisV5(
        homography_interval=homography_interval,
        player_stride=player_stride,
        headless=headless,
        show_frame_number=True,
    )

    pipeline.process_video(
        video_path=str(resolved_video),
        output_path=actual_output,
        csv_output_path=actual_csv,
        json_output_path=actual_json,
        parquet_output_path=actual_parquet,
        mot_output_path=mot_path,
        max_frames=max_frames,
        headless=headless,
        debug_homography=debug_homography,
        homography_json_path=actual_homography_json,
    )

    logger.info("=" * 68)
    logger.info(f"🎉 Proceso completado exitosamente en Google Colab.")
    if not headless:
        logger.info(f"   🎬 Video MP4:        {actual_output}")
    logger.info(f"   📄 Tracking JSON:    {actual_json}")
    logger.info(f"   📐 Homografía JSON:  {actual_homography_json}")
    logger.info(f"   📊 Tracking Parquet: {actual_parquet}")
    logger.info(f"   📋 Eventos CSV:      {actual_csv}")
    logger.info("=" * 68)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Colab L4 GPU Runner — ScoutingData v5.0")
    parser.add_argument("--video",  "-v", default=settings.DEFAULT_INPUT_VIDEO, help="Ruta del video (ej: /content/partido.mp4)")
    parser.add_argument("--output", "-o", default=None, help="Ruta de video de salida MP4 (default: outputs/scouting_output.mp4)")
    parser.add_argument("--csv",    "-c", default=None, help="Ruta del CSV de eventos")
    parser.add_argument("--json",   "-j", default=None, help="Ruta del JSON de tracking")
    parser.add_argument("--parquet", "-p", default=None, help="Ruta del Parquet de tracking (ML/DL)")
    parser.add_argument("--mot",          default=None, help="Ruta del archivo MOTChallenge (.mot)")
    parser.add_argument("--homography-json", default=None, help="Ruta para exportar auditoría JSON de homografía")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Límite de frames a procesar")
    parser.add_argument("--headless", action="store_true", default=False, help="Modo headless (solo datos, sin renderizar video MP4)")
    parser.add_argument("--with-video", dest="headless", action="store_false", help="Forzar generación de video MP4 renderizado (activo por defecto)")
    parser.add_argument("--no-video", dest="headless", action="store_true", help="Omitir generación de video MP4 para máxima velocidad")
    parser.add_argument("--homography-interval", type=int, default=1, help="Intervalo de cálculo de homografía (1=cada frame, 2=cada 2 frames)")
    parser.add_argument("--player-stride", type=int, default=settings.PLAYER_INFERENCE_STRIDE, help="Stride de inferencia de jugadores (1=todos, 2=interpolar alternados)")
    parser.add_argument("--debug", action="store_true", default=False, help="Modo debug de homografía")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        csv_path=args.csv,
        json_path=args.json,
        parquet_path=args.parquet,
        mot_path=args.mot,
        max_frames=args.max_frames,
        headless=args.headless,
        homography_interval=args.homography_interval,
        player_stride=args.player_stride,
        debug_homography=args.debug,
        homography_json_path=args.homography_json,
    )
