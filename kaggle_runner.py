"""
ScoutingData v5.0 - Kaggle GPU Runner

Diferencias vs v4.0:
  - Usa DirectInferenceEngine (sin levantar microservicios)
  - Selecciona .pt + CUDA automáticamente en Kaggle T4
  - No lanza run_services.py (no es necesario en modo directo)
"""

import os
import sys
import time
import subprocess
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR  = BASE_DIR / "app"
sys.path.insert(0, str(APP_DIR))

import warnings
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Silenciar warnings y deprecaciones del logger interno de Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("KaggleRunner")

def run_pipeline(video_path: str, output_path: str, csv_path: str,
                 json_path: str = None, parquet_path: str = None,
                 mot_path: str = None, homography_json_path: str = None, max_frames: int = None,
                 homography_interval: int = 1, player_stride: int = 1,
                 headless: bool = False):
    # Auto-detectar entorno: COLAB o KAGGLE
    target_env = "COLAB" if settings.ENVIRONMENT == "COLAB" else "KAGGLE"

    logger.info("=" * 65)
    logger.info(f"⚽ ScoutingData v5.0 — Cloud GPU Runner ({target_env} / {settings.GPU_DEVICE_NAME or 'GPU'})")
    logger.info("=" * 65)
    for k, v in settings.get_summary().items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 65)

    # Validar existencia del video de entrada antes de invocar
    resolved_video = Path(video_path).resolve()
    if not resolved_video.exists():
        logger.error(f"❌ Error: El archivo de video no existe en '{video_path}'")
        logger.info("   Verifica la ruta con '!ls /content' o proporciona la ruta exacta con --video.")
        return

    main_script = APP_DIR / "main.py"
    cmd = [
        sys.executable, str(main_script),
        "--video",  str(resolved_video),
        "--output", output_path,
        "--csv",    csv_path,
        "--env",    target_env,
        "--homography-interval", str(homography_interval),
        "--player-stride", str(player_stride),
    ]
    if headless:
        cmd.append("--headless")
    if json_path:
        cmd.extend(["--json", json_path])
    if parquet_path:
        cmd.extend(["--parquet", parquet_path])
    if mot_path:
        cmd.extend(["--mot", mot_path])
    if homography_json_path:
        cmd.extend(["--homography-json", homography_json_path])
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])

    logger.info(f"🎬 Ejecutando: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            actual_json = json_path or str(settings.OUTPUT_DIR / f"tracking_{resolved_video.stem}.json")
            actual_parquet = parquet_path or str(settings.OUTPUT_DIR / f"tracking_{resolved_video.stem}.parquet")
            actual_homography = homography_json_path or str(settings.OUTPUT_DIR / f"homography_{resolved_video.stem}.json")
            logger.info("✅ Completado exitosamente.")
            if not headless:
                logger.info(f"   🎬 Video MP4:        {output_path}")
            logger.info(f"   📄 Tracking JSON:    {actual_json}")
            logger.info(f"   📐 Homografía JSON:  {actual_homography}")
            logger.info(f"   📊 Tracking Parquet: {actual_parquet}")
            logger.info(f"   📋 Eventos CSV:      {csv_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error: El pipeline finalizó con código de error {e.returncode}.")
        logger.error("   Revisa el log detallado superior para verificar si faltó algún modelo o librería.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle / Cloud GPU Runner — ScoutingData v5.0")
    parser.add_argument("--video",  "-v", default=settings.DEFAULT_INPUT_VIDEO)
    parser.add_argument("--output", "-o", default=settings.DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--csv",    "-c", default=settings.DEFAULT_OUTPUT_CSV)
    parser.add_argument("--json",   "-j", default=None)
    parser.add_argument("--parquet", "-p", default=None)
    parser.add_argument("--mot",          default=None)
    parser.add_argument("--homography-json", default=None, help="Ruta para exportar auditoría JSON de homografía")
    parser.add_argument("--max-frames", "-m", type=int, default=None)
    parser.add_argument("--headless", action="store_true", default=False, help="Modo headless (solo datos, sin video MP4)")
    parser.add_argument("--with-video", dest="headless", action="store_false", help="Generar video MP4 renderizado (activo por defecto)")
    parser.add_argument("--no-video", dest="headless", action="store_true", help="Omitir video MP4 para máxima velocidad")
    parser.add_argument("--homography-interval", type=int, default=1, help="Intervalo de cálculo de homografía")
    parser.add_argument("--player-stride", type=int, default=1, help="Cadencia de detección de jugadores (1 o 2)")
    args = parser.parse_args()
    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        csv_path=args.csv,
        json_path=args.json,
        parquet_path=args.parquet,
        mot_path=args.mot,
        homography_json_path=args.homography_json,
        max_frames=args.max_frames,
        homography_interval=args.homography_interval,
        player_stride=args.player_stride,
        headless=args.headless,
    )
