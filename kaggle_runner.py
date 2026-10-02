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

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("KaggleRunner")

def run_pipeline(video_path: str, output_path: str, csv_path: str,
                 max_frames: int = None):
    logger.info("=" * 65)
    logger.info("⚽ ScoutingData v5.0 — Kaggle GPU Runner (Direct Inference)")
    logger.info("=" * 65)
    for k, v in settings.get_summary().items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 65)

    main_script = APP_DIR / "main.py"
    cmd = [
        sys.executable, str(main_script),
        "--video",  video_path,
        "--output", output_path,
        "--csv",    csv_path,
        "--env",    "KAGGLE",
    ]
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])

    logger.info(f"🎬 Ejecutando: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            logger.info(f"✅ Completado. Video: {output_path} | CSV: {csv_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle GPU Runner — ScoutingData v5.0")
    parser.add_argument("--video",  "-v", default=settings.DEFAULT_INPUT_VIDEO)
    parser.add_argument("--output", "-o", default=settings.DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--csv",    "-c", default=settings.DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-frames", "-m", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(args.video, args.output, args.csv, args.max_frames)
