"""
ScoutingData v4.0 - Kaggle GPU Runner Script

Ejecuta el pipeline completo SOA + OpenVINO dentro de notebooks o scripts en Kaggle.
Levanta los microservicios FastAPI en background, verifica su estado de salud,
procesa el video especificado y exporta los resultados a /kaggle/working/outputs/
"""

import os
import sys
import time
import subprocess
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"
sys.path.insert(0, str(APP_DIR))

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("KaggleRunner")

def run_kaggle_pipeline(video_path: str, output_path: str, csv_path: str, max_frames: int = None):
    logger.info("==========================================================================")
    logger.info("⚽ INICIANDO PIPELINE SCOUTINGDATA v4.0 EN KAGGLE GPU (Tesla T4)")
    logger.info("==========================================================================")
    
    summary = settings.get_summary()
    for key, val in summary.items():
        logger.info(f"  - {key}: {val}")
    logger.info("==========================================================================")

    # 1. Iniciar Microservicios en Background
    logger.info("🚀 Lanzando microservicios SOA en background...")
    services_script = APP_DIR / "run_services.py"
    
    svc_proc = subprocess.Popen([sys.executable, str(services_script)], cwd=str(APP_DIR))
    
    try:
        # 2. Dar tiempo para que levanten y verificar health checks
        time.sleep(20)
        
        # 3. Ejecutar main.py
        main_script = APP_DIR / "main.py"
        cmd = [
            sys.executable, str(main_script),
            "--video", video_path,
            "--output", output_path,
            "--csv", csv_path,
            "--env", "KAGGLE"
        ]
        if max_frames:
            cmd.extend(["--max-frames", str(max_frames)])

        logger.info(f"🎬 Ejecutando main.py con comando: {' '.join(cmd)}")
        main_res = subprocess.run(cmd, check=True)
        
        if main_res.returncode == 0:
            logger.info("✅ Procesamiento completado exitosamente en Kaggle.")
            logger.info(f"📹 Video procesado: {output_path}")
            logger.info(f"📊 CSV de eventos: {csv_path}")
        else:
            logger.error(f"❌ Error en la ejecución de main.py (código: {main_res.returncode})")

    except Exception as e:
        logger.error(f"❌ Error durante la ejecución en Kaggle: {e}")
    finally:
        logger.info("🧹 Finalizando microservicios SOA en background...")
        svc_proc.terminate()
        try:
            svc_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            svc_proc.kill()
        logger.info("✅ Microservicios terminados limpios.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner para Kaggle GPU - ScoutingData v4.0")
    parser.add_argument("--video", "-v", type=str, default=settings.DEFAULT_INPUT_VIDEO, help="Ruta al video en /kaggle/input/...")
    parser.add_argument("--output", "-o", type=str, default=settings.DEFAULT_OUTPUT_VIDEO, help="Ruta al video de salida")
    parser.add_argument("--csv", "-c", type=str, default=settings.DEFAULT_OUTPUT_CSV, help="Ruta al CSV de salida")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Límite opcional de frames")

    args = parser.parse_args()
    
    run_kaggle_pipeline(
        video_path=args.video,
        output_path=args.output,
        csv_path=args.csv,
        max_frames=args.max_frames
    )
