import os
import sys
import asyncio
import logging
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from main import SoccerAnalysisSOA
from config.variables import path_video, path_img

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TestSOA")

async def run_test():
    logger.info("=== ENTORNO DE PRUEBAS SOA (MICROSERVICIOS v2.0 + SoccerNet Events) ===")
    pipeline = SoccerAnalysisSOA()

    output_video = str(APP_DIR / "outputs" / "test_soa_preview.mp4")
    output_csv = str(APP_DIR / "outputs" / "detected_events.csv")

    if os.path.exists(path_video):
        logger.info(f"Ejecutando test SOA sobre fragmento de video: {path_video}")
        events = await pipeline.process_video_async(
            video_path=path_video,
            output_path=output_video,
            csv_output_path=output_csv,
            max_frames=1000
        )
        logger.info(f"[TEST SOA OK] Video generado en: {output_video}")
        logger.info(f"[TEST SOA OK] CSV de eventos exportado en: {output_csv} (Total eventos: {len(events)})")
    else:
        logger.warning(f"Video no encontrado en {path_video}. Probando frame estático: {path_img}")

if __name__ == "__main__":
    asyncio.run(run_test())

