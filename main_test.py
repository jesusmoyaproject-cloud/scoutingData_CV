"""
main_test.py — Runner de Pruebas Rápidas en CPU / Local (1000 frames)

Diseñado para pruebas locales y auditoría forense rápida de homografía y tracking:
  - Procesa por defecto 1000 frames (configurable con --frames)
  - Ejecuta con OpenVINO en CPU (o CUDA si está forzado)
  - Genera video renderizado MP4 con minimapa y overlays para inspección visual
  - Exporta todos los archivos de auditoría forense:
      * CSV de auditoría de Keypoints punto a punto (inliers, outliers, residuales)
      * CSV de rechazos cinemáticos de homografía (deriva, rotación, escala)
      * JSON completo de homografía y tracking
      * CSV de eventos del partido
  - Todo se guarda en outputs/test/ para no pisar producción

Uso:
    python main_test.py                         # 1000 frames con video y auditoría completa
    python main_test.py --frames 1500           # 1500 frames
    python main_test.py --video "data/input.mp4" --frames 500
    python main_test.py --headless              # Sin video, máxima velocidad
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR  = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Asegurar entorno LOCAL para OpenVINO en CPU
if "ENVIRONMENT" not in os.environ:
    os.environ["ENVIRONMENT"] = "LOCAL"

import warnings
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Silenciar logger de ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from config import settings
from main import SoccerAnalysisV5, _enable_debug_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MainTest")

TEST_OUTPUT_DIR = Path(settings.OUTPUT_DIR) / "test"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_main_test(
    video_path: str,
    max_frames: int = 1000,
    headless: bool = False,
    debug: bool = True,
    interval: int = 1,
    player_stride: int = 1,
):
    resolved = settings.resolve_video_source(video_path)
    if not resolved.exists():
        logger.error(f"❌ Video no encontrado en: {resolved}")
        logger.info("   Coloca el video en data/input.mp4 o especifícalo con --video <ruta>")
        return False

    if debug:
        _enable_debug_mode()

    video_stem = resolved.stem
    json_out        = str(TEST_OUTPUT_DIR / f"test_{video_stem}_{max_frames}f.json")
    video_out       = str(TEST_OUTPUT_DIR / f"test_{video_stem}_{max_frames}f.mp4")
    events_csv      = str(TEST_OUTPUT_DIR / f"events_test_{video_stem}_{max_frames}f.csv")
    rejections_csv  = str(TEST_OUTPUT_DIR / f"rejections_test_{video_stem}_{max_frames}f.csv")
    keypoints_csv   = str(TEST_OUTPUT_DIR / f"keypoints_audit_test_{video_stem}_{max_frames}f.csv")
    homography_json = str(TEST_OUTPUT_DIR / f"homography_test_{video_stem}_{max_frames}f.json")

    logger.info("=" * 68)
    logger.info("  🧪 MAIN TEST — ScoutingData v5.0 (CPU / Local)")
    logger.info("=" * 68)
    logger.info(f"  📹 Video Entrada:       {resolved.name}")
    logger.info(f"  🎞️  Frames Máximos:      {max_frames}")
    logger.info(f"  ⚙️  Modo Inferencia:     {settings.INFERENCE_DEVICE.upper()} ({settings.MODEL_FORMAT.upper()})")
    logger.info(f"  📐 Intervalo H:         cada {interval} frame/s")
    logger.info(f"  🏃 Player Stride:       {player_stride}")
    logger.info(f"  📺 Salida Video:        {video_out if not headless else '[HEADLESS - Omitido]'}")
    logger.info(f"  📊 CSV Auditoría KPs:   {keypoints_csv}")
    logger.info(f"  📉 CSV Rechazos H:      {rejections_csv}")
    logger.info(f"  📐 JSON Homografía:     {homography_json}")
    logger.info(f"  📁 Directorio Test:     {TEST_OUTPUT_DIR}")
    logger.info("=" * 68)

    pipeline = SoccerAnalysisV5(
        homography_interval=interval,
        player_stride=player_stride,
        headless=headless,
    )

    t_start = time.time()
    pipeline.process_video(
        video_path=str(resolved),
        output_path=video_out,
        csv_output_path=events_csv,
        json_output_path=json_out,
        max_frames=max_frames,
        headless=headless,
        debug_homography=debug,
        rejections_csv_path=rejections_csv,
        keypoints_csv_path=keypoints_csv,
        homography_json_path=homography_json,
    )
    elapsed = time.time() - t_start

    logger.info("=" * 68)
    logger.info(f"  ✅ Test completado exitosamente en {elapsed:.1f}s")
    logger.info(f"  📄 Tracking JSON:       {json_out}")
    logger.info(f"  📊 CSV Auditoría KPs:   {keypoints_csv}")
    logger.info(f"  📐 JSON Homografía:     {homography_json}")
    logger.info(f"  📉 CSV Rechazos H:      {rejections_csv}")
    logger.info(f"  📋 CSV Eventos:         {events_csv}")
    if not headless:
        logger.info(f"  🎬 Video con Overlays:  {video_out}")
    logger.info("=" * 68)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main Test — ScoutingData v5.0 (1000 frames CPU con exportación de CSV de homografía)"
    )
    parser.add_argument("--video", "-v", default=settings.DEFAULT_INPUT_VIDEO, help="Video de entrada a procesar")
    parser.add_argument("--frames", "-f", type=int, default=1000, help="Número de frames a procesar (default: 1000)")
    parser.add_argument("--headless", action="store_true", default=False, help="Modo headless (sin generar video MP4)")
    parser.add_argument("--no-debug", action="store_true", default=False, help="Desactivar logs de debug de homografía")
    parser.add_argument("--interval", "-i", type=int, default=1, help="Intervalo de cálculo de homografía (default: 1)")
    parser.add_argument("--player-stride", type=int, default=1, help="Stride de inferencia de jugadores (1 o 2)")
    args = parser.parse_args()

    run_main_test(
        video_path=args.video,
        max_frames=args.frames,
        headless=args.headless,
        debug=not args.no_debug,
        interval=args.interval,
        player_stride=args.player_stride,
    )
