"""
test_pipeline.py — Script de prueba rápida del pipeline ScoutingData v5.0

Ejecuta los primeros N frames generando video de salida para revisión visual.
Los resultados se guardan en outputs/test/ para no pisar los outputs de producción.

Uso (ejecutar desde la raíz del proyecto):
    python app/test/test_pipeline.py                         # 1000 frames + video de revisión
    python app/test/test_pipeline.py --video "D:/ruta/partido2.mp4"
    python app/test/test_pipeline.py --frames 300            # menos frames, más rápido
    python app/test/test_pipeline.py --headless              # sin video (solo JSON, más veloz)
    python app/test/test_pipeline.py --no-debug              # sin logs de homografía
    python app/test/test_pipeline.py --interval 5            # recalcular H cada 5 frames
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# ── app/ al path para imports directos ───────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent.parent  # app/test/ → app/
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings
from main import SoccerAnalysisV5, _enable_debug_mode

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_pipeline")

# ── Output separado de producción ────────────────────────────────────────────
TEST_OUTPUT_DIR = Path(settings.OUTPUT_DIR) / "test"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_test(video_path: str, max_frames: int, debug: bool, interval: int, headless: bool):
    logger.info("=" * 65)
    logger.info("  🧪 TEST PIPELINE — ScoutingData v5.0")
    logger.info("=" * 65)
    logger.info(f"  Video:        {video_path}")
    logger.info(f"  Max frames:   {max_frames}")
    logger.info(f"  Debug homog:  {debug}")
    logger.info(f"  H interval:   cada {interval} frame/s")
    logger.info(f"  Modo:         {'HEADLESS (solo JSON)' if headless else 'VIDEO (con overlay)'}")
    logger.info(f"  Output dir:   {TEST_OUTPUT_DIR}")
    logger.info("=" * 65)

    if debug:
        _enable_debug_mode()

    video_stem = Path(video_path).stem
    json_out   = str(TEST_OUTPUT_DIR / f"test_{video_stem}_{max_frames}f.json")
    video_out  = str(TEST_OUTPUT_DIR / f"test_{video_stem}_{max_frames}f.mp4")

    pipeline = SoccerAnalysisV5(
        homography_interval=interval,
        headless=headless,
    )

    t_start = time.time()
    pipeline.process_video(
        video_path=video_path,
        output_path=video_out,
        csv_output_path=None,
        json_output_path=json_out,
        max_frames=max_frames,
        headless=headless,
        debug_homography=debug,
    )
    elapsed = time.time() - t_start

    logger.info("=" * 65)
    logger.info(f"  ✅ Test completado en {elapsed:.1f}s")
    logger.info(f"  📄 JSON:  {json_out}")
    if not headless:
        logger.info(f"  🎬 Video: {video_out}")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test rápido del pipeline ScoutingData (headless, N frames)"
    )
    parser.add_argument(
        "--video", "-v",
        default=settings.DEFAULT_INPUT_VIDEO,
        help=f"Video a procesar (default: {settings.DEFAULT_INPUT_VIDEO})"
    )
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=1000,
        help="Número máximo de frames (default: 1000)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Modo headless: solo JSON, sin generar video (más rápido)"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        default=False,
        help="Desactivar logs de debug de homografía"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=1,
        help="Recalcular H cada N frames (default: 1)"
    )
    args = parser.parse_args()

    run_test(
        video_path=args.video,
        max_frames=args.frames,
        debug=not args.no_debug,
        interval=args.interval,
        headless=args.headless,
    )
