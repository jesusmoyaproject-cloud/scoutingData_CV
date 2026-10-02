import os
import sys
import logging
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from main_monolithic import SoccerAnalysisPipeline
from config.variables import path_model_keypoint, path_model_player, path_model_ball, path_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TestMonolithic")

def run_test():
    logger.info("=== ENTORNO DE PRUEBAS MONOLÍTICO (OPENVINO LOCAL v2.0) ===")

    pipeline = SoccerAnalysisPipeline(
        model_keypoint_path=path_model_keypoint,
        model_player_path=path_model_player,
        model_ball_path=path_model_ball,
        conf_keypoint=0.5,
        conf_player=0.3,
        conf_ball=0.2
    )

    output_video = str(APP_DIR / "outputs" / "test_monolithic_preview.mp4")
    output_txt = str(APP_DIR / "outputs" / "test_monolithic_results.txt")

    if os.path.exists(path_video):
        logger.info(f"Ejecutando test monolítico sobre fragmento de video (60 frames): {path_video}")
        pipeline.process_video(
            video_path=path_video,
            output_video_path=output_video,
            output_txt_path=output_txt,
            max_frames=60
        )
        logger.info(f"[TEST MONOLITHIC OK] Video generado en: {output_video}")

if __name__ == "__main__":
    run_test()
