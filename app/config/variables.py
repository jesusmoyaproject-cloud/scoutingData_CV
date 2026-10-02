"""Bridge de compatibilidad hacia settings.py (v5.0)."""
from config.settings import (
    APP_DIR,
    MODEL_PLAYER_PATH   as path_model_player,
    MODEL_BALL_PATH     as path_model_ball,
    MODEL_KEYPOINT_PATH as path_model_keypoint,
    DEFAULT_INPUT_VIDEO as path_video,
)

path_model_player    = str(path_model_player)
path_model_ball      = str(path_model_ball)
path_model_keypoint  = str(path_model_keypoint)
path_model_segmentation = ""
path_img = str(APP_DIR / "img" / "frame6.png")
