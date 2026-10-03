import os
import sys
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings
from main import SoccerAnalysisV5

def test_headless_tracking_json(tmp_path):
    output_json_path = str(tmp_path / "tracking_test_video.json")
    output_mp4_path = str(tmp_path / "test_video_output.mp4")

    # Si hay video de prueba en data/input.mp4
    video_input = settings.DEFAULT_INPUT_VIDEO
    if not Path(video_input).exists():
        print(f"⚠️ Video de prueba no disponible en {video_input}. Test ignorado.")
        return

    pipeline = SoccerAnalysisV5(headless=True)
    pipeline.process_video(
        video_path=video_input,
        output_path=output_mp4_path,
        json_output_path=output_json_path,
        max_frames=10,
        headless=True,
    )

    # 1. Comprobar que NO se generó MP4
    assert not Path(output_mp4_path).exists(), "En modo Headless NO debe generarse MP4."

    # 2. Comprobar que el JSON de tracking existe
    assert Path(output_json_path).exists(), "El archivo tracking.json debe ser generado."

    # 3. Validar Estructura del JSON según el contrato
    with open(output_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Metadata
    assert "metadata" in data
    meta = data["metadata"]
    assert "video_name" in meta
    assert "fps" in meta
    assert "total_frames" in meta
    assert "duration_seconds" in meta
    assert meta["total_frames"] == 10

    # Frames
    assert "frames" in data
    assert len(data["frames"]) == 10
    frame1 = data["frames"][0]
    assert "frame" in frame1
    assert frame1["frame"] == 1
    assert "timestamp" in frame1
    assert "players" in frame1
    assert "homography_valid" in frame1
    assert isinstance(frame1["homography_valid"], bool)

    if len(frame1["players"]) > 0:
        player = frame1["players"][0]
        assert "track_id" in player
        assert "pixel_x" in player
        assert "pixel_y" in player
        assert "pitch_x" in player
        assert "pitch_y" in player

    if frame1["ball"] is not None:
        ball = frame1["ball"]
        assert "pixel_x" in ball
        assert "pixel_y" in ball
        assert "pitch_x" in ball
        assert "pitch_y" in ball

    # Events
    assert "events" in data
    assert isinstance(data["events"], list)

    print("\n✅ Test de verificación de Headless Tracking JSON OK!")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_headless_tracking_json(Path(tmpdir))
