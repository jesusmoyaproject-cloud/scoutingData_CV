"""
Test de integración con video sintético para Headless Tracking JSON.
"""

import os
import sys
import json
import tempfile
import cv2
import numpy as np
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings
from main import SoccerAnalysisV5

def create_synthetic_video(file_path: str, num_frames: int = 5, width: int = 1280, height: int = 720, fps: int = 30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Crear frame sintético verde tipo cancha
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34) # Verde césped
        # Dibujar algunas líneas de prueba
        cv2.line(frame, (100, 100), (1180, 100), (255, 255, 255), 3)
        cv2.line(frame, (100, 620), (1180, 620), (255, 255, 255), 3)
        cv2.line(frame, (640, 100), (640, 620), (255, 255, 255), 3)
        cv2.circle(frame, (640, 360), 80, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()

def test_headless_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "synthetic_match.mp4")
        output_mp4  = os.path.join(tmpdir, "output_rendered.mp4")
        output_json = os.path.join(tmpdir, "tracking_synthetic_match.json")

        create_synthetic_video(input_video, num_frames=5)
        print(f"🎬 Video sintético creado: {input_video}")

        pipeline = SoccerAnalysisV5(headless=True)
        pipeline.process_video(
            video_path=input_video,
            output_path=output_mp4,
            json_output_path=output_json,
            headless=True,
        )

        # 1. Verificar que en headless NO existe MP4 de salida
        assert not os.path.exists(output_mp4), "❌ ERROR: MP4 fue generado en modo Headless."
        print("✅ Verificado: No se generó MP4 en modo Headless.")

        # 2. Verificar que existe JSON de tracking
        assert os.path.exists(output_json), "❌ ERROR: No se generó tracking JSON."
        print(f"✅ Verificado: JSON generado en {output_json}")

        # 3. Validar esquema JSON del contrato
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        meta = data["metadata"]
        assert meta["video_name"] == "synthetic_match.mp4"
        assert meta["total_frames"] == 5
        assert meta["fps"] == 30.0
        assert meta["duration_seconds"] == round(5 / 30.0, 2)

        assert "frames" in data
        assert len(data["frames"]) == 5

        f1 = data["frames"][0]
        assert f1["frame"] == 1
        assert "timestamp" in f1
        assert "players" in f1
        assert "ball" in f1
        assert "homography_valid" in f1

        assert "events" in data
        assert data["events"] == []

        print("\n🎉 ¡TODAS LAS VERIFICACIONES DEL CONTRATO Y MODO HEADLESS PASARON EXITOSAMENTE!")

if __name__ == "__main__":
    test_headless_pipeline()
