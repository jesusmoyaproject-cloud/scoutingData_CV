"""
Test de Integración para BoTSORT, Detección de Color de Camiseta y Multi-rate Tracking (Fase 1 y 2).
"""

import sys
import json
import tempfile
from pathlib import Path
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings
from sports.team_detector import TeamDetector
from main import SoccerAnalysisV5


def test_team_detector_synthetic():
    """Valida la lógica de filtrado de césped, extracción CIELAB y clasificación de TeamDetector."""
    detector = TeamDetector(min_samples_to_cluster=4, history_window=10, lock_threshold=5)

    # Frame sintético con fondo de césped verde (BGR: [34, 139, 34])
    h, w = 720, 1280
    frame = np.full((h, w, 3), [34, 139, 34], dtype=np.uint8)

    # Jugador 1 (Equipo Rojo): Torso BGR [20, 20, 220]
    bbox1 = [200.0, 200.0, 260.0, 360.0]
    # Pintar torso rojo en el ROI [0.15H - 0.50H]
    ry1, ry2 = int(200 + 0.15 * 160), int(200 + 0.50 * 160)
    rx1, rx2 = int(200 + 0.15 * 60), int(260 - 0.15 * 60)
    frame[ry1:ry2, rx1:rx2] = [20, 20, 220]

    # Jugador 2 (Equipo Blanco): Torso BGR [240, 240, 240]
    bbox2 = [400.0, 200.0, 460.0, 360.0]
    ry1_b, ry2_b = int(200 + 0.15 * 160), int(200 + 0.50 * 160)
    rx1_b, rx2_b = int(400 + 0.15 * 60), int(460 - 0.15 * 60)
    frame[ry1_b:ry2_b, rx1_b:rx2_b] = [240, 240, 240]

    # 1. Probar extracción de color de camiseta
    ext1 = detector.extract_jersey_color(frame, bbox1)
    assert ext1 is not None, "Debe extraer color para bbox1"
    lab1, bgr1 = ext1
    assert len(lab1) == 3 and len(bgr1) == 3

    ext2 = detector.extract_jersey_color(frame, bbox2)
    assert ext2 is not None, "Debe extraer color para bbox2"
    lab2, bgr2 = ext2

    # 2. Entrenar clustering sintético agregando muestras
    for _ in range(5):
        detector.classify_player(lab1, bgr1, track_id=1, pitch_x_m=50.0, pitch_y_m=30.0)
        detector.classify_player(lab2, bgr2, track_id=2, pitch_x_m=55.0, pitch_y_m=35.0)

    assert detector._is_calibrated, "El detector debe calibrarse tras superar min_samples_to_cluster"
    assert detector.centroids_lab is not None
    assert len(detector.centroids_lab) == 2

    # 3. Probar locking temporal (Fase 2)
    for _ in range(8):
        res1 = detector.classify_player(lab1, bgr1, track_id=1, pitch_x_m=50.0, pitch_y_m=30.0)
    assert res1["is_locked"], "El track_id=1 debe quedar bloqueado tras observaciones consistentes"

    # 4. Probar rol contextual de Portero (cerca de portería)
    # Crear color outlier (Amarillo Fluor BGR [0, 255, 255]) cerca de x=5.0m (portería)
    lab_gk = np.array([200.0, 20.0, 180.0], dtype=np.float32)
    bgr_gk = (0, 255, 255)
    gk_res = detector.classify_player(lab_gk, bgr_gk, track_id=99, pitch_x_m=5.0, pitch_y_m=34.0)
    assert gk_res["role"] == "goalkeeper", f"Debe clasificarse como portero, obtuvo: {gk_res['role']}"

    # 5. Probar rol contextual de Árbitro (outlier en medio campo)
    ref_res = detector.classify_player(lab_gk, bgr_gk, track_id=100, pitch_x_m=52.0, pitch_y_m=34.0)
    assert ref_res["role"] == "referee", f"Debe clasificarse como árbitro, obtuvo: {ref_res['role']}"
    assert ref_res["team_id"] == -1

    print("✅ test_team_detector_synthetic pasó exitosamente.")


def test_pipeline_botsort_and_json():
    """Valida la ejecución del pipeline con BoTSORT, player_stride y estructura JSON v6.0."""
    video_input = settings.DEFAULT_INPUT_VIDEO
    if not Path(video_input).exists():
        print(f"⚠️ Video de prueba no encontrado en {video_input}, saltando test de video.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        output_json = tmp_path / "tracking_test_botsort.json"
        output_mp4 = tmp_path / "output_dummy.mp4"

        pipeline = SoccerAnalysisV5(headless=True, player_stride=2)
        pipeline.process_video(
            video_path=video_input,
            output_path=str(output_mp4),
            json_output_path=str(output_json),
            max_frames=6,
            headless=True,
        )

        assert output_json.exists(), "El archivo tracking JSON debe existir."

        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Validar metadata enriquecida
        assert "metadata" in data
        meta = data["metadata"]
        assert "teams" in meta, "Metadata debe incluir el diccionario de equipos"
        assert "0" in meta["teams"] and "1" in meta["teams"] and "-1" in meta["teams"]
        assert "tracking_config" in meta
        assert meta["tracking_config"]["player_stride"] == 2

        # 2. Validar frames y estructura de jugadores
        assert "frames" in data
        assert len(data["frames"]) == 6

        has_players = False
        has_interpolated = False
        for f in data["frames"]:
            assert "frame" in f
            assert "timestamp" in f
            assert "homography_valid" in f
            assert "players" in f

            for p in f["players"]:
                has_players = True
                # Claves existentes (Retrocompatibilidad)
                assert "track_id" in p
                assert "pixel_x" in p
                assert "pixel_y" in p
                assert "pitch_x" in p
                assert "pitch_y" in p

                # Claves nuevas Fases 1 y 2
                assert "player_id" in p
                assert "role" in p
                assert "team_id" in p
                assert "team_color" in p
                assert "team_confidence" in p
                assert "tracking_confidence" in p
                assert "is_interpolated" in p
                assert "bbox" in p

                if p["is_interpolated"]:
                    has_interpolated = True

        if has_players:
            print(f"✅ Se validaron {len(data['frames'])} frames con jugadores enriquecidos.")
            if has_interpolated:
                print("✅ Se comprobó con éxito la interpolación cinemática multi-rate (Fase 2).")

        print("✅ test_pipeline_botsort_and_json pasó exitosamente.")


if __name__ == "__main__":
    test_team_detector_synthetic()
    test_pipeline_botsort_and_json()
