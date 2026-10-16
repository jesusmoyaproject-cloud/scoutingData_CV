"""
Test unitario para exportación ML / Deep Learning (Parquet y MOTChallenge).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from sports.ml_exporter import export_tracking_to_parquet, export_to_mot_challenge


class TestMLExporter(unittest.TestCase):

    def setUp(self):
        # Dataset sintético de tracking con 3 frames
        self.sample_dataset = {
            "metadata": {
                "video_name": "match_test.mp4",
                "fps": 25.0,
                "total_frames": 3,
            },
            "frames": [
                {
                    "frame": 1,
                    "timestamp": 0.04,
                    "homography_valid": True,
                    "ball": {
                        "detected": True,
                        "confidence": 0.95,
                        "pixel_x": 640.0,
                        "pixel_y": 360.0,
                        "pitch_x": 52.5,
                        "pitch_y": 34.0,
                    },
                    "players": [
                        {
                            "track_id": 10,
                            "role": "player",
                            "team_id": 0,
                            "team_color": "#FF0000",
                            "tracking_confidence": 0.92,
                            "bbox": [100.0, 150.0, 140.0, 250.0],
                            "pixel_x": 120.0,
                            "pixel_y": 250.0,
                            "pitch_x": 50.0,
                            "pitch_y": 30.0,
                            "is_interpolated": False,
                        },
                        {
                            "track_id": 1,
                            "role": "goalkeeper",
                            "team_id": 1,
                            "team_color": "#00FF00",
                            "tracking_confidence": 0.88,
                            "bbox": [20.0, 200.0, 50.0, 300.0],
                            "pixel_x": 35.0,
                            "pixel_y": 300.0,
                            "pitch_x": 5.0,
                            "pitch_y": 34.0,
                            "is_interpolated": False,
                        },
                    ],
                },
                {
                    "frame": 2,
                    "timestamp": 0.08,
                    "homography_valid": True,
                    "ball": {
                        "detected": True,
                        "confidence": 0.94,
                        "pixel_x": 650.0,
                        "pixel_y": 360.0,
                        "pitch_x": 53.0,
                        "pitch_y": 34.0,
                    },
                    "players": [
                        {
                            "track_id": 10,
                            "role": "player",
                            "team_id": 0,
                            "team_color": "#FF0000",
                            "tracking_confidence": 0.93,
                            "bbox": [105.0, 150.0, 145.0, 250.0],
                            "pixel_x": 125.0,
                            "pixel_y": 250.0,
                            "pitch_x": 50.2,  # Se movió 0.2m en 0.04s => 5.0 m/s = 18.0 km/h
                            "pitch_y": 30.0,
                            "is_interpolated": False,
                        },
                    ],
                },
            ],
        }

    def test_export_to_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_parquet = os.path.join(tmpdir, "tracking_test.parquet")
            res = export_tracking_to_parquet(self.sample_dataset, out_parquet)
            self.assertIsNotNone(res)
            self.assertTrue(os.path.exists(out_parquet))

            # Leer Parquet y verificar integridad de columnas
            df = pd.read_parquet(out_parquet)
            self.assertEqual(len(df), 3)  # 2 jugadores frame 1 + 1 jugador frame 2

            expected_cols = [
                "frame_idx", "timestamp_sec", "track_id", "role", "team_id",
                "pitch_x_m", "pitch_y_m", "velocity_x_ms", "speed_kmh",
                "dist_to_ball_m", "ball_detected", "homography_valid"
            ]
            for col in expected_cols:
                self.assertIn(col, df.columns)

            # Verificar cálculo cinemático en frame 2 para track_id 10
            row_f2_p10 = df[(df["frame_idx"] == 2) & (df["track_id"] == 10)].iloc[0]
            self.assertAlmostEqual(row_f2_p10["velocity_x_ms"], 5.0, places=1)
            self.assertAlmostEqual(row_f2_p10["speed_kmh"], 18.0, places=1)

            # Distancia al balón en frame 1 para track_id 10: (50.0, 30.0) a (52.5, 34.0) = sqrt(2.5^2 + 4^2) = 4.72m
            row_f1_p10 = df[(df["frame_idx"] == 1) & (df["track_id"] == 10)].iloc[0]
            self.assertAlmostEqual(row_f1_p10["dist_to_ball_m"], 4.72, places=1)

    def test_export_to_mot_challenge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_mot = os.path.join(tmpdir, "tracking_test.mot")
            res = export_to_mot_challenge(self.sample_dataset, out_mot)
            self.assertIsNotNone(res)
            self.assertTrue(os.path.exists(out_mot))

            with open(out_mot, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 3)
            # Primera línea: frame 1, track 10
            parts = lines[0].strip().split(",")
            self.assertEqual(parts[0], "1")
            self.assertEqual(parts[1], "10")


if __name__ == "__main__":
    unittest.main()
