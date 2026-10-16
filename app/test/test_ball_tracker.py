import os
import sys
import unittest
from pathlib import Path
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from sports.ball_tracker import BallKalmanFilter, BallTracker


class TestBallTrackerPhase1(unittest.TestCase):
    def test_kalman_filter_linear_motion(self):
        """Verifica que el filtro de Kalman estime correctamente una trayectoria lineal con ruido."""
        kf = BallKalmanFilter(dt=1.0, damping=1.0)
        kf.initialize(100.0, 200.0)

        # Simular movimiento con velocidad constante (vx=10, vy=5)
        true_x, true_y = 100.0, 200.0
        for _ in range(5):
            kf.predict()
            true_x += 10.0
            true_y += 5.0
            kf.update(true_x, true_y, confidence=0.9)

        pos_x, pos_y = kf.get_position()
        vel_x, vel_y = kf.get_velocity()

        self.assertAlmostEqual(pos_x, 150.0, delta=2.0)
        self.assertAlmostEqual(pos_y, 225.0, delta=2.0)
        self.assertAlmostEqual(vel_x, 10.0, delta=1.5)
        self.assertAlmostEqual(vel_y, 5.0, delta=1.5)

    def test_mahalanobis_outlier_rejection(self):
        """Verifica que un salto brusco e irrazonable tenga alta distancia de Mahalanobis."""
        kf = BallKalmanFilter()
        kf.initialize(200.0, 300.0)
        kf.predict()
        kf.update(205.0, 305.0, confidence=0.85)

        kf.predict()
        # Punto cercano esperable
        d_close = kf.compute_mahalanobis_distance(210.0, 310.0, confidence=0.8)
        # Punto lejano (outlier/falso positivo a 300px)
        d_far = kf.compute_mahalanobis_distance(550.0, 650.0, confidence=0.8)

        self.assertLess(d_close, 3.5)
        self.assertGreater(d_far, 10.0)

    def test_multicandidate_selection(self):
        """
        Verifica que entre un candidato ruidoso (alta conf pero fuera de trayectoria)
        y el balón real (conf media pero perfectamente en trayectoria), elija el balón real.
        """
        tracker = BallTracker(mahalanobis_gate=4.0)

        # Frame 1: Inicialización
        f1 = {"pixel_x": 100.0, "pixel_y": 100.0, "confidence": 0.8, "bbox": [90, 90, 110, 110]}
        det1, _ = tracker.update([f1])
        self.assertEqual(det1["pixel_x"], 100.0)

        # Frame 2: Establecer dirección hacia (110, 105)
        f2 = {"pixel_x": 110.0, "pixel_y": 105.0, "confidence": 0.8, "bbox": [100, 95, 120, 115]}
        det2, _ = tracker.update([f2])
        self.assertEqual(det2["pixel_x"], 110.0)

        # Frame 3: Dos candidatos:
        # C1: Botín/rodilla falsa a (280, 290) con confianza 0.90 (salto imposible)
        # C2: Balón real en trayectoria continua a (120, 110) con confianza 0.55
        c_false = {"pixel_x": 280.0, "pixel_y": 290.0, "confidence": 0.90, "bbox": [270, 280, 290, 300]}
        c_real = {"pixel_x": 120.0, "pixel_y": 110.0, "confidence": 0.55, "bbox": [110, 100, 130, 120]}

        det3, _ = tracker.update([c_false, c_real])

        self.assertIsNotNone(det3)
        self.assertEqual(det3["pixel_x"], 120.0)
        self.assertEqual(det3["pixel_y"], 110.0)

    def test_occlusion_interpolation(self):
        """Verifica interpolación suave durante frames de oclusión con atenuación de confianza."""
        tracker = BallTracker(max_missing_frames=5)
        f1 = {"pixel_x": 100.0, "pixel_y": 100.0, "confidence": 0.8, "bbox": [90, 90, 110, 110]}
        f2 = {"pixel_x": 110.0, "pixel_y": 100.0, "confidence": 0.8, "bbox": [100, 90, 120, 110]}
        tracker.update(f1)
        tracker.update(f2)

        # Frame 3: Balón ocluido (sin detección)
        det_interp1, hist = tracker.update(None)
        self.assertIsNotNone(det_interp1)
        self.assertTrue(det_interp1.get("interpolated", False))
        # Debe avanzar cinemáticamente en x
        self.assertGreater(det_interp1["pixel_x"], 115.0)
        self.assertLess(det_interp1["confidence"], 0.8)

        # Tras exceder max_missing_frames, debe retornar None
        for _ in range(6):
            det_none, _ = tracker.update(None)
        self.assertIsNone(det_none)

    def test_backward_compatibility_single_dict(self):
        """Verifica que tracker.update() siga funcionando pasando un dict simple o None."""
        tracker = BallTracker()
        det = {"pixel_x": 50.0, "pixel_y": 50.0, "confidence": 0.9, "bbox": [40, 40, 60, 60]}
        out, hist = tracker.update(det)
        self.assertIsNotNone(out)
        self.assertEqual(out["pixel_x"], 50.0)
        self.assertEqual(len(hist), 1)


if __name__ == "__main__":
    unittest.main()
