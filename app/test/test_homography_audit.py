"""
test_homography_audit.py - Suite de pruebas exhaustiva para la auditoría de homografía.
Verifica:
1. Métricas de diagnóstico (inliers, outliers, reproyección, convex hull, cobertura, colinealidad).
2. Validación geométrica (rechazo por colinealidad, área de hull insuficiente, dispersión).
3. Filtrado de puntos pegados al borde de imagen.
4. Robustez de USAC_MAGSAC con umbral calibrado (1.5m).
5. Validación temporal y tracking (continuidad de traslación/escala/rotación, holding y auto-reset).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from homography.metrics import HomographyDiagnostics, compute_reprojection_errors
from homography.geometric_validator import (
    compute_convex_hull_and_coverage,
    compute_collinearity_score,
    validate_keypoint_geometry,
    filter_border_keypoints,
)
from homography.temporal_validator import (
    decompose_homography_transition,
    TemporalHomographyTracker,
    normalize_homography,
)
from homography.validation import validate_homography_matrix, is_homography_transition_valid
from config.field_points import FIELD_COORDS
from config.field_dimensions import FieldDimensions
from homography.diagnostics_tool import (
    compute_point_residuals,
    generate_residuals_table,
    run_leave_one_out_analysis,
    detect_spatial_anomalies_and_swaps,
    format_leave_one_out_report,
)
from visualization.overlays import draw_reprojection_residuals


class TestHomographyAudit(unittest.TestCase):

    def setUp(self):
        # Generar un terreno canónico y una homografía sintética realista de cámara broadcast
        self.img_w = 1920
        self.img_h = 1080

        # Puntos reales de cancha (metros)
        self.field_pts = np.array([
            [0.0, 0.0],
            [0.0, 13.84],
            [0.0, 24.84],
            [5.5, 24.84],
            [16.5, 13.84],
            [16.5, 24.84],
            [16.5, 43.16],
            [16.5, 54.16],
            [43.35, 34.0],
        ], dtype=np.float32)

        # Matriz H ground-truth que proyecta imagen a campo
        # (src=imagen 1920x1080, dst=campo 105x68)
        src_corners = np.array([
            [500.0, 190.0],
            [315.0, 260.0],
            [130.0, 330.0],
            [285.0, 345.0],
            [760.0, 300.0],
            [590.0, 390.0],
            [325.0, 540.0],
            [10.0,  720.0],
            [1510.0, 565.0],
        ], dtype=np.float32)

        self.H_gt, _ = cv2.findHomography(src_corners, self.field_pts)
        self.clean_image_pts = src_corners

    def test_reprojection_error_computation(self):
        """Verifica el cálculo de error medio y máximo en metros."""
        err_info = compute_reprojection_errors(self.H_gt, self.clean_image_pts, self.field_pts)
        self.assertLess(err_info["mean_error_m"], 0.75)
        self.assertLess(err_info["max_error_m"], 2.0)
        self.assertEqual(len(err_info["raw_errors_m"]), len(self.clean_image_pts))

    def test_convex_hull_and_coverage(self):
        """Verifica el cálculo de Convex Hull y áreas de cobertura en px² y m²."""
        cov = compute_convex_hull_and_coverage(
            self.clean_image_pts, self.field_pts, self.img_w, self.img_h
        )
        self.assertGreater(cov["hull_area_field_m2"], 800.0)  # > 800 m²
        self.assertGreater(cov["field_coverage_ratio"], 0.10) # > 10% del campo
        self.assertGreater(cov["image_coverage_ratio"], 0.10) # > 10% de pantalla

    def test_collinearity_detection(self):
        """Verifica que puntos colineales sean detectados con score de colinealidad bajo."""
        # Puntos perfectamente en una recta x=52.5 (línea de mediocampo)
        collinear_field_pts = np.array([
            [52.5, 0.0],
            [52.5, 10.0],
            [52.5, 25.0],
            [52.5, 50.0],
            [52.5, 68.0],
        ], dtype=np.float32)

        collin_score = compute_collinearity_score(collinear_field_pts)
        self.assertLess(collin_score, 0.01)

        # Validación geométrica debe rechazar puntos colineales
        collinear_img_pts = np.array([
            [960.0, 100.0],
            [960.0, 300.0],
            [960.0, 500.0],
            [960.0, 700.0],
            [960.0, 900.0],
        ], dtype=np.float32)

        is_ok, reason, _ = validate_keypoint_geometry(
            collinear_img_pts, collinear_field_pts, self.img_w, self.img_h
        )
        self.assertFalse(is_ok)
        self.assertIn("colineales", reason.lower())

    def test_small_area_rejection(self):
        """Verifica el rechazo de configuraciones con área de hull inferior al mínimo requerido."""
        # 4 puntos apiñados en un poste de arco de 2m x 2m
        tiny_field_pts = np.array([
            [0.0, 32.0],
            [2.0, 32.0],
            [2.0, 34.0],
            [0.0, 34.0],
        ], dtype=np.float32)
        tiny_img_pts = np.array([
            [100.0, 100.0],
            [110.0, 100.0],
            [110.0, 110.0],
            [100.0, 110.0],
        ], dtype=np.float32)

        is_ok, reason, meta = validate_keypoint_geometry(
            tiny_img_pts, tiny_field_pts, self.img_w, self.img_h, min_hull_area_m2=80.0
        )
        self.assertFalse(is_ok)
        self.assertIn("insuficiente", reason.lower())

    def test_border_keypoint_filtering(self):
        """Verifica que los puntos truncados en el borde de imagen (x=0, y=0, etc.) sean descartados."""
        candidates = [
            (0, np.array([0.0, 300.0]), 0.9, "12"),     # Borde izquierdo (x=0)
            (1, np.array([500.0, 200.0]), 0.95, "01"),  # Válido
            (2, np.array([1920.0, 400.0]), 0.8, "32"),  # Borde derecho (x=1920)
            (3, np.array([800.0, 1079.0]), 0.7, "18"),  # Borde inferior (y=1079)
            (4, np.array([600.0, 350.0]), 0.9, "11"),   # Válido
        ]

        filtered = filter_border_keypoints(candidates, img_width=1920, img_height=1080, border_margin_px=4.0)
        self.assertEqual(len(filtered), 2)
        labels = [c[3] for c in filtered]
        self.assertEqual(labels, ["01", "11"])

    def test_magsac_outlier_suppression(self):
        """Verifica que MAGSAC con umbral calibrado (1.5m) elimine outliers contaminantes."""
        pts_img = self.clean_image_pts.copy()
        pts_fld = self.field_pts.copy()

        # Inyectar un outlier masivo en el punto 7 (desplazado 12 metros)
        pts_fld[7] = pts_fld[7] + np.array([12.0, -8.0], dtype=np.float32)

        # 1. Con umbral antiguo (5.0m) o RANSAC tradicional:
        H_loose, mask_loose = cv2.findHomography(pts_img, pts_fld, cv2.RANSAC, 5.0)

        # 2. Con MAGSAC y umbral calibrado (1.5m):
        H_tight, mask_tight = cv2.findHomography(pts_img, pts_fld, cv2.USAC_MAGSAC, 1.5)

        # El punto contaminado debe ser marcado como outlier (mask=0) en el modo estricto
        self.assertEqual(mask_tight[7][0], 0, "El punto contaminado debe ser descartado como outlier")

        # Evaluar error sobre los 8 puntos inliers verdaderos
        proj_tight = cv2.perspectiveTransform(pts_img[:7].reshape(-1, 1, 2), H_tight).reshape(-1, 2)
        err_tight = np.linalg.norm(proj_tight - self.field_pts[:7], axis=1).mean()
        self.assertLess(err_tight, 0.75, "H_tight debe mantener bajo error sobre los inliers reales")

    def test_temporal_tracker_smooth_motion(self):
        """Verifica que una cámara con movimiento fluido sea aceptada y suavizada."""
        tracker = TemporalHomographyTracker(max_shift_m=4.5)

        # Frame 1: Inicial
        H1, _, status1, _ = tracker.update(self.H_gt, None, self.img_w, self.img_h)
        self.assertEqual(status1, "INITIAL_ACCEPTED")

        # Frame 2: Pequeño pan suave (desplazamiento de 0.6m)
        shift_mat = np.array([
            [1.0, 0.0, 0.6],
            [0.0, 1.0, 0.2],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        H2_cand = shift_mat @ self.H_gt

        H2, _, status2, meta2 = tracker.update(H2_cand, None, self.img_w, self.img_h)
        self.assertEqual(status2, "ACCEPTED_SMOOTHED")
        self.assertLess(meta2["center_shift_m"], 1.0)
        self.assertEqual(tracker.reject_streak, 0)

    def test_temporal_tracker_rejects_sudden_jump(self):
        """Verifica que saltos bruscos de traslación o rotación sean rechazados y se mantenga H_prev."""
        tracker = TemporalHomographyTracker(max_shift_m=4.5, max_rot_deg=7.0)

        # Frame 1: Inicial
        H1, _, _, _ = tracker.update(self.H_gt, None, self.img_w, self.img_h)

        # Frame 2: Salto irreal de 18 metros
        jump_mat = np.array([
            [1.0, 0.0, 18.0],
            [0.0, 1.0, 5.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        H_bad = jump_mat @ self.H_gt

        H_out, _, status, meta = tracker.update(H_bad, None, self.img_w, self.img_h)
        self.assertEqual(status, "REJECTED_HELD")
        self.assertEqual(tracker.reject_streak, 1)
        # Debe haber devuelto H1 (la homografía previa)
        diff = np.linalg.norm(H_out - H1)
        self.assertAlmostEqual(diff, 0.0, places=4)

    def test_temporal_tracker_auto_reset(self):
        """Verifica que tras una racha de rechazos continuos (corte de toma), el tracker se auto-reinicie."""
        tracker = TemporalHomographyTracker(max_shift_m=4.5, max_reject_streak=5)
        tracker.update(self.H_gt, None, self.img_w, self.img_h)

        jump_mat = np.array([
            [1.0, 0.0, 20.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        H_new_scene = jump_mat @ self.H_gt

        # Mandar 4 frames rechazados
        for i in range(4):
            _, _, st, _ = tracker.update(H_new_scene, None, self.img_w, self.img_h)
            self.assertEqual(st, "REJECTED_HELD")

        # Al frame 5 debe hacer auto-reset y aceptar la nueva posición de cámara
        _, _, st5, _ = tracker.update(H_new_scene, None, self.img_w, self.img_h)
        self.assertEqual(st5, "RESET_ACCEPTED")
        self.assertEqual(tracker.reject_streak, 0)

    def test_field_coords_fifa_geometry(self):
        """Verifica que las coordenadas del arco de penal (medialuna) respeten la física y geometría FIFA."""
        dims = FieldDimensions()
        half_wid = dims.width / 2.0  # 34.0m
        dx_penalty = dims.penalty_box_length - dims.penalty_spot_distance  # 16.5 - 11.0 = 5.5m
        dy_arc_expected = np.sqrt(dims.centre_circle_radius**2 - dx_penalty**2)  # 7.3125m

        # Área grande izquierda
        self.assertAlmostEqual(FIELD_COORDS["11"][0], 16.5, places=4)
        self.assertAlmostEqual(FIELD_COORDS["11"][1], half_wid - dy_arc_expected, places=4)
        self.assertAlmostEqual(FIELD_COORDS["12"][0], 16.5, places=4)
        self.assertAlmostEqual(FIELD_COORDS["12"][1], half_wid + dy_arc_expected, places=4)
        self.assertAlmostEqual(FIELD_COORDS["12"][1] - FIELD_COORDS["11"][1], 2 * dy_arc_expected, places=3)

        # Área grande derecha
        self.assertAlmostEqual(FIELD_COORDS["21"][0], dims.length - 16.5, places=4)
        self.assertAlmostEqual(FIELD_COORDS["21"][1], half_wid - dy_arc_expected, places=4)
        self.assertAlmostEqual(FIELD_COORDS["22"][0], dims.length - 16.5, places=4)
        self.assertAlmostEqual(FIELD_COORDS["22"][1], half_wid + dy_arc_expected, places=4)

    def test_point_residuals_and_table(self):
        """Verifica el cálculo de residuales punto a punto y el formateo de la tabla de diagnóstico."""
        test_labels = ["01", "02", "03", "07", "10", "14"]
        fld_pts = np.array([FIELD_COORDS[l] for l in test_labels], dtype=np.float32)
        H_inv = np.linalg.inv(self.H_gt)
        img_pts = cv2.perspectiveTransform(fld_pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)

        candidates = [
            (i, img_pts[i], 0.95, test_labels[i])
            for i in range(len(test_labels))
        ]

        residuals = compute_point_residuals(self.H_gt, candidates)
        self.assertEqual(len(residuals), len(candidates))
        for r in residuals:
            self.assertLess(r.error_m, 0.05, f"Error residual en metros debe ser casi nulo para H_gt: {r.error_m}")
            self.assertLess(r.error_px, 1.0, f"Error residual en píxeles debe ser casi nulo: {r.error_px}")

        table_str = generate_residuals_table(residuals)
        self.assertIn("KP_ID", table_str)
        self.assertIn("INLIER", table_str)
        self.assertIn("Error (m)", table_str)

    def test_leave_one_out_analysis_detects_corrupted_keypoint(self):
        """Verifica que el análisis Leave-One-Out (Jackknife) aísle y ranquee al keypoint más dañino."""
        test_labels = ["01", "02", "03", "07", "10", "13", "14"]
        fld_pts = np.array([FIELD_COORDS[l] for l in test_labels], dtype=np.float32)
        H_inv = np.linalg.inv(self.H_gt)
        img_pts = cv2.perspectiveTransform(fld_pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)

        candidates = []
        for i, lbl in enumerate(test_labels):
            xy = img_pts[i].copy()
            # Corromper deliberadamente el punto en índice 2 ("03") desplazándolo 100px en imagen
            if i == 2:
                xy += np.array([100.0, -80.0], dtype=np.float32)
            candidates.append((i, xy, 0.95, lbl))

        results, meta = run_leave_one_out_analysis(candidates)
        self.assertGreater(len(results), 0)
        # El punto más dañino (Rank 1) debe ser el índice 2 ("03")
        top_bad = results[0]
        self.assertEqual(top_bad.excluded_idx, 2)
        self.assertEqual(top_bad.excluded_label, "03")
        self.assertGreater(top_bad.delta_error_m, 0.0)
        self.assertGreater(top_bad.damage_score, 0.0)

        report_str = format_leave_one_out_report(results, meta)
        self.assertIn("LEAVE-ONE-OUT", report_str)
        self.assertIn("Rank", report_str)

    def test_detect_spatial_anomalies(self):
        """Verifica la detección automática de swaps de identidad y outliers sistemáticos."""
        # Simular residuales donde KP 07 y KP 08 tienen coordenadas verticales invertidas
        from homography.diagnostics_tool import PointResidual
        res_normal = PointResidual(
            kp_idx=6, kp_label="07", confidence=0.9,
            detected_px=(300.0, 700.0), reprojected_px=(300.0, 400.0),
            field_m=(5.5, 24.84), reprojected_m=(5.5, 38.0),
            error_px=300.0, error_m=13.0, is_inlier=False
        )
        res_inverted = PointResidual(
            kp_idx=7, kp_label="08", confidence=0.9,
            detected_px=(300.0, 300.0), reprojected_px=(300.0, 650.0),
            field_m=(5.5, 43.16), reprojected_m=(5.5, 20.0),
            error_px=350.0, error_m=15.0, is_inlier=False
        )
        anomalies = detect_spatial_anomalies_and_swaps([res_normal, res_inverted])
        swap_detected = any(a["type"] == "IDENTITY_SWAP" for a in anomalies)
        self.assertTrue(swap_detected, "Debe detectar inversión espacial entre 07 y 08")

    def test_draw_reprojection_residuals_renders_correctly(self):
        """Verifica que draw_reprojection_residuals dibuje sin error y modifique el canvas."""
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Mock de objeto resultado con keypoints
        class MockKeypoints:
            xy = np.array([self.clean_image_pts], dtype=np.float32)
            conf = np.ones((1, len(self.clean_image_pts)), dtype=np.float32) * 0.9

            def __len__(self):
                return len(self.xy)

        class MockResult:
            keypoints = MockKeypoints()

        drawn = draw_reprojection_residuals(canvas, self.H_gt, MockResult())
        self.assertIsNotNone(drawn)
        self.assertGreater(drawn.sum(), 0, "El overlay residual debe pintar píxeles en el canvas")


if __name__ == "__main__":
    unittest.main()
