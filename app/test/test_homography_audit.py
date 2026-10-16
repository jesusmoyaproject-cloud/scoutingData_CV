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
    filter_spatial_swaps_and_hallucinations,
)
from homography.estimator import compute_homography
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

    def test_homography_audit_exporter_json_structure(self):
        """Verifica que HomographyAuditExporter exporte el JSON con la estructura correcta y completa."""
        import json
        import tempfile
        from homography.keypoint_audit_logger import HomographyAuditExporter

        exporter = HomographyAuditExporter()

        # Mock de keypoint result
        class MockKeypoints:
            xy = np.array([self.clean_image_pts], dtype=np.float32)
            conf = np.ones((1, len(self.clean_image_pts)), dtype=np.float32) * 0.95

            def __len__(self):
                return len(self.xy)

        class MockResult:
            keypoints = MockKeypoints()

        # Simular frame 1 (Initial)
        trans_1 = {"status": "INITIAL_ACCEPTED", "reject_streak": 0}
        exporter.record_frame(
            frame_idx=1,
            timestamp_sec=0.033,
            kp_result=MockResult(),
            homography_status="INITIAL_ACCEPTED",
            H_effective=self.H_gt,
            H_candidate=self.H_gt,
            transition_metrics=trans_1,
        )

        # Simular frame 2 (Smoothed con corrección)
        trans_2 = {
            "status": "ACCEPTED_SMOOTHED",
            "center_shift_m": 0.45,
            "max_corner_shift_m": 0.80,
            "rotation_deg": 0.35,
            "scale_ratio": 1.01,
            "shear_ratio": 1.02,
            "smoothing_alpha": 0.65,
            "effective_max_center_shift_m": 3.5,
            "effective_max_corner_shift_m": 6.0,
            "effective_max_rot_deg": 5.0,
            "max_shear_ratio": 1.30,
            "allowed_scale_range": "[0.75, 1.30]",
            "reject_streak": 0,
        }
        exporter.record_frame(
            frame_idx=2,
            timestamp_sec=0.066,
            kp_result=MockResult(),
            homography_status="ACCEPTED_SMOOTHED",
            H_effective=self.H_gt,
            H_candidate=self.H_gt,
            transition_metrics=trans_2,
        )

        # Simular frame 3 (Rechazado con holding)
        trans_3 = {
            "status": "REJECTED_HELD",
            "center_shift_m": 4.8,
            "max_corner_shift_m": 7.2,
            "rotation_deg": 1.2,
            "scale_ratio": 1.05,
            "shear_ratio": 1.01,
            "effective_max_center_shift_m": 3.5,
            "effective_max_corner_shift_m": 6.0,
            "effective_max_rot_deg": 5.0,
            "max_shear_ratio": 1.30,
            "allowed_scale_range": "[0.75, 1.30]",
            "reject_reason": "CENTER_SHIFT",
            "reject_details": "Salto de centro excesivo: 4.80m > 3.50m",
            "reject_streak": 1,
        }
        exporter.record_frame(
            frame_idx=3,
            timestamp_sec=0.100,
            kp_result=MockResult(),
            homography_status="REJECTED_HELD",
            H_effective=self.H_gt,
            H_candidate=self.H_gt,
            transition_metrics=trans_3,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            exported_path = exporter.save_to_json(
                tmp_path,
                metadata={"video_name": "match_test.mp4", "fps": 30.0, "total_frames": 3}
            )
            self.assertTrue(Path(exported_path).exists())

            with open(exported_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 1. Verificar metadata y summary
            self.assertIn("metadata", data)
            self.assertEqual(data["metadata"]["video_name"], "match_test.mp4")
            self.assertEqual(data["metadata"]["fps"], 30.0)
            self.assertIn("summary", data["metadata"])
            summary = data["metadata"]["summary"]
            self.assertEqual(summary["total_frames_evaluated"], 3)
            self.assertEqual(summary["frames_with_valid_homography"], 3)
            self.assertEqual(summary["frames_with_corrections_applied"], 2) # frame 2 y 3
            self.assertIn("CENTER_SHIFT", summary["rejections_by_reason"])

            # 2. Verificar frames
            self.assertIn("frames", data)
            self.assertEqual(len(data["frames"]), 3)

            # Frame 1
            f1 = data["frames"][0]
            self.assertEqual(f1["frame"], 1)
            self.assertEqual(f1["timestamp"], 0.033)
            self.assertTrue(f1["homography"]["valid"])
            self.assertIsInstance(f1["homography"]["matrix_effective"], list)
            self.assertEqual(len(f1["homography"]["matrix_effective"]), 3)
            self.assertEqual(len(f1["homography"]["matrix_effective"][0]), 3)
            self.assertIsInstance(f1["puntos_correctos"], list)
            self.assertIsInstance(f1["puntos_fallidos"], list)
            self.assertFalse(f1["correcciones"]["correction_applied"])
            self.assertEqual(f1["correcciones"]["correction_type"], "INITIAL")

            # Frame 2 (Corrección de suavizado)
            f2 = data["frames"][1]
            self.assertEqual(f2["frame"], 2)
            self.assertTrue(f2["correcciones"]["correction_applied"])
            self.assertEqual(f2["correcciones"]["correction_type"], "TEMPORAL_SMOOTHING")
            self.assertEqual(f2["correcciones"]["transition_metrics"]["center_shift_m"], 0.45)
            self.assertEqual(f2["correcciones"]["transition_metrics"]["smoothing_alpha"], 0.65)

            # Frame 3 (Corrección de holding por rechazo)
            f3 = data["frames"][2]
            self.assertEqual(f3["frame"], 3)
            self.assertTrue(f3["correcciones"]["correction_applied"])
            self.assertEqual(f3["correcciones"]["correction_type"], "HOLD_PREVIOUS_ON_REJECTION")
            self.assertEqual(f3["correcciones"]["reject_reason"], "CENTER_SHIFT")
            self.assertEqual(f3["correcciones"]["reject_streak"], 1)
            self.assertIn("4.80m", f3["correcciones"]["details"])

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_homography_audit_points_details(self):
        """Verifica la clasificación precisa de puntos correctos y fallidos."""
        from homography.keypoint_audit_logger import HomographyAuditExporter
        from homography.diagnostics_tool import PointResidual

        exporter = HomographyAuditExporter(conf_threshold=0.35)

        # Mock de 3 puntos: 1 inlier, 1 outlier RANSAC, 1 baja confianza
        # Puntos: 01 (u=500, v=190), 02 (u=315, v=260), 03 (u=130, v=330)
        kps_xy = np.array([[
            [500.0, 190.0],  # 01
            [315.0, 260.0],  # 02
            [130.0, 330.0],  # 03
            [2.0, 2.0],      # borde pantalla < 4px
        ]], dtype=np.float32)
        kps_conf = np.array([[0.90, 0.85, 0.20, 0.95]], dtype=np.float32)

        class MockKeypoints:
            xy = kps_xy
            conf = kps_conf

            def __len__(self):
                return len(self.xy)

        class MockResult:
            keypoints = MockKeypoints()

        res_01 = PointResidual(
            kp_idx=0, kp_label="01", confidence=0.90,
            detected_px=(500.0, 190.0), reprojected_px=(500.2, 189.9),
            field_m=(0.0, 0.0), reprojected_m=(0.02, -0.01),
            error_px=0.22, error_m=0.022, is_inlier=True
        )
        res_02 = PointResidual(
            kp_idx=1, kp_label="02", confidence=0.85,
            detected_px=(315.0, 260.0), reprojected_px=(340.0, 290.0),
            field_m=(0.0, 13.84), reprojected_m=(3.2, 17.5),
            error_px=39.05, error_m=4.82, is_inlier=False
        )

        exporter.record_frame(
            frame_idx=10,
            timestamp_sec=0.333,
            kp_result=MockResult(),
            homography_status="ACCEPTED_SMOOTHED",
            H_effective=self.H_gt,
            H_candidate=self.H_gt,
            residuals=[res_01, res_02],
            conf_threshold=0.35,
        )

        f_data = exporter.frames_data[0]
        correctos = f_data["puntos_correctos"]
        fallidos = f_data["puntos_fallidos"]

        # 01 debe estar en puntos_correctos como inlier
        self.assertEqual(len(correctos), 1)
        self.assertEqual(correctos[0]["kp_label"], "01")
        self.assertEqual(correctos[0]["status"], "INLIER")
        self.assertAlmostEqual(correctos[0]["error_m"], 0.022, places=2)

        # 02 (outlier), 03 (baja conf < 0.35) y 4to punto (borde) deben estar en puntos_fallidos
        fail_reasons = {p["reason"] for p in fallidos}
        self.assertIn("OUTLIER_RANSAC", fail_reasons)
        self.assertIn("LOW_CONFIDENCE", fail_reasons)
        self.assertIn("DISCARDED_SCREEN_BORDER", fail_reasons)

    def test_homography_audit_no_console_logging(self):
        """Verifica que record_frame y save_to_json no emitan logs de puntos por consola."""
        import logging
        from homography.keypoint_audit_logger import HomographyAuditExporter

        exporter = HomographyAuditExporter()

        class MockKeypoints:
            xy = np.array([self.clean_image_pts], dtype=np.float32)
            conf = np.ones((1, len(self.clean_image_pts)), dtype=np.float32) * 0.95

            def __len__(self):
                return len(self.xy)

        class MockResult:
            keypoints = MockKeypoints()

        with self.assertLogs("homography.audit", level="DEBUG") as cm:
            # Emite un log de prueba para verificar que el logger funciona
            logging.getLogger("homography.audit").debug("TEST_LOG_MARKER")
            exporter.record_frame(
                frame_idx=1,
                timestamp_sec=0.033,
                kp_result=MockResult(),
                homography_status="INITIAL_ACCEPTED",
                H_effective=self.H_gt,
                H_candidate=self.H_gt,
            )

        # Verificar que solo esté nuestro marcador y NINGÚN log de puntos o frame
        self.assertEqual(len(cm.output), 1)
        self.assertIn("TEST_LOG_MARKER", cm.output[0])
        self.assertNotIn("puntos_correctos", cm.output[0])
        self.assertNotIn("puntos_fallidos", cm.output[0])

    def test_border_keypoint_dynamic_banner_grass_filter(self):
        """Verifica que filter_border_keypoints descarte puntos sobre banners sin césped cuando se provee frame_bgr."""
        frame = np.full((300, 300, 3), (35, 140, 35), dtype=np.uint8)  # Césped verde
        frame[240:300, :] = (0, 0, 240)  # Banner rojo inferior

        candidates = [
            (0, np.array([150.0, 100.0]), 0.90, "15"),  # Sobre césped -> válido
            (1, np.array([150.0, 260.0]), 0.85, "18"),  # Sobre banner publicitario -> descartado
        ]

        filtered = filter_border_keypoints(
            candidates, img_width=300, img_height=300, border_margin_px=10.0, frame_bgr=frame
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][3], "15")

    def test_filter_spatial_swaps_and_hallucinations_corner(self):
        """Verifica que un córner alucinado (27) sobre el área grande (20) sea descartado por topología."""
        candidates = [
            (19, np.array([500.0, 200.0]), 0.85, "20"),
            (20, np.array([500.0, 400.0]), 0.85, "21"),
            (21, np.array([500.0, 600.0]), 0.85, "22"),
            (22, np.array([500.0, 800.0]), 0.85, "23"),
            # Alucinación: córner 27 muy pegado al vértice 20 (<120 px pero a >12m en campo)
            (26, np.array([508.0, 205.0]), 0.75, "27"),
        ]

        filtered = filter_spatial_swaps_and_hallucinations(candidates, FIELD_COORDS)
        labels = [c[3] for c in filtered]
        self.assertNotIn("27", labels, "El córner alucinado debe haber sido descartado")
        self.assertIn("20", labels)
        self.assertEqual(len(filtered), 4)

    def test_fast_recovery_rejects_scale_contraction(self):
        """Verifica que Fast-Recovery no re-ancle si la homografía candidata tiene contracción de escala."""
        tracker = TemporalHomographyTracker(max_shift_m=4.5)
        tracker.update(self.H_gt, np.ones((len(self.clean_image_pts), 1), dtype=np.uint8), frame_id=1)

        T = np.array([[1, 0, -960.0], [0, 1, -540.0], [0, 0, 1]], dtype=np.float64)
        S = np.array([[0.65, 0, 0], [0, 0.65, 0], [0, 0, 1]], dtype=np.float64)
        T_inv = np.array([[1, 0, 960.0], [0, 1, 540.0], [0, 0, 1]], dtype=np.float64)
        M_img = T_inv @ S @ T
        H_deformed = self.H_gt @ M_img

        diag = HomographyDiagnostics(
            mean_reprojection_error_m=0.30,
            ransac_inliers=10,
            field_coverage_ratio=0.20,
        )

        _, _, status, _ = tracker.update(
            H_deformed,
            np.ones((10, 1), dtype=np.uint8),
            frame_id=2,
            diagnostics=diag,
        )
        self.assertEqual(status, "REJECTED_HELD")
        self.assertNotEqual(status, "FAST_RECOVERY_ACCEPTED")

    def test_auto_reset_avoids_scale_or_shear(self):
        """Verifica que tras exceder max_reject_streak, no se produzca auto-reset si la falla es SCALE o SHEAR."""
        tracker = TemporalHomographyTracker(max_shift_m=4.5, max_reject_streak=3)
        tracker.update(self.H_gt, np.ones((len(self.clean_image_pts), 1), dtype=np.uint8), frame_id=1)

        T = np.array([[1, 0, -960.0], [0, 1, -540.0], [0, 0, 1]], dtype=np.float64)
        S = np.array([[0.50, 0, 0], [0, 0.50, 0], [0, 0, 1]], dtype=np.float64)
        T_inv = np.array([[1, 0, 960.0], [0, 1, 540.0], [0, 0, 1]], dtype=np.float64)
        M_img = T_inv @ S @ T
        H_scale = self.H_gt @ M_img

        # Enviar 3 frames consecutivos con deformación de escala
        for fid in range(2, 5):
            _, _, status, metrics = tracker.update(
                H_scale,
                np.ones((len(self.clean_image_pts), 1), dtype=np.uint8),
                frame_id=fid,
            )

        # En el 3er frame de rechazo (streak=3 >= max_reject_streak), el estado debe seguir siendo REJECTED_HELD
        self.assertEqual(status, "REJECTED_HELD")
        self.assertNotEqual(status, "RESET_ACCEPTED")
        self.assertEqual(metrics.get("reject_reason"), "SCALE")


if __name__ == "__main__":
    unittest.main()

