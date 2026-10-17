"""
ScoutingData v5.0 - Pipeline Principal Optimizado

Cambios vs v4.0:
  - Modo directo (sin HTTP SOA): DirectInferenceEngine
  - Sin event_service (SoccerNet eliminado)
  - Selección automática .pt CUDA (Kaggle) / OpenVINO CPU (Local)
  - Caché de líneas de cancha (OPT-4)
  - Caché de minimap template (OPT-5)
  - Homografía cada N frames configurable (OPT-7)
  - AsyncVideoWriter para escritura en background (OPT-8)
  - ThreadedOpenCVReader para lectura/decodificación asíncrona de video (OPT-9)
  - Profiling de read_and_decode por frame y worker decode metrics
  - Renders in-place (sin .copy() redundantes) (OPT-6)
"""

import os
import sys
import time
import logging
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
import supervision as sv

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings
from config.field_dimensions import FieldDimensions
from engine.direct_inference import DirectInferenceEngine
from engine.async_writer import AsyncVideoWriter
from engine.threaded_reader import ThreadedOpenCVReader
from homography.estimator import compute_homography
from homography.temporal_validator import TemporalHomographyTracker
from homography.transforms import pixel_to_field
from homography.keypoint_audit_logger import HomographyAuditExporter
from homography.diagnostics_tool import (
    generate_residuals_table,
    run_leave_one_out_analysis,
    format_leave_one_out_report,
    detect_spatial_anomalies_and_swaps,
)
from sports.ball_tracker import BallTracker
from sports.team_detector import TeamDetector
from visualization.overlays import (
    get_pitch_field_points,          # OPT-4: puntos pre-calculados
    draw_pitch_lines_on_image,
    draw_calibration_overlay,
    draw_player_detections,
    draw_ball_detections,
)
from visualization.minimap import (
    get_cached_pitch_template,       # OPT-5: template pre-renderizado
    draw_minimap_with_projections,
    draw_player_projections_on_minimap,
    draw_ball_projection_on_minimap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SoccerAnalysisV5")

# Silenciar warnings y deprecaciones del logger interno de Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Logger de timing de inferencia — desactivado por defecto, se activa con --debug-homography
perf_logger = logging.getLogger("DirectInferenceEngine.perf")
perf_logger.setLevel(logging.WARNING)  # Silencioso por defecto
perf_logger.propagate = False

# Logger de homografía — desactivado por defecto, se activa con --debug-homography
homography_logger = logging.getLogger("Homography.debug")
homography_logger.setLevel(logging.WARNING)  # Silencioso por defecto
homography_logger.propagate = False

def _enable_debug_mode():
    """Activa logs detallados de homografía e inferencia (--debug-homography)."""
    _fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    for lg in (perf_logger, homography_logger):
        lg.setLevel(logging.DEBUG)
        if not lg.handlers:
            _h = logging.StreamHandler()
            _h.setLevel(logging.DEBUG)
            _h.setFormatter(_fmt)
            lg.addHandler(_h)



def _extract_ball_candidates(ball_result, conf_threshold: float) -> List[Dict]:
    """Extrae todos los candidatos de detección de balón ordenados por confianza."""
    if ball_result is None or ball_result.boxes is None or len(ball_result.boxes) == 0:
        return []
    b_boxes = ball_result.boxes.xyxy.cpu().numpy()
    b_confs = ball_result.boxes.conf.cpu().numpy()
    candidates = []
    for box, conf in zip(b_boxes, b_confs):
        c = float(conf)
        if c >= conf_threshold:
            bx1, by1, bx2, by2 = box
            candidates.append({
                "pixel_x":     round(float((bx1 + bx2) / 2), 2),
                "pixel_y":     round(float((by1 + by2) / 2), 2),
                "confidence":  round(c, 4),
                "interpolated": False,
                "bbox":        [round(float(bx1), 2), round(float(by1), 2), round(float(bx2), 2), round(float(by2), 2)],
            })
    # Ordenar candidatos de mayor a menor confianza
    candidates.sort(key=lambda d: d["confidence"], reverse=True)
    return candidates


def _extract_ball_detection(ball_result, conf_threshold: float) -> Optional[Dict]:
    """Extrae la mejor detección de balón de un resultado YOLO nativo (compatibilidad)."""
    candidates = _extract_ball_candidates(ball_result, conf_threshold)
    return candidates[0] if candidates else None


class SoccerAnalysisV5:
    """
    Pipeline de análisis v5.0: Inferencia directa + renders cacheados / Modo Headless.
    Sin SoccerNet, sin HTTP SOA.
    """

    def __init__(
        self,
        conf_keypoint: float = settings.CONF_KEYPOINT,
        conf_player:   float = settings.CONF_PLAYER,
        conf_ball:     float = settings.CONF_BALL,
        ransac_threshold: float = settings.RANSAC_THRESHOLD,
        field_dims: FieldDimensions = FieldDimensions(),
        minimap_scale: float = settings.MINIMAP_SCALE,
        homography_interval: int = settings.HOMOGRAPHY_INTERVAL,
        headless: Optional[bool] = None,
        draw_outliers: Optional[bool] = None,
        enable_corner_shift: Optional[bool] = None,
        use_threaded_reader: bool = True,
        reader_buffer_size: int = 64,
        player_stride: int = settings.PLAYER_INFERENCE_STRIDE,
        show_frame_number: Optional[bool] = None,
    ):
        self.conf_keypoint   = conf_keypoint
        self.conf_player     = conf_player
        self.conf_ball       = conf_ball
        self.ransac_threshold = ransac_threshold
        self.field_dims      = field_dims
        self.minimap_scale   = minimap_scale
        self.homography_interval = homography_interval
        self.headless        = headless if headless is not None else settings.HEADLESS_MODE
        self.draw_outliers   = draw_outliers if draw_outliers is not None else settings.DRAW_OUTLIERS
        self.enable_corner_shift = enable_corner_shift if enable_corner_shift is not None else settings.ENABLE_CORNER_SHIFT
        self.use_threaded_reader = use_threaded_reader
        self.reader_buffer_size = reader_buffer_size
        self.player_stride   = max(1, player_stride)
        self.show_frame_number = (
            show_frame_number
            if show_frame_number is not None
            else getattr(settings, "SHOW_FRAME_NUMBER", settings.ENVIRONMENT == "LOCAL")
        )


        # Motor de inferencia directa (sin HTTP)
        self.engine = DirectInferenceEngine()

        # OPT-4: Pre-calcular puntos de líneas de cancha (solo una vez)
        self._pitch_field_segments = get_pitch_field_points(field_dims)

        # OPT-5: Pre-renderizar minimap template (solo una vez)
        self._minimap_template = get_cached_pitch_template(
            field_dims, minimap_scale, padding_meters=5.0
        )

        # Estado del tracker de balón
        self.ball_tracker = BallTracker()

        # Clasificador de equipos y color de camiseta (Fases 1 y 2)
        self.team_detector = TeamDetector(
            pitch_length=field_dims.length,
            pitch_width=field_dims.width,
        )

        # Buffers cinemáticos para tracking y multi-rate player inference (Fase 2)
        self._last_player_dets: List[Dict] = []
        self._last_player_result = None
        self._player_positions: Dict[int, Tuple[float, float]] = {}
        self._player_velocities: Dict[int, Tuple[float, float]] = {}

        # Tracker cinemático y caché de homografía (Robustez Temporal v5.1)
        self.homography_tracker = TemporalHomographyTracker(
            enable_corner_shift=self.enable_corner_shift,
        )
        self._cached_H:    Optional[np.ndarray] = None
        self._cached_mask: Optional[np.ndarray] = None
        self._h_reject_streak: int = 0   # Contador de rechazos consecutivos para auto-reset
        self._cached_field_coverage_ratio: float = 0.0
        self._cached_hull_m2: float = 0.0

        # Auditor y exportador forense de homografía (silencioso sin logs)
        self.homography_audit = HomographyAuditExporter()


    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
        is_video: bool = True,
        headless: Optional[bool] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[Dict], Optional[Dict]]:

        is_headless = headless if headless is not None else self.headless

        # ── 1. Inferencia Directa ────────────────────────────────────────────
        should_infer_players = (frame_idx % self.player_stride == 0) or (len(self._last_player_dets) == 0)
        if should_infer_players:
            kp_result, player_result, ball_result = self.engine.infer(
                frame_bgr,
                conf_kp=self.conf_keypoint,
                conf_player=self.conf_player,
                conf_ball=self.conf_ball,
                is_video=is_video,
            )
            self._last_player_result = player_result
        else:
            # En frames omitidos para inferencia de jugadores (Fase 2):
            # Keypoints y balón se ejecutan con normalidad (keypoints no desacoplado)
            kp_result = self.engine.infer_keypoints(frame_bgr, conf=self.conf_keypoint)
            ball_result = self.engine.infer_ball(frame_bgr, conf=self.conf_ball)
            player_result = self._last_player_result


        # ── 2. Ball Tracker (Filtro de Kalman + Mahalanobis + Selección Multicandidato) ──
        # Extraer candidatos por encima de un umbral dinámico permisivo para rescate cinemático
        ball_candidates = _extract_ball_candidates(ball_result, max(0.12, self.conf_ball * 0.4))
        if is_video:
            ball_detection, ball_history = self.ball_tracker.update(
                ball_candidates,
                H=self.homography_tracker.cached_H,
            )
        else:
            raw_ball_det = ball_candidates[0] if ball_candidates else None
            ball_detection = raw_ball_det
            ball_history = [(raw_ball_det["pixel_x"], raw_ball_det["pixel_y"])] if raw_ball_det else []

        # ── 3. Homografía (Tracker cinemático temporal + USAC_MAGSAC) ──
        h_img, w_img = frame_bgr.shape[:2]
        recalc = (frame_idx % self.homography_interval == 0) or (self.homography_tracker.cached_H is None)
        H_cand = None
        cand_residuals = None
        cand_reject_reason = None
        cand_details = None
        status = "HELD_INTERVAL"
        diag = None
        trans_meta = {}

        if recalc:
            mask_cand = None
            conf_weight = 1.0
            try:
                res = compute_homography(
                    kp_result,
                    conf_threshold=self.conf_keypoint,
                    ransac_threshold=self.ransac_threshold,
                    img_width=w_img,
                    img_height=h_img,
                    frame_bgr=frame_bgr,
                )
                H_cand = res.H
                mask_cand = res.mask
                diag = getattr(res, "diagnostics", None)
                cand_residuals = getattr(res, "residuals", None)
                if diag is not None:
                    err_pen = 1.0 / max(0.5, diag.mean_reprojection_error_m)
                    conf_weight = float(np.clip(diag.inlier_ratio * err_pen, 0.5, 1.5))
                    self._cached_field_coverage_ratio = diag.field_coverage_ratio
                    self._cached_hull_m2 = diag.hull_area_field_m2

                # Telemetría forense detallada en modo debug
                if self.draw_outliers and cand_residuals:
                    homography_logger.info(
                        f"\n[Frame {frame_idx}] 📋 TABLA DE RESIDUALES:\n"
                        + generate_residuals_table(cand_residuals)
                    )
                    anomalies = detect_spatial_anomalies_and_swaps(cand_residuals)
                    for anom in anomalies:
                        homography_logger.warning(
                            f"[Frame {frame_idx}] 🚨 {anom['type']} ({anom['severity']}): {anom['message']}"
                        )
                    if len(getattr(res, "valid_candidates", [])) >= 5:
                        loo_res, loo_meta = run_leave_one_out_analysis(res.valid_candidates)
                        homography_logger.info(
                            f"\n[Frame {frame_idx}] 🔬 " + format_leave_one_out_report(loo_res, loo_meta)
                        )
            except Exception as exc:
                cand_details = str(exc)
                exc_str = cand_details.lower()
                if "colineal" in exc_str:
                    cand_reject_reason = "COLLINEARITY"
                elif "cobertura" in exc_str or "área cubierta" in exc_str or "dispersión" in exc_str:
                    cand_reject_reason = "LOW_COVERAGE"
                else:
                    cand_reject_reason = "NO_HOMOGRAPHY"
                homography_logger.debug(
                    f"[Frame {frame_idx}] ✖ compute_homography descartó candidato: {exc}"
                )

            H_eff, mask_eff, status, trans_meta = self.homography_tracker.update(
                H_cand, mask_cand,
                img_width=w_img, img_height=h_img,
                confidence_weight=conf_weight,
                frame_id=frame_idx,
                diagnostics=diag,
                candidate_reject_reason=cand_reject_reason,
                details=cand_details,
            )
            self._cached_H = H_eff
            self._cached_mask = mask_eff
            self._h_reject_streak = self.homography_tracker.reject_streak

            if status in ("INITIAL_ACCEPTED", "ACCEPTED_SMOOTHED", "RESET_ACCEPTED"):
                inliers_str = f"inliers={diag.ransac_inliers}/{diag.kps_used}" if diag else "inliers=OK"
                shift_str = f"| dC={trans_meta.get('center_shift_m', 0.0):.1f}m" if trans_meta else ""
                homography_logger.debug(
                    f"[Frame {frame_idx}] ✅ H {status} | {inliers_str} {shift_str}"
                )
            else:
                homography_logger.debug(
                    f"[Frame {frame_idx}] ⚠ H {status} (streak={self._h_reject_streak}) — manteniendo previa"
                )
        else:
            H_eff = self._cached_H
            mask_eff = self._cached_mask
            trans_meta = {
                "status": "HELD_INTERVAL",
                "reject_streak": self.homography_tracker.reject_streak,
            }

        # Auditoría punto a punto y telemetría de correcciones (exportación silenciosa a JSON)
        if hasattr(self, "homography_audit") and self.homography_audit is not None:
            self.homography_audit.record_frame(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                kp_result=kp_result,
                homography_status=status,
                H_effective=H_eff,
                H_candidate=H_cand,
                diagnostics=diag,
                residuals=cand_residuals,
                compute_error_reason=cand_reject_reason,
                compute_error_details=cand_details,
                transition_metrics=trans_meta,
                conf_threshold=self.conf_keypoint,
                img_width=w_img,
                img_height=h_img,
            )

        H    = self._cached_H
        mask = self._cached_mask

        # Coordenadas de terreno para balón
        if ball_detection is not None and H is not None:
            bf_x, bf_y = pixel_to_field(ball_detection["pixel_x"], ball_detection["pixel_y"], H)
            ball_detection["pitch_x_m"] = round(bf_x, 2)
            ball_detection["pitch_y_m"] = round(bf_y, 2)
        elif ball_detection is not None:
            ball_detection["pitch_x_m"] = None
            ball_detection["pitch_y_m"] = None

        # ── 4. Player Detections, Tracking & Jersey Colors ───────────────────
        player_detections: List[Dict] = []
        if should_infer_players:
            if player_result is not None and player_result.boxes is not None and len(player_result.boxes.xyxy) > 0:
                # Conversión e indexación vectorizada con supervision
                sv_dets = sv.Detections.from_ultralytics(player_result)
                if len(sv_dets) > 0 and sv_dets.confidence is not None:
                    sv_dets = sv_dets[sv_dets.confidence >= self.conf_player]
                if len(sv_dets) > 1:
                    sv_dets = sv_dets.with_nms(
                        threshold=getattr(settings, "PLAYER_NMS_IOU", 0.45),
                        class_agnostic=True,
                    )

                for idx in range(len(sv_dets)):
                    box = sv_dets.xyxy[idx]
                    conf = float(sv_dets.confidence[idx]) if sv_dets.confidence is not None else 1.0
                    cls_id = int(sv_dets.class_id[idx]) if sv_dets.class_id is not None else 0
                    p_id = int(sv_dets.tracker_id[idx]) if sv_dets.tracker_id is not None else idx
                    x1, y1, x2, y2 = box
                    xc = float((x1 + x2) / 2)
                    yb = float(y2)
                    fx, fy = (pixel_to_field(xc, yb, H) if H is not None else (None, None))

                    # Actualizar velocidad para interpolación cinemática (Fase 2)
                    if p_id in self._player_positions:
                        prev_xc, prev_yb = self._player_positions[p_id]
                        self._player_velocities[p_id] = (xc - prev_xc, yb - prev_yb)
                    else:
                        self._player_velocities[p_id] = (0.0, 0.0)
                    self._player_positions[p_id] = (xc, yb)

                    player_detections.append({
                        "player_id":   p_id,
                        "class_id":    cls_id,
                        "confidence":  round(float(conf), 4),
                        "bbox":        [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                        "pixel_x":     round(xc, 2),
                        "pixel_y":     round(yb, 2),
                        "pitch_x_m":   round(fx, 2) if fx is not None else None,
                        "pitch_y_m":   round(fy, 2) if fy is not None else None,
                        "is_interpolated": False,
                    })

            # Clasificación de camisetas y equipos (Fases 1 y 2)
            self.team_detector.annotate_player_detections(frame_bgr, player_detections, frame_idx=frame_idx)
            self._last_player_dets = [dict(p) for p in player_detections]
        else:
            # Extrapolación cinemática en frames intermedios (Fase 2)
            for p_prev in self._last_player_dets:
                p = dict(p_prev)
                p_id = p["player_id"]
                vx, vy = self._player_velocities.get(p_id, (0.0, 0.0))
                p["pixel_x"] = round(p["pixel_x"] + vx, 2)
                p["pixel_y"] = round(p["pixel_y"] + vy, 2)
                if p.get("bbox") is not None:
                    bx1, by1, bx2, by2 = p["bbox"]
                    p["bbox"] = [round(bx1 + vx, 2), round(by1 + vy, 2), round(bx2 + vx, 2), round(by2 + vy, 2)]
                if H is not None:
                    fx, fy = pixel_to_field(p["pixel_x"], p["pixel_y"], H)
                    p["pitch_x_m"] = round(fx, 2) if fx is not None else None
                    p["pitch_y_m"] = round(fy, 2) if fy is not None else None
                p["is_interpolated"] = True
                player_detections.append(p)
            self._last_player_dets = [dict(p) for p in player_detections]

        # Si estamos en modo headless, omitir renderizado OpenCV (Overlays, Minimap, cvtColor)
        if is_headless:
            return None, None, H, player_detections, ball_detection

        # ── 5. Render Camera Frame (OPT-6: in-place, sin copias extra) ──────
        annotated = frame_bgr.copy()   # Una sola copia al inicio

        # Pintar nro de frame en LOCAL (al costado de Field Coverage)
        display_frame_idx = frame_idx if self.show_frame_number else None

        if H is not None:
            # OPT-4: usar puntos pre-calculados, solo aplicar H_inv
            draw_pitch_lines_on_image(
                annotated, H,
                precomputed_segments=self._pitch_field_segments,
                color=(0, 255, 255), thickness=2,
            )
            draw_calibration_overlay(
                annotated, kp_result, mask=mask,
                conf_threshold=self.conf_keypoint,
                draw_outliers=self.draw_outliers,
                field_coverage_ratio=self._cached_field_coverage_ratio,
                hull_area_m2=self._cached_hull_m2,
                H=H,
                draw_residuals=self.draw_outliers,
                frame_idx=display_frame_idx,
            )
        else:
            # Si H no está disponible, dibujar overlay con status H: LOST y mostrar keypoints candidatos detectados
            draw_calibration_overlay(
                annotated, kp_result, mask=mask,
                conf_threshold=self.conf_keypoint,
                draw_outliers=True,
                field_coverage_ratio=0.0,
                hull_area_m2=0.0,
                H=None,
                draw_residuals=False,
                frame_idx=display_frame_idx,
            )

        draw_player_detections(annotated, player_result,
                               player_detections=player_detections,
                               conf_threshold=self.conf_player,
                               ellipse_color=(255, 0, 255))
        draw_ball_detections(annotated, ball_detection=ball_detection,
                             ball_history=ball_history)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # ── 6. Render Minimap (OPT-5: partir de template cacheado) ──────────
        if H is not None:
            minimap = draw_minimap_with_projections(
                H=H, result=kp_result,
                conf_threshold=self.conf_keypoint,
                mask=mask, scale=self.minimap_scale,
                cached_template=self._minimap_template,   # OPT-5
                field_coverage_ratio=self._cached_field_coverage_ratio,
                hull_area_m2=self._cached_hull_m2,
            )
            draw_player_projections_on_minimap(
                minimap, player_result,
                player_detections=player_detections,
                H=H,
                conf_threshold=self.conf_player,
                scale=self.minimap_scale, player_color=(255, 0, 255),
            )
            draw_ball_projection_on_minimap(
                minimap, ball_detection=ball_detection, scale=self.minimap_scale,
            )

        else:
            minimap = self._minimap_template.copy()
            # Overlay visual de pérdida de homografía en el minimap
            h_m, w_m = minimap.shape[:2]
            banner_text = "HOMOGRAPHY LOST / SEARCHING"
            (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = max(10, (w_m - bw) // 2)
            by = h_m // 2
            sub = minimap[by - bh - 10 : by + 16, bx - 12 : bx + bw + 12]
            if sub.shape[0] > 0 and sub.shape[1] > 0:
                bg = np.zeros_like(sub)
                bg[:] = (0, 0, 110)
                cv2.addWeighted(bg, 0.70, sub, 0.30, 0, sub)
                minimap[by - bh - 10 : by + 16, bx - 12 : bx + bw + 12] = sub
                cv2.rectangle(minimap, (bx - 12, by - bh - 10), (bx + bw + 12, by + 16), (0, 0, 230), 2, cv2.LINE_AA)
                cv2.putText(minimap, banner_text, (bx, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        minimap_rgb = cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB)

        return annotated_rgb, minimap_rgb, H, player_detections, ball_detection

    def process_video(
        self,
        video_path: str,
        output_path: str,
        csv_output_path: Optional[str] = None,
        json_output_path: Optional[str] = None,
        parquet_output_path: Optional[str] = None,
        mot_output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        headless: Optional[bool] = None,
        debug_homography: bool = False,
        rejections_csv_path: Optional[str] = None,
        keypoints_csv_path: Optional[str] = None,
        homography_json_path: Optional[str] = None,
        use_threaded_reader: Optional[bool] = None,
        reader_buffer_size: Optional[int] = None,
    ) -> List[Dict]:
        import csv

        is_headless = headless if headless is not None else self.headless
        use_reader  = use_threaded_reader if use_threaded_reader is not None else self.use_threaded_reader
        buf_size    = reader_buffer_size if reader_buffer_size is not None else self.reader_buffer_size

        resolved = settings.resolve_video_source(video_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Video no encontrado: {resolved}")

        # Reiniciar auditor de homografía para este video
        if hasattr(self, "homography_audit") and self.homography_audit is not None:
            self.homography_audit.reset()

        # Configurar destino de CSV para rechazos de homografía
        actual_rejections_csv = rejections_csv_path or str(settings.OUTPUT_DIR / f"rejections_{resolved.stem}.csv")
        self.homography_tracker.csv_path = actual_rejections_csv

        # OPT-9: ThreadedOpenCVReader (prefetching asíncrono + profiling)
        if use_reader:
            reader = ThreadedOpenCVReader(str(resolved), buffer_size=buf_size)
        else:
            reader = cv2.VideoCapture(str(resolved))

        if not reader.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {resolved}")

        width  = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = float(reader.get(cv2.CAP_PROP_FPS)) or 25.0
        total  = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

        cuda_info = f" (CUDA: {settings.CUDA_AVAILABLE})" if settings.INFERENCE_DEVICE == "cuda" else ""
        reader_info = f" | ThreadedReader (buf={buf_size})" if use_reader else " | SequentialReader"
        # Frames efectivos a procesar (respetando max_frames)
        effective_total = min(total, max_frames) if max_frames else total
        limit_str = f" [límite: {max_frames}]" if max_frames else ""
        logger.info(
            f"📹 {resolved.name} | {width}x{height} @ {fps:.1f}fps | "
            f"{total} frames totales{limit_str} | {settings.INFERENCE_DEVICE.upper()} {settings.MODEL_FORMAT.upper()}{cuda_info}{reader_info}"
        )
        if debug_homography:
            _enable_debug_mode()
            self.draw_outliers = True
            logger.info("🔍 Modo DEBUG de homografía activado (outliers visibles)")

        writer: Optional[AsyncVideoWriter] = None
        if not is_headless:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        start_time  = time.time()
        all_events: List[Dict] = []
        tracking_frames: List[Dict] = []

        # ── Acumuladores de tiempo por fase ─────────────────────────────────
        total_read_decode_time = 0.0
        total_inference_time   = 0.0
        total_tracking_time    = 0.0
        total_homography_time  = 0.0
        total_json_time        = 0.0
        total_video_time       = 0.0

        try:
            while reader.isOpened():
                # ── Fase 0: Lectura y Decodificación (read_and_decode) ──────
                _t_read0 = time.perf_counter()
                ret, frame_bgr = reader.read()
                total_read_decode_time += time.perf_counter() - _t_read0
                if not ret or frame_bgr is None:
                    break
                if max_frames and frame_count >= max_frames:
                    break
                frame_count += 1


                ts = frame_count / fps

                # ── Fase 1: Inferencia (3 modelos) ──────────────────────────
                _t0 = time.perf_counter()
                annotated_rgb, minimap_rgb, H, player_dets, ball_det = self.process_frame(
                    frame_bgr, frame_idx=frame_count, timestamp_sec=ts, is_video=True, headless=is_headless
                )
                total_inference_time += time.perf_counter() - _t0

                # ── Fase 2: Tracking / Ball Tracker (incluido en process_frame pero separado aquí conceptualmente)
                # El tiempo de tracking se mide internamente; el resto de process_frame es homografía

                # ── Fase 3: Construcción del JSON por frame ──────────────────
                _t1 = time.perf_counter()
                frame_data = {
                    "frame": frame_count,
                    "timestamp": round(ts, 3),
                    "players": [
                        {
                            "track_id": p["player_id"],
                            "player_id": p["player_id"],
                            "role": p.get("role", "player"),
                            "team_id": p.get("team_id", 0),
                            "team_color": p.get("team_color", {}).get("hex", "#FFFFFF") if isinstance(p.get("team_color"), dict) else p.get("team_color", "#FFFFFF"),
                            "team_confidence": p.get("team_confidence", 1.0),
                            "tracking_confidence": p.get("confidence", 1.0),
                            "is_interpolated": p.get("is_interpolated", False),
                            "bbox": p.get("bbox"),
                            "pixel_x": p["pixel_x"],
                            "pixel_y": p["pixel_y"],
                            "pitch_x": p["pitch_x_m"],
                            "pitch_y": p["pitch_y_m"],
                        }
                        for p in player_dets
                    ],
                    "ball": {
                        "detected": True,
                        "confidence": ball_det.get("confidence", 1.0),
                        "bbox": ball_det.get("bbox"),
                        "pixel_x": ball_det["pixel_x"],
                        "pixel_y": ball_det["pixel_y"],
                        "pitch_x": ball_det["pitch_x_m"],
                        "pitch_y": ball_det["pitch_y_m"],
                        "is_interpolated": ball_det.get("interpolated", False),
                    } if ball_det is not None else None,
                    "homography_valid": bool(H is not None),
                }
                tracking_frames.append(frame_data)
                total_json_time += time.perf_counter() - _t1

                # ── Fase 4: Renderizado y escritura de video ─────────────────
                if not is_headless:
                    _t2 = time.perf_counter()
                    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                    minimap_bgr   = cv2.cvtColor(minimap_rgb,   cv2.COLOR_RGB2BGR)

                    h_f, w_f = annotated_bgr.shape[:2]
                    h_m, w_m = minimap_bgr.shape[:2]
                    scale_r   = h_f / float(h_m)
                    minimap_r = cv2.resize(minimap_bgr, (int(w_m * scale_r), h_f))
                    combined  = np.hstack([annotated_bgr, minimap_r])

                    if writer is None:
                        h_c, w_c = combined.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = AsyncVideoWriter(output_path, fourcc, fps, (w_c, h_c))

                    writer.write(combined)
                    total_video_time += time.perf_counter() - _t2

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_avg = frame_count / elapsed
                    pct     = frame_count / effective_total * 100 if effective_total > 0 else 0
                    bar_len = 30
                    filled  = int(bar_len * frame_count / effective_total) if effective_total > 0 else 0
                    bar     = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  [{bar}] {pct:5.1f}%  {frame_count}/{effective_total}  {fps_avg:.1f} FPS", end="", flush=True)

        finally:
            reader.release()
            if writer:
                writer.release()
            print()  # Salto de línea tras la barra de progreso

        # ── Exportar JSON Contract ───────────────────────────────────────────
        video_name = resolved.name
        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0
        actual_homography_json = homography_json_path or str(settings.OUTPUT_DIR / f"homography_{resolved.stem}.json")

        tracking_dataset = {
            "metadata": {
                "video_name": video_name,
                "fps": round(fps, 2),
                "total_frames": frame_count,
                "duration_seconds": duration_sec,
                "teams": self.team_detector.get_teams_metadata(),
                "tracking_config": {
                    "tracker": getattr(settings, "TRACKER_TYPE", "botsort"),
                    "player_stride": self.player_stride,
                },
                "inference": {
                    "device": settings.INFERENCE_DEVICE,
                    "model_format": settings.MODEL_FORMAT,
                    "cuda_available": settings.CUDA_AVAILABLE,
                    "environment": settings.ENVIRONMENT,
                },
                "performance": {
                    "read_and_decode_sec": round(total_read_decode_time, 4),
                    "reader_metrics": reader.get_metrics() if hasattr(reader, "get_metrics") else None,
                },
                "rejections_csv": actual_rejections_csv,
                "homography_json": actual_homography_json,
            },
            "frames": tracking_frames,
            "events": [],
        }


        actual_json_path = json_output_path or str(settings.OUTPUT_DIR / f"tracking_{resolved.stem}.json")
        json_path_obj = Path(actual_json_path)
        json_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path_obj, "w", encoding="utf-8") as f:
            json.dump(tracking_dataset, f, indent=2)
        logger.info(f"💾 Tracking JSON exportado: {actual_json_path}")

        # ── Parquet & MOT Export (Machine Learning / Deep Learning) ───────────
        actual_parquet_path = parquet_output_path or str(settings.OUTPUT_DIR / f"tracking_{resolved.stem}.parquet")
        try:
            from sports.ml_exporter import export_tracking_to_parquet, export_to_mot_challenge
            export_tracking_to_parquet(tracking_dataset, actual_parquet_path)
            if mot_output_path:
                export_to_mot_challenge(tracking_dataset, mot_output_path)
        except Exception as e:
            logger.warning(f"[SoccerAnalysisV5] Advertencia al exportar formato ML/Parquet: {e}")

        # ── CSV Export ────────────────────────────────────────────────────────
        if csv_output_path and all_events:
            csv_path = Path(csv_output_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx", "timestamp_sec", "timestamp_fmt", "event_type", "confidence"])
                for evt in all_events:
                    ts = evt.get("timestamp_sec", 0.0)
                    mins, secs = int(ts // 60), int(ts % 60)
                    w.writerow([evt.get("frame_idx"), ts,
                                f"{mins:02d}:{secs:02d}", evt.get("event_type"), evt.get("confidence")])
            logger.info(f"📄 CSV exportado: {csv_output_path}")

        total_time = time.time() - start_time
        fps_final  = frame_count / total_time if total_time > 0 else 0.0

        # Porcentajes sobre tiempo total para identificar cuellos de botella
        def _pct(t): return f"{t / total_time * 100:.1f}%" if total_time > 0 else "—"

        logger.info("=" * 65)
        logger.info("📊 RESUMEN FINAL DE PROCESAMIENTO")
        logger.info(f"   Modo:              {'HEADLESS (Sin MP4/Overlays)' if is_headless else 'FULL (Video + JSON)'}")
        logger.info(f"   Motor inferencia:  {settings.INFERENCE_DEVICE.upper()} | {settings.MODEL_FORMAT.upper()}")
        logger.info(f"   CUDA disponible:   {settings.CUDA_AVAILABLE}")
        logger.info(f"   Frames procesados: {frame_count}")
        logger.info(f"   Tiempo total:      {total_time:.2f} s")
        logger.info(f"   FPS promedio:      {fps_final:.2f}")
        logger.info("-" * 65)
        logger.info("⏱  DESGLOSE DE TIEMPOS POR FASE:")
        logger.info(f"   Total read & decode:{total_read_decode_time:.2f}s  ({_pct(total_read_decode_time)} del total)")
        if hasattr(reader, "get_metrics"):
            m = reader.get_metrics()
            logger.info(
                f"     └─ Worker decode:  {m['total_decode_time_s']:.2f}s "
                f"({m['avg_decode_time_ms']:.2f} ms/frame, {m['decode_fps']:.1f} FPS, wait={m['total_consumer_wait_s']:.3f}s)"
            )
        logger.info(f"   Total inferencia:  {total_inference_time:.2f}s  ({_pct(total_inference_time)} del total)")
        logger.info(f"   Total tracking:    {total_tracking_time:.2f}s  ({_pct(total_tracking_time)} del total)")
        logger.info(f"   Total homography:  {total_homography_time:.2f}s  ({_pct(total_homography_time)} del total)")
        logger.info(f"   Total json build:  {total_json_time:.2f}s  ({_pct(total_json_time)} del total)")
        logger.info(f"   Total video write: {total_video_time:.2f}s  ({_pct(total_video_time)} del total)")
        logger.info("-" * 65)
        logger.info(f"   JSON Output:       {actual_json_path}")
        logger.info(f"   Homography JSON:   {actual_homography_json}")
        logger.info(f"   Parquet ML Output: {actual_parquet_path}")
        if not is_headless:
            logger.info(f"   MP4 Output:        {output_path}")
        logger.info("=" * 65)

        # ── Estadísticas de Rechazos de Homografía y Exportación CSV ──────────
        rejection_summary = self.homography_tracker.get_rejection_summary()
        logger.info("\n" + rejection_summary)
        saved_rejections_csv = self.homography_tracker.save_rejections_to_csv(actual_rejections_csv)
        logger.info(f"📄 CSV de rechazos exportado: {saved_rejections_csv}")

        # ── Exportar Homography JSON (Detecciones, Puntos Correctos/Fallidos y Correcciones) ──
        audit_metadata = {
            "video_name": video_name,
            "fps": round(fps, 2),
            "total_frames": frame_count,
            "duration_seconds": duration_sec,
            "homography_interval": self.homography_interval,
            "rejections_csv": actual_rejections_csv,
        }
        saved_homography_json = self.homography_audit.save_to_json(actual_homography_json, metadata=audit_metadata)
        logger.info(f"📐 Homografía JSON exportado: {saved_homography_json}")

        if keypoints_csv_path:
            saved_keypoints_csv = self.homography_audit.save_to_csv(keypoints_csv_path)
            logger.info(f"📊 CSV de keypoints exportado: {saved_keypoints_csv}")

        logger.info("=" * 65)

        return all_events


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ScoutingData v5.0 — Soccer Analysis Pipeline (Direct + OpenVINO/CUDA)"
    )
    parser.add_argument("--video",  "-v", default=settings.DEFAULT_INPUT_VIDEO,  help="Video de entrada")
    parser.add_argument("--output", "-o", default=settings.DEFAULT_OUTPUT_VIDEO, help="Video de salida MP4")
    parser.add_argument("--csv",    "-c", default=settings.DEFAULT_OUTPUT_CSV,   help="CSV de eventos")
    parser.add_argument("--json",   "-j", default=None,                          help="JSON de tracking output")
    parser.add_argument("--parquet", "-p", default=None,                         help="Parquet de tracking output (ML/DL)")
    parser.add_argument("--mot",          default=None,                          help="MOTChallenge format output")
    parser.add_argument("--rejections-csv", default=None,                        help="CSV para exportar rechazos de homografía")
    parser.add_argument("--keypoints-csv",  default=None,                        help="CSV para exportar auditoría forense de keypoints")
    parser.add_argument("--homography-json", default=None,                        help="JSON para exportar auditoría de homografía (detecciones, puntos correctos, fallidos y correcciones)")
    parser.add_argument("--env",    "-e", choices=["LOCAL", "KAGGLE", "COLAB"],           help="Forzar entorno (LOCAL, KAGGLE, COLAB)")
    parser.add_argument("--headless", action="store_true", default=False,        help="Modo headless (Sin renderizar MP4/Overlays)")
    parser.add_argument("--with-video", dest="headless", action="store_false",   help="Forzar renderizado de video MP4 (activo por defecto)")
    parser.add_argument("--no-video", dest="headless", action="store_true",      help="Omitir renderizado de video MP4 para máxima velocidad")
    parser.add_argument("--max-frames", "-m", type=int, default=None,            help="Límite de frames")
    parser.add_argument("--homography-interval", type=int,
                        default=settings.HOMOGRAPHY_INTERVAL,
                        help="Recalcular H cada N frames (1=cada frame, 5=cada 5 frames)")
    parser.add_argument("--debug-homography", action="store_true", default=False,
                        help="Activar logs detallados de homografía e inferencia por frame")
    parser.add_argument("--disable-corner-shift", action="store_true", default=False,
                        help="Desactivar temporalmente verificación de deriva de esquinas (prueba A/B)")
    parser.add_argument("--disable-threaded-reader", action="store_true", default=False,
                        help="Desactivar ThreadedOpenCVReader y usar cv2.VideoCapture secuencial")
    parser.add_argument("--reader-buffer", type=int, default=64,
                        help="Tamaño de buffer de frames para ThreadedOpenCVReader (default: 64)")
    parser.add_argument("--player-stride", type=int, default=settings.PLAYER_INFERENCE_STRIDE,
                        help="Cadencia de inferencia de jugadores (1=todos los frames, 2=skip adaptativo)")
    parser.add_argument("--show-frame-num", action="store_true", default=None,
                        help="Forzar visualización del número de frame junto a Field Coverage")
    parser.add_argument("--hide-frame-num", action="store_true", default=False,
                        help="Ocultar badge de número de frame")
    args = parser.parse_args()

    if args.env:
        env_upper = args.env.upper()
        os.environ["ENVIRONMENT"] = env_upper
        settings.ENVIRONMENT = env_upper
        if env_upper in ("KAGGLE", "COLAB"):
            if settings.CUDA_AVAILABLE:
                settings.MODEL_FORMAT = "pt"
                settings.INFERENCE_DEVICE = "cuda"
            else:
                logger.warning(
                    f"⚠ Entorno {env_upper} seleccionado pero CUDA no está disponible en esta sesión. "
                    f"En Colab: Ve a 'Entorno de ejecución' -> 'Cambiar tipo de entorno de ejecución' -> Selecciona GPU (L4 o T4)."
                )
                settings.MODEL_FORMAT = "openvino" if Path(settings.MODEL_KEYPOINT_PATH).exists() else "pt"
                settings.INFERENCE_DEVICE = "cpu"
        elif env_upper == "LOCAL":
            settings.MODEL_FORMAT = "openvino"
            settings.INFERENCE_DEVICE = "cpu"
        settings.configure_gpu_optimizations()

    if args.headless:
        os.environ["HEADLESS_MODE"] = "True"
        settings.HEADLESS_MODE = True
    else:
        os.environ["HEADLESS_MODE"] = "False"
        settings.HEADLESS_MODE = False

    show_frame = False if args.hide_frame_num else (True if args.show_frame_num else None)

    logger.info("=" * 65)
    logger.info("  ScoutingData v5.0 — Direct Inference Pipeline")
    logger.info("=" * 65)
    for k, v in settings.get_summary().items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 65)

    pipeline = SoccerAnalysisV5(
        homography_interval=args.homography_interval,
        headless=args.headless,
        enable_corner_shift=not args.disable_corner_shift,
        use_threaded_reader=not args.disable_threaded_reader,
        reader_buffer_size=args.reader_buffer,
        player_stride=args.player_stride,
        show_frame_number=show_frame,
    )



    resolved = settings.resolve_video_source(args.video)
    if resolved.exists():
        pipeline.process_video(
            video_path=str(resolved),
            output_path=args.output,
            csv_output_path=args.csv,
            json_output_path=args.json,
            parquet_output_path=args.parquet,
            mot_output_path=args.mot,
            max_frames=args.max_frames,
            headless=args.headless,
            debug_homography=args.debug_homography,
            rejections_csv_path=args.rejections_csv,
            keypoints_csv_path=args.keypoints_csv,
            homography_json_path=args.homography_json,
            use_threaded_reader=not args.disable_threaded_reader,
            reader_buffer_size=args.reader_buffer,
        )
    else:
        logger.warning(f"Video no encontrado: {resolved}")
        logger.info("Usa --video <ruta> o define VIDEO_INPUT como variable de entorno.")


if __name__ == "__main__":
    main()
