"""
ScoutingData v5.2 - Registro Forense, Auditoría y Exportación JSON de Homografía
Exporta a formato estructurado JSON (y CSV opcional) la auditoría punto a punto:
  - Detecciones de homografía (matrices H candidata y efectiva, diagnóstico RANSAC/MAGSAC, status)
  - Puntos correctos (inliers RANSAC/MAGSAC, proyecciones imagen <-> terreno, errores métricos)
  - Puntos fallidos (outliers, truncados por borde, descartados por mitad opuesta, anomalías espaciales)
  - Correcciones por frame (suavizado temporal EMA, re-anclaje fast-recovery, holding por rechazo cinemático)

Nota: No emite logs intrusivos durante el procesamiento (sin saturar logs), solo exporta la telemetría a JSON.
"""

import csv
import json
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from config.field_points import FIELD_COORDS
from homography.mappings import KP_MAP
from homography.estimator import filter_keypoints_by_pitch_side
from homography.geometric_validator import filter_border_keypoints
from homography.diagnostics_tool import PointResidual, detect_spatial_anomalies_and_swaps

logger = logging.getLogger("homography.audit")

CSV_FIELDNAMES = [
    "frame_id",
    "timestamp_sec",
    "homography_status",
    "kp_idx",
    "kp_label",
    "confidence",
    "status_quality",
    "error_px",
    "error_m",
    "detected_x_px",
    "detected_y_px",
    "reprojected_x_px",
    "reprojected_y_px",
    "field_x_m",
    "field_y_m",
    "diagnostic_note",
]


def _sanitize_for_json(obj: Any) -> Any:
    """Convierte de forma recursiva arrays y tipos numéricos de NumPy a tipos estándar de Python."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


class HomographyAuditExporter:
    """
    Auditor forense y exportador JSON de homografía frame a frame.
    Recopila detalladamente detecciones de homografía, puntos correctos,
    puntos fallidos y correcciones cinemáticas sin emitir mensajes en el log.
    """

    def __init__(
        self,
        output_json_path: Optional[str] = None,
        output_csv_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        ransac_threshold: float = 1.5,
    ):
        self.output_json_path = output_json_path
        self.output_csv_path = output_csv_path
        self.conf_threshold = conf_threshold
        self.ransac_threshold = ransac_threshold
        self.frames_data: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, Any]] = []

        # Contadores globales de telemetría
        self.total_evaluated_points = 0
        self.total_good_inliers = 0
        self.total_outliers_ransac = 0
        self.total_discarded_side = 0
        self.total_discarded_border = 0
        self.total_swap_anomalies = 0
        self.total_low_conf = 0

        self._kp_stats: Dict[str, Dict[str, int]] = {}

    def reset(self):
        """Reinicia los buffers de datos acumulados."""
        self.frames_data.clear()
        self.rows.clear()
        self.total_evaluated_points = 0
        self.total_good_inliers = 0
        self.total_outliers_ransac = 0
        self.total_discarded_side = 0
        self.total_discarded_border = 0
        self.total_swap_anomalies = 0
        self.total_low_conf = 0
        self._kp_stats.clear()

    def record_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        kp_result: Any,
        homography_status: str,
        H_effective: Optional[np.ndarray],
        H_candidate: Optional[np.ndarray] = None,
        diagnostics: Optional[Any] = None,
        residuals: Optional[List[PointResidual]] = None,
        compute_error_reason: Optional[str] = None,
        compute_error_details: Optional[str] = None,
        transition_metrics: Optional[Dict[str, Any]] = None,
        conf_threshold: float = 0.35,
        img_width: int = 1920,
        img_height: int = 1080,
    ):
        """
        Registra la auditoría completa de un frame sin generar logs por consola.
        """
        ts = round(float(timestamp_sec), 3)

        # ── 1. Detecciones de Homografía ──────────────────────────────────────
        H_eff_list = None
        if H_effective is not None and np.all(np.isfinite(H_effective)):
            h22 = H_effective[2, 2] if abs(H_effective[2, 2]) > 1e-9 else 1.0
            H_eff_norm = H_effective / h22
            H_eff_list = [[round(float(v), 6) for v in row] for row in H_eff_norm]

        H_cand_list = None
        if H_candidate is not None and np.all(np.isfinite(H_candidate)):
            h22_c = H_candidate[2, 2] if abs(H_candidate[2, 2]) > 1e-9 else 1.0
            H_cand_norm = H_candidate / h22_c
            H_cand_list = [[round(float(v), 6) for v in row] for row in H_cand_norm]

        diag_dict = None
        if diagnostics is not None:
            diag_dict = {
                "mean_reprojection_error_m": round(float(getattr(diagnostics, "mean_reprojection_error_m", 0.0)), 3),
                "max_reprojection_error_m": round(float(getattr(diagnostics, "max_reprojection_error_m", 0.0)), 3),
                "inliers_count": int(getattr(diagnostics, "ransac_inliers", 0)),
                "outliers_count": int(getattr(diagnostics, "ransac_outliers", 0)),
                "inlier_ratio": round(float(getattr(diagnostics, "inlier_ratio", 0.0)), 3),
                "field_coverage_ratio": round(float(getattr(diagnostics, "field_coverage_ratio", 0.0)), 4),
                "hull_area_field_m2": round(float(getattr(diagnostics, "hull_area_field_m2", 0.0)), 1),
                "collinearity_score": round(float(getattr(diagnostics, "collinearity_score", 0.0)), 3),
                "condition_number": round(float(getattr(diagnostics, "condition_number", 0.0)), 1),
            }

        homography_data = {
            "detected": bool(H_candidate is not None),
            "valid": bool(H_effective is not None),
            "status": homography_status,
            "matrix_effective": H_eff_list,
            "matrix_candidate": H_cand_list,
            "diagnostics": diag_dict,
        }

        # ── 2. Puntos Correctos y Puntos Fallidos ────────────────────────────
        puntos_correctos: List[Dict[str, Any]] = []
        puntos_fallidos: List[Dict[str, Any]] = []

        # Caso en que YOLO no devuelve keypoints
        has_keypoints = (
            kp_result is not None
            and getattr(kp_result, "keypoints", None) is not None
            and len(kp_result.keypoints) > 0
            and getattr(kp_result.keypoints, "xy", None) is not None
            and len(kp_result.keypoints.xy) > 0
        )

        if not has_keypoints:
            fail_reason = compute_error_reason or "NO_DETECTIONS"
            fail_details = f"YOLO no detectó keypoints en este frame. Causa H: {compute_error_reason or 'SIN_DATOS'}"
            puntos_fallidos.append({
                "kp_idx": -1,
                "kp_label": "NONE",
                "confidence": 0.0,
                "detected_px": None,
                "reprojected_px": None,
                "field_coords_m": None,
                "reprojected_m": None,
                "error_px": None,
                "error_m": None,
                "reason": fail_reason,
                "details": fail_details,
            })
            self.rows.append({
                "frame_id": frame_idx,
                "timestamp_sec": ts,
                "homography_status": homography_status,
                "kp_idx": -1,
                "kp_label": "NONE",
                "confidence": 0.0,
                "status_quality": "MALO_NO_DETECTIONS",
                "error_px": "",
                "error_m": "",
                "detected_x_px": "",
                "detected_y_px": "",
                "reprojected_x_px": "",
                "reprojected_y_px": "",
                "field_x_m": "",
                "field_y_m": "",
                "diagnostic_note": fail_details,
            })
        else:
            raw_xy = kp_result.keypoints.xy[0]
            kps_xy = raw_xy.cpu().numpy() if hasattr(raw_xy, "cpu") else np.asarray(raw_xy)
            raw_conf = kp_result.keypoints.conf[0]
            kps_conf = raw_conf.cpu().numpy() if hasattr(raw_conf, "cpu") else np.asarray(raw_conf)

            if len(kps_xy) == 0:
                puntos_fallidos.append({
                    "kp_idx": -1,
                    "kp_label": "NONE",
                    "confidence": 0.0,
                    "detected_px": None,
                    "reprojected_px": None,
                    "field_coords_m": None,
                    "reprojected_m": None,
                    "error_px": None,
                    "error_m": None,
                    "reason": "EMPTY_ARRAY",
                    "details": "Array de keypoints vacío.",
                })
            else:
                # 1. Filtro de lado dominante de campo
                valid_side_cands, disc_l, disc_r = filter_keypoints_by_pitch_side(
                    kps_xy, kps_conf, conf_threshold=conf_threshold
                )
                kept_side_idxs = {c[0] for c in valid_side_cands}

                # 2. Filtro de truncamiento por bordes de pantalla
                valid_border_cands = filter_border_keypoints(
                    valid_side_cands, img_width=img_width, img_height=img_height, border_margin_px=4.0
                )
                kept_border_idxs = {c[0] for c in valid_border_cands}

                # 3. Residuales RANSAC / MAGSAC
                res_map: Dict[int, PointResidual] = {}
                anomaly_labels = set()
                if residuals:
                    res_map = {r.kp_idx: r for r in residuals}
                    anomalies = detect_spatial_anomalies_and_swaps(residuals)
                    for a in anomalies:
                        for anom_lbl in a.get("labels", []):
                            anomaly_labels.add(anom_lbl)

                # 4. Matriz H para evaluación de proyección
                H_eval = H_candidate if (H_candidate is not None and np.all(np.isfinite(H_candidate))) else H_effective
                H_eval_inv = None
                if H_eval is not None:
                    try:
                        H_eval_inv = np.linalg.inv(H_eval)
                    except np.linalg.LinAlgError:
                        H_eval_inv = None

                detected_pts_count = 0

                for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
                    conf_val = float(conf)
                    u, v = float(xy[0]), float(xy[1])
                    lbl = KP_MAP.get(idx, f"KP_{idx:02d}")
                    fld_pt = FIELD_COORDS.get(lbl)

                    if lbl not in self._kp_stats:
                        self._kp_stats[lbl] = {"total": 0, "inliers": 0, "outliers": 0}

                    # Sub-umbral mínimo absoluto (0.15)
                    if conf_val < 0.15:
                        puntos_fallidos.append({
                            "kp_idx": idx,
                            "kp_label": lbl,
                            "confidence": round(conf_val, 3),
                            "detected_px": [round(u, 1), round(v, 1)],
                            "reprojected_px": None,
                            "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)] if fld_pt else None,
                            "reprojected_m": None,
                            "error_px": None,
                            "error_m": None,
                            "reason": "LOW_CONFIDENCE_SUBTHRESHOLD",
                            "details": f"Confianza muy baja ({conf_val:.2f} < 0.15)",
                        })
                        self.total_low_conf += 1
                        continue

                    detected_pts_count += 1
                    self.total_evaluated_points += 1
                    self._kp_stats[lbl]["total"] += 1

                    # Cálculo proyectivo
                    err_m_val: Optional[float] = None
                    err_px_val: Optional[float] = None
                    proj_img_pair: Optional[List[float]] = None
                    proj_fld_pair: Optional[List[float]] = None

                    if H_eval is not None and fld_pt is not None:
                        try:
                            pt_img = np.array([u, v], dtype=np.float32).reshape(1, 1, 2)
                            p_fld = cv2.perspectiveTransform(pt_img, H_eval)[0, 0]
                            err_m_val = float(np.linalg.norm(p_fld - np.array(fld_pt, dtype=np.float32)))
                            proj_fld_pair = [round(float(p_fld[0]), 2), round(float(p_fld[1]), 2)]

                            if H_eval_inv is not None:
                                pt_fld = np.array(fld_pt, dtype=np.float32).reshape(1, 1, 2)
                                p_img = cv2.perspectiveTransform(pt_fld, H_eval_inv)[0, 0]
                                err_px_val = float(np.linalg.norm(p_img - np.array([u, v], dtype=np.float32)))
                                proj_img_pair = [round(float(p_img[0]), 1), round(float(p_img[1]), 1)]
                        except Exception:
                            pass

                    # ── Clasificación rigurosa de calidad ──
                    if fld_pt is None:
                        reason = "UNMAPPED"
                        note = "Keypoint no mapeado en modelo de campo FIELD_COORDS"
                        puntos_fallidos.append({
                            "kp_idx": idx,
                            "kp_label": lbl,
                            "confidence": round(conf_val, 3),
                            "detected_px": [round(u, 1), round(v, 1)],
                            "reprojected_px": None,
                            "field_coords_m": None,
                            "reprojected_m": None,
                            "error_px": None,
                            "error_m": None,
                            "reason": reason,
                            "details": note,
                        })
                        self._kp_stats[lbl]["outliers"] += 1
                        status_quality = "MALO_NO_MAP"

                    elif conf_val < conf_threshold:
                        reason = "LOW_CONFIDENCE"
                        note = f"Confianza baja ({conf_val:.2f} < {conf_threshold:.2f})"
                        puntos_fallidos.append({
                            "kp_idx": idx,
                            "kp_label": lbl,
                            "confidence": round(conf_val, 3),
                            "detected_px": [round(u, 1), round(v, 1)],
                            "reprojected_px": proj_img_pair,
                            "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)],
                            "reprojected_m": proj_fld_pair,
                            "error_px": round(err_px_val, 2) if err_px_val is not None else None,
                            "error_m": round(err_m_val, 3) if err_m_val is not None else None,
                            "reason": reason,
                            "details": note,
                        })
                        self.total_low_conf += 1
                        self._kp_stats[lbl]["outliers"] += 1
                        status_quality = "MALO_LOW_CONF"

                    elif idx not in kept_side_idxs:
                        side_str = "arco izquierdo" if disc_l else "arco derecho"
                        reason = "DISCARDED_PITCH_SIDE"
                        note = f"Descartado: mitad dominante opuesta ({side_str})"
                        puntos_fallidos.append({
                            "kp_idx": idx,
                            "kp_label": lbl,
                            "confidence": round(conf_val, 3),
                            "detected_px": [round(u, 1), round(v, 1)],
                            "reprojected_px": proj_img_pair,
                            "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)],
                            "reprojected_m": proj_fld_pair,
                            "error_px": round(err_px_val, 2) if err_px_val is not None else None,
                            "error_m": round(err_m_val, 3) if err_m_val is not None else None,
                            "reason": reason,
                            "details": note,
                        })
                        self.total_discarded_side += 1
                        self._kp_stats[lbl]["outliers"] += 1
                        status_quality = "MALO_DISCARDED_SIDE"

                    elif idx not in kept_border_idxs:
                        reason = "DISCARDED_SCREEN_BORDER"
                        note = "Descartado: truncado en borde de pantalla (<4px)"
                        puntos_fallidos.append({
                            "kp_idx": idx,
                            "kp_label": lbl,
                            "confidence": round(conf_val, 3),
                            "detected_px": [round(u, 1), round(v, 1)],
                            "reprojected_px": proj_img_pair,
                            "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)],
                            "reprojected_m": proj_fld_pair,
                            "error_px": round(err_px_val, 2) if err_px_val is not None else None,
                            "error_m": round(err_m_val, 3) if err_m_val is not None else None,
                            "reason": reason,
                            "details": note,
                        })
                        self.total_discarded_border += 1
                        self._kp_stats[lbl]["outliers"] += 1
                        status_quality = "MALO_DISCARDED_BORDER"

                    elif idx in res_map:
                        pr = res_map[idx]
                        err_px_val = pr.error_px
                        err_m_val = pr.error_m
                        proj_img_pair = [round(float(pr.reprojected_px[0]), 1), round(float(pr.reprojected_px[1]), 1)]
                        proj_fld_pair = [round(float(pr.reprojected_m[0]), 2), round(float(pr.reprojected_m[1]), 2)]

                        if lbl in anomaly_labels:
                            reason = "SPATIAL_ANOMALY"
                            note = f"Sospecha de swap o inversión de orden espacial (err={err_m_val:.2f}m)"
                            puntos_fallidos.append({
                                "kp_idx": idx,
                                "kp_label": lbl,
                                "confidence": round(conf_val, 3),
                                "detected_px": [round(u, 1), round(v, 1)],
                                "reprojected_px": proj_img_pair,
                                "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)],
                                "reprojected_m": proj_fld_pair,
                                "error_px": round(err_px_val, 2),
                                "error_m": round(err_m_val, 3),
                                "reason": reason,
                                "details": note,
                            })
                            self.total_swap_anomalies += 1
                            self._kp_stats[lbl]["outliers"] += 1
                            status_quality = "MALO_SWAP_ANOMALY"

                        elif pr.is_inlier:
                            puntos_correctos.append({
                                "kp_idx": idx,
                                "kp_label": lbl,
                                "confidence": round(conf_val, 3),
                                "detected_px": [round(u, 1), round(v, 1)],
                                "reprojected_px": proj_img_pair,
                                "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)],
                                "reprojected_m": proj_fld_pair,
                                "error_px": round(err_px_val, 2),
                                "error_m": round(err_m_val, 3),
                                "status": "INLIER",
                            })
                            self.total_good_inliers += 1
                            self._kp_stats[lbl]["inliers"] += 1
                            status_quality = "BUENO_INLIER"
                            note = f"Inlier RANSAC/MAGSAC válido (err={err_m_val:.2f}m)"

                        else:
                            reason = "OUTLIER_RANSAC"
                            note = f"Outlier descartado por USAC_MAGSAC (err={err_m_val:.2f}m > umbral)"
                            puntos_fallidos.append({
                                "kp_idx": idx,
                                "kp_label": lbl,
                                "confidence": round(conf_val, 3),
                                "detected_px": [round(u, 1), round(v, 1)],
                                "reprojected_px": proj_img_pair,
                                "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)],
                                "reprojected_m": proj_fld_pair,
                                "error_px": round(err_px_val, 2),
                                "error_m": round(err_m_val, 3),
                                "reason": reason,
                                "details": note,
                            })
                            self.total_outliers_ransac += 1
                            self._kp_stats[lbl]["outliers"] += 1
                            status_quality = "MALO_OUTLIER_RANSAC"

                    else:
                        # Candidato evaluado sin residuales de findHomography
                        if err_m_val is not None and err_m_val <= 1.5:
                            puntos_correctos.append({
                                "kp_idx": idx,
                                "kp_label": lbl,
                                "confidence": round(conf_val, 3),
                                "detected_px": [round(u, 1), round(v, 1)],
                                "reprojected_px": proj_img_pair,
                                "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)],
                                "reprojected_m": proj_fld_pair,
                                "error_px": round(err_px_val, 2) if err_px_val is not None else None,
                                "error_m": round(err_m_val, 3),
                                "status": "COHERENT_PREV_H",
                            })
                            self.total_good_inliers += 1
                            self._kp_stats[lbl]["inliers"] += 1
                            status_quality = "BUENO_COHERENTE"
                            note = f"Coherente con H previa (err={err_m_val:.2f}m)"
                        else:
                            reason = "HIGH_DISCREPANCY" if (err_m_val is not None and err_m_val > 1.5) else "HOMOGRAPHY_FAILED"
                            note = (
                                f"Discrepancia alta con H previa (err={err_m_val:.2f}m > 1.5m)"
                                if (err_m_val is not None and err_m_val > 1.5)
                                else f"H no calculada: {compute_error_reason or 'FALLO_CALCULO'} ({compute_error_details or ''})"
                            )
                            puntos_fallidos.append({
                                "kp_idx": idx,
                                "kp_label": lbl,
                                "confidence": round(conf_val, 3),
                                "detected_px": [round(u, 1), round(v, 1)],
                                "reprojected_px": proj_img_pair,
                                "field_coords_m": [round(float(fld_pt[0]), 2), round(float(fld_pt[1]), 2)],
                                "reprojected_m": proj_fld_pair,
                                "error_px": round(err_px_val, 2) if err_px_val is not None else None,
                                "error_m": round(err_m_val, 3) if err_m_val is not None else None,
                                "reason": reason,
                                "details": note,
                            })
                            self.total_outliers_ransac += 1
                            self._kp_stats[lbl]["outliers"] += 1
                            status_quality = "MALO_DEFORMACION" if reason == "HIGH_DISCREPANCY" else "MALO_H_FALLO"

                    # Almacenar en filas CSV para compatibilidad
                    self.rows.append({
                        "frame_id": frame_idx,
                        "timestamp_sec": ts,
                        "homography_status": homography_status,
                        "kp_idx": idx,
                        "kp_label": lbl,
                        "confidence": round(conf_val, 3),
                        "status_quality": status_quality,
                        "error_px": round(err_px_val, 2) if err_px_val is not None else "",
                        "error_m": round(err_m_val, 3) if err_m_val is not None else "",
                        "detected_x_px": round(u, 1),
                        "detected_y_px": round(v, 1),
                        "reprojected_x_px": proj_img_pair[0] if proj_img_pair else "",
                        "reprojected_y_px": proj_img_pair[1] if proj_img_pair else "",
                        "field_x_m": f"{fld_pt[0]:.2f}" if fld_pt else "",
                        "field_y_m": f"{fld_pt[1]:.2f}" if fld_pt else "",
                        "diagnostic_note": note,
                    })

                if detected_pts_count == 0:
                    fail_reason = "LOW_CONF_ALL"
                    fail_details = f"Ningún keypoint superó conf=0.15. Causa H: {compute_error_reason or 'INSUFICIENTE'}"
                    puntos_fallidos.append({
                        "kp_idx": -1,
                        "kp_label": "NONE",
                        "confidence": 0.0,
                        "detected_px": None,
                        "reprojected_px": None,
                        "field_coords_m": None,
                        "reprojected_m": None,
                        "error_px": None,
                        "error_m": None,
                        "reason": fail_reason,
                        "details": fail_details,
                    })

        # ── 3. Correcciones por Frame ─────────────────────────────────────────
        trans = transition_metrics or {}
        streak = int(trans.get("reject_streak", 0))
        reject_r = trans.get("reject_reason") or compute_error_reason
        reject_d = trans.get("reject_details") or compute_error_details

        correction_applied = False
        correction_type = "NONE"
        correction_details = ""

        if homography_status == "ACCEPTED_SMOOTHED":
            correction_applied = True
            correction_type = "TEMPORAL_SMOOTHING"
            alpha = trans.get("smoothing_alpha")
            alpha_str = f"alpha={alpha:.2f}, " if alpha is not None else ""
            shift_val = trans.get("center_shift_m")
            shift_str = f"shift={shift_val:.2f}m" if shift_val is not None else ""
            correction_details = f"Suavizado adaptativo EMA aplicado ({alpha_str}{shift_str})"
        elif homography_status == "REJECTED_HELD":
            correction_applied = True
            correction_type = "HOLD_PREVIOUS_ON_REJECTION"
            detail_extra = f" ({reject_d})" if reject_d else ""
            correction_details = f"Candidato rechazado por {reject_r or 'ESTABILIDAD'}{detail_extra}. Se sostuvo la homografía previa (streak={streak})."
        elif homography_status == "FAST_RECOVERY_ACCEPTED":
            correction_applied = True
            correction_type = "FAST_RECOVERY_REANCHOR"
            correction_details = "Recuperación rápida (Fast-Recovery): re-anclaje inmediato en candidato de alta confianza geométrica."
        elif homography_status == "RESET_ACCEPTED":
            correction_applied = True
            correction_type = "STREAK_RESET_REANCHOR"
            correction_details = f"Auto-reset tras streak de rechazos. Re-anclando en nueva posición de cámara."
        elif homography_status == "HELD_INTERVAL":
            correction_applied = False
            correction_type = "INTERVAL_HOLD"
            correction_details = "Homografía sostenida por intervalo de muestreo temporal."
        elif homography_status == "INITIAL_ACCEPTED":
            correction_applied = False
            correction_type = "INITIAL"
            correction_details = "Primera homografía válida aceptada."
        elif homography_status in ("FAILED_NO_PREV", "RESET_EMPTY"):
            correction_applied = False
            correction_type = "FAILED_NO_PREV"
            correction_details = f"Sin homografía previa válida: {reject_r or 'SIN_DATOS'}."
        else:
            correction_applied = False
            correction_type = homography_status
            correction_details = f"Estado: {homography_status}"

        correcciones_data = {
            "correction_applied": correction_applied,
            "correction_type": correction_type,
            "reject_streak": streak,
            "reject_reason": reject_r,
            "transition_metrics": {
                "center_shift_m": round(float(trans["center_shift_m"]), 2) if "center_shift_m" in trans and trans["center_shift_m"] is not None else None,
                "corner_shift_m": round(float(trans["max_corner_shift_m"]), 2) if "max_corner_shift_m" in trans and trans["max_corner_shift_m"] is not None else (round(float(trans["corner_shift_m"]), 2) if "corner_shift_m" in trans and trans["corner_shift_m"] is not None else None),
                "rotation_deg": round(float(trans["rotation_deg"]), 2) if "rotation_deg" in trans and trans["rotation_deg"] is not None else None,
                "scale_ratio": round(float(trans["scale_ratio"]), 3) if "scale_ratio" in trans and trans["scale_ratio"] is not None else None,
                "shear_ratio": round(float(trans["shear_ratio"]), 3) if "shear_ratio" in trans and trans["shear_ratio"] is not None else None,
                "smoothing_alpha": round(float(trans["smoothing_alpha"]), 3) if "smoothing_alpha" in trans and trans["smoothing_alpha"] is not None else None,
            },
            "thresholds": {
                "max_center_shift_m": round(float(trans["effective_max_center_shift_m"]), 2) if "effective_max_center_shift_m" in trans and trans["effective_max_center_shift_m"] is not None else None,
                "max_corner_shift_m": round(float(trans["effective_max_corner_shift_m"]), 2) if "effective_max_corner_shift_m" in trans and trans["effective_max_corner_shift_m"] is not None else None,
                "max_rotation_deg": round(float(trans["effective_max_rot_deg"]), 2) if "effective_max_rot_deg" in trans and trans["effective_max_rot_deg"] is not None else None,
                "allowed_scale_range": trans.get("allowed_scale_range"),
                "max_shear_ratio": round(float(trans["max_shear_ratio"]), 2) if "max_shear_ratio" in trans and trans["max_shear_ratio"] is not None else None,
            },
            "details": correction_details,
        }

        frame_record = {
            "frame": frame_idx,
            "timestamp": ts,
            "homography": homography_data,
            "puntos_correctos": puntos_correctos,
            "puntos_fallidos": puntos_fallidos,
            "correcciones": correcciones_data,
        }
        self.frames_data.append(frame_record)

    def log_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        kp_result: Any,
        homography_status: str,
        H_effective: Optional[np.ndarray],
        H_candidate: Optional[np.ndarray] = None,
        residuals: Optional[List[PointResidual]] = None,
        compute_error_reason: Optional[str] = None,
        compute_error_details: Optional[str] = None,
        conf_threshold: float = 0.35,
        img_width: int = 1920,
        img_height: int = 1080,
    ):
        """Método de retrocompatibilidad con la API de KeypointAuditLogger."""
        self.record_frame(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            kp_result=kp_result,
            homography_status=homography_status,
            H_effective=H_effective,
            H_candidate=H_candidate,
            diagnostics=None,
            residuals=residuals,
            compute_error_reason=compute_error_reason,
            compute_error_details=compute_error_details,
            transition_metrics=None,
            conf_threshold=conf_threshold,
            img_width=img_width,
            img_height=img_height,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Calcula el resumen analítico global de telemetría de homografía y keypoints."""
        total_frames = len(self.frames_data)
        frames_valid_h = sum(1 for f in self.frames_data if f["homography"]["valid"])
        frames_detected_h = sum(1 for f in self.frames_data if f["homography"]["detected"])
        frames_corrections = sum(1 for f in self.frames_data if f["correcciones"]["correction_applied"])
        total_correct = sum(len(f["puntos_correctos"]) for f in self.frames_data)
        total_failed = sum(len(f["puntos_fallidos"]) for f in self.frames_data)

        rejections_count: Dict[str, int] = {}
        for f in self.frames_data:
            rr = f["correcciones"].get("reject_reason")
            if rr:
                rejections_count[rr] = rejections_count.get(rr, 0) + 1

        return {
            "total_frames_evaluated": total_frames,
            "frames_with_candidate_detected": frames_detected_h,
            "frames_with_valid_homography": frames_valid_h,
            "frames_with_corrections_applied": frames_corrections,
            "total_correct_points": total_correct,
            "total_failed_points": total_failed,
            "rejections_by_reason": rejections_count,
        }

    def save_to_json(self, filepath: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Exporta el conjunto de datos completo a un archivo JSON sin emitir logs de puntos por consola.
        """
        target = Path(filepath or self.output_json_path or "outputs/homography_audit.json")
        target.parent.mkdir(parents=True, exist_ok=True)

        full_metadata: Dict[str, Any] = {
            "export_timestamp": datetime.datetime.now().isoformat(),
            "summary": self.get_summary(),
        }
        if metadata:
            for k, v in metadata.items():
                if k != "summary":
                    full_metadata[k] = v
            # Resumen recalculado actualizado
            full_metadata["summary"] = self.get_summary()

        export_data = {
            "metadata": full_metadata,
            "frames": self.frames_data,
        }

        sanitized_data = _sanitize_for_json(export_data)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(sanitized_data, f, indent=2, ensure_ascii=False)

        return str(target.resolve())

    def save_to_csv(self, target_path: Optional[str] = None) -> str:
        """
        Exporta todas las filas acumuladas al archivo CSV destino (compatibilidad tabular).
        """
        csv_path = Path(target_path or self.output_csv_path or "outputs/keypoints_audit.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)

        return str(csv_path.resolve())

    def get_summary_report(self) -> str:
        """
        Genera un resumen analítico en texto de los keypoints más ruidosos y tasas de éxito.
        """
        inlier_pct = (self.total_good_inliers / max(1, self.total_evaluated_points)) * 100.0
        outlier_pct = (self.total_outliers_ransac / max(1, self.total_evaluated_points)) * 100.0

        lines = [
            "=" * 65,
            "📋 RESUMEN DE AUDITORÍA FORENSE DE KEYPOINTS (CALIDAD H)",
            "=" * 65,
            f"   Total puntos evaluados:        {self.total_evaluated_points}",
            f"   Inliers válidos (BUENOS):      {self.total_good_inliers} ({inlier_pct:.1f}%)",
            f"   Outliers RANSAC (MALOS):       {self.total_outliers_ransac} ({outlier_pct:.1f}%)",
            f"   Descartados por mitad campo:   {self.total_discarded_side}",
            f"   Descartados por borde (<4px):  {self.total_discarded_border}",
            f"   Anomalías espaciales / Swaps:  {self.total_swap_anomalies}",
            "-" * 65,
            "   Top Keypoints con mayor tasa de error / outliers:",
        ]

        noisy_kps = []
        for lbl, stats in self._kp_stats.items():
            tot = stats["total"]
            if tot >= 5:
                fail_rate = stats["outliers"] / tot
                noisy_kps.append((lbl, fail_rate, stats["outliers"], tot))

        noisy_kps.sort(key=lambda x: x[1], reverse=True)
        for rank, (lbl, fr, outl, tot) in enumerate(noisy_kps[:5], start=1):
            lines.append(f"     {rank}. Keypoint '{lbl}': {fr*100:.1f}% outliers ({outl}/{tot} descartados)")

        lines.append("=" * 65)
        return "\n".join(lines)


# Alias para retrocompatibilidad completa
KeypointAuditLogger = HomographyAuditExporter
