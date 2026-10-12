"""
ScoutingData v5.2 - Registro Forense y Auditoría Punto a Punto de Keypoints
Genera un log detallado (CSV) de cada keypoint detectado por frame, indicando:
  - Cuáles fueron buenos (inliers RANSAC/MAGSAC con bajo error métrico)
  - Cuáles fueron malos (outliers, alucinaciones de arco opuesto, truncados por borde, o swaps de identidad)
  - Métricas de error en píxeles y en metros de cancha
  - Diagnóstico exacto para auditar frames específicos donde la homografía se malogra (ej: Frame 603).
"""

import csv
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


class KeypointAuditLogger:
    """
    Registrador forense de keypoints frame a frame para análisis de calidad
    y auditoría de fallos de homografía tanto en local como en la nube (Kaggle).
    """

    def __init__(self, output_csv_path: Optional[str] = None):
        self.output_csv_path = output_csv_path
        self.rows: List[Dict[str, Any]] = []

        # Contadores globales de telemetría
        self.total_evaluated = 0
        self.total_good_inliers = 0
        self.total_outliers_ransac = 0
        self.total_discarded_side = 0
        self.total_discarded_border = 0
        self.total_swap_anomalies = 0
        self.total_low_conf = 0

        # Historial de fallos por keypoint label para ranking de nocividad
        self._kp_stats: Dict[str, Dict[str, int]] = {}

    def log_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        kp_result,
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
        """
        Registra cada keypoint del frame analizando su validez geométrica y métrica.
        """
        ts = round(float(timestamp_sec), 3)

        if (
            kp_result is None
            or kp_result.keypoints is None
            or len(kp_result.keypoints) == 0
            or kp_result.keypoints.xy is None
            or len(kp_result.keypoints.xy) == 0
        ):
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
                "diagnostic_note": f"YOLO no detectó keypoints en este frame. Razón H: {compute_error_reason or 'SIN_DATOS'}",
            })
            return

        raw_xy = kp_result.keypoints.xy[0]
        kps_xy = raw_xy.cpu().numpy() if hasattr(raw_xy, "cpu") else np.asarray(raw_xy)
        raw_conf = kp_result.keypoints.conf[0]
        kps_conf = raw_conf.cpu().numpy() if hasattr(raw_conf, "cpu") else np.asarray(raw_conf)

        if len(kps_xy) == 0:
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
                "diagnostic_note": "Array de keypoints vacío.",
            })
            return

        # 1. Filtro de lado dominante de cancha
        valid_side_cands, disc_l, disc_r = filter_keypoints_by_pitch_side(
            kps_xy, kps_conf, conf_threshold=conf_threshold
        )
        kept_side_idxs = {c[0] for c in valid_side_cands}

        # 2. Filtro de truncamiento por bordes de pantalla
        valid_border_cands = filter_border_keypoints(
            valid_side_cands, img_width=img_width, img_height=img_height, border_margin_px=4.0
        )
        kept_border_idxs = {c[0] for c in valid_border_cands}

        # 3. Residuales RANSAC / MAGSAC (si compute_homography tuvo éxito)
        res_map: Dict[int, PointResidual] = {}
        anomaly_labels = set()
        if residuals:
            res_map = {r.kp_idx: r for r in residuals}
            anomalies = detect_spatial_anomalies_and_swaps(residuals)
            for a in anomalies:
                for lbl in a.get("labels", []):
                    anomaly_labels.add(lbl)

        # 4. Matriz H para evaluación proyectiva
        H_eval = H_candidate if (H_candidate is not None and np.all(np.isfinite(H_candidate))) else H_effective
        H_eval_inv = None
        if H_eval is not None:
            try:
                H_eval_inv = np.linalg.inv(H_eval)
            except np.linalg.LinAlgError:
                H_eval_inv = None

        detected_count = 0

        for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
            conf_val = float(conf)
            # Evaluar puntos con confianza mínima
            if conf_val < 0.15:
                continue

            detected_count += 1
            self.total_evaluated += 1

            lbl = KP_MAP.get(idx, f"KP_{idx:02d}")
            fld_pt = FIELD_COORDS.get(lbl)
            u, v = float(xy[0]), float(xy[1])

            if lbl not in self._kp_stats:
                self._kp_stats[lbl] = {"total": 0, "inliers": 0, "outliers": 0}
            self._kp_stats[lbl]["total"] += 1

            # Proyección para calcular errores
            err_m_val = None
            err_px_val = None
            proj_img_x = ""
            proj_img_y = ""
            fld_x_str = f"{fld_pt[0]:.2f}" if fld_pt else ""
            fld_y_str = f"{fld_pt[1]:.2f}" if fld_pt else ""

            if H_eval is not None and fld_pt is not None:
                try:
                    pt_img = np.array([u, v], dtype=np.float32).reshape(1, 1, 2)
                    proj_fld = cv2.perspectiveTransform(pt_img, H_eval)[0, 0]
                    err_m_val = float(np.linalg.norm(proj_fld - np.array(fld_pt, dtype=np.float32)))

                    if H_eval_inv is not None:
                        pt_fld = np.array(fld_pt, dtype=np.float32).reshape(1, 1, 2)
                        proj_img = cv2.perspectiveTransform(pt_fld, H_eval_inv)[0, 0]
                        err_px_val = float(np.linalg.norm(proj_img - np.array([u, v], dtype=np.float32)))
                        proj_img_x = f"{proj_img[0]:.1f}"
                        proj_img_y = f"{proj_img[1]:.1f}"
                except Exception:
                    pass

            # ── Clasificación de Calidad del Keypoint ──
            if fld_pt is None:
                status_quality = "MALO_NO_MAP"
                note = f"Keypoint no mapeado en FIELD_COORDS"
                self.total_low_conf += 1
            elif conf_val < conf_threshold:
                status_quality = "MALO_LOW_CONF"
                note = f"Confianza baja ({conf_val:.2f} < {conf_threshold:.2f})"
                self.total_low_conf += 1
            elif idx not in kept_side_idxs:
                status_quality = "MALO_DISCARDED_SIDE"
                side_str = "arco izquierdo" if disc_l else "arco derecho"
                note = f"Descartado: mitad dominante opuesta ({side_str})"
                self.total_discarded_side += 1
                self._kp_stats[lbl]["outliers"] += 1
            elif idx not in kept_border_idxs:
                status_quality = "MALO_DISCARDED_BORDER"
                note = "Descartado: truncado en borde de pantalla (<4px)"
                self.total_discarded_border += 1
                self._kp_stats[lbl]["outliers"] += 1
            elif idx in res_map:
                pr = res_map[idx]
                err_px_val = pr.error_px
                err_m_val = pr.error_m
                proj_img_x = f"{pr.reprojected_px[0]:.1f}"
                proj_img_y = f"{pr.reprojected_px[1]:.1f}"

                if lbl in anomaly_labels:
                    status_quality = "MALO_SWAP_ANOMALY"
                    note = f"Sospecha de swap o inversión de etiqueta (err={err_m_val:.2f}m)"
                    self.total_swap_anomalies += 1
                    self._kp_stats[lbl]["outliers"] += 1
                elif pr.is_inlier:
                    status_quality = "BUENO_INLIER"
                    note = f"Inlier RANSAC/MAGSAC válido (err={err_m_val:.2f}m)"
                    self.total_good_inliers += 1
                    self._kp_stats[lbl]["inliers"] += 1
                else:
                    status_quality = "MALO_OUTLIER_RANSAC"
                    note = f"Outlier descartado por MAGSAC (err={err_m_val:.2f}m > umbral)"
                    self.total_outliers_ransac += 1
                    self._kp_stats[lbl]["outliers"] += 1
            else:
                # Candidato que pasó filtros pero compute_homography falló antes o durante
                if err_m_val is not None:
                    if err_m_val <= 1.5:
                        status_quality = "BUENO_COHERENTE"
                        note = f"Coherente con H previa (err={err_m_val:.2f}m). H falló: {compute_error_reason or 'DESCONOCIDO'}"
                        self.total_good_inliers += 1
                        self._kp_stats[lbl]["inliers"] += 1
                    else:
                        status_quality = "MALO_DEFORMACION"
                        note = f"Discrepancia alta con H previa (err={err_m_val:.2f}m). Causante probable de fallo de H"
                        self.total_outliers_ransac += 1
                        self._kp_stats[lbl]["outliers"] += 1
                else:
                    status_quality = "MALO_H_FALLO"
                    note = f"H no calculada: {compute_error_reason or 'FALLO_CALCULO'} ({compute_error_details or ''})"
                    self.total_outliers_ransac += 1
                    self._kp_stats[lbl]["outliers"] += 1

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
                "reprojected_x_px": proj_img_x,
                "reprojected_y_px": proj_img_y,
                "field_x_m": fld_x_str,
                "field_y_m": fld_y_str,
                "diagnostic_note": note,
            })

        # Si ningún punto superó el umbral mínimo
        if detected_count == 0:
            self.rows.append({
                "frame_id": frame_idx,
                "timestamp_sec": ts,
                "homography_status": homography_status,
                "kp_idx": -1,
                "kp_label": "NONE",
                "confidence": 0.0,
                "status_quality": "MALO_LOW_CONF_ALL",
                "error_px": "",
                "error_m": "",
                "detected_x_px": "",
                "detected_y_px": "",
                "reprojected_x_px": "",
                "reprojected_y_px": "",
                "field_x_m": "",
                "field_y_m": "",
                "diagnostic_note": f"Ningún keypoint superó conf=0.15. Razón H: {compute_error_reason or 'INSUFICIENTE'}",
            })

    def save_to_csv(self, target_path: Optional[str] = None) -> str:
        """
        Exporta todas las filas acumuladas al archivo CSV destino.
        """
        csv_path = Path(target_path or self.output_csv_path or "outputs/keypoints_audit.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)

        logger.info(f"📊 CSV de Auditoría de Keypoints exportado: {csv_path} ({len(self.rows)} registros)")
        return str(csv_path.resolve())

    def get_summary_report(self) -> str:
        """
        Genera un resumen analítico de los keypoints más ruidosos y tasas de éxito.
        """
        inlier_pct = (self.total_good_inliers / max(1, self.total_evaluated)) * 100.0
        outlier_pct = (self.total_outliers_ransac / max(1, self.total_evaluated)) * 100.0

        lines = [
            "=" * 65,
            "📋 RESUMEN DE AUDITORÍA FORENSE DE KEYPOINTS (CALIDAD H)",
            "=" * 65,
            f"   Total puntos evaluados:        {self.total_evaluated}",
            f"   Inliers válidos (BUENOS):      {self.total_good_inliers} ({inlier_pct:.1f}%)",
            f"   Outliers RANSAC (MALOS):       {self.total_outliers_ransac} ({outlier_pct:.1f}%)",
            f"   Descartados por mitad campo:   {self.total_discarded_side}",
            f"   Descartados por borde (<4px):  {self.total_discarded_border}",
            f"   Anomalías espaciales / Swaps:  {self.total_swap_anomalies}",
            "-" * 65,
            "   Top Keypoints con mayor tasa de error / outliers:",
        ]

        # Ordenar por porcentaje de fallos
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
