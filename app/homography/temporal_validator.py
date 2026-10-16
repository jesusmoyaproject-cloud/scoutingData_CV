"""
ScoutingData - Validador Temporal y Tracker de Homografía (v5.1 Refactorizado)
Control de Jitter, Detección Cinemática Robusta (SVD/Polar), ROI Segura de Césped y Prevención de Cascada.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2

logger = logging.getLogger("homography.temporal_validator")


@dataclass
class RejectionRecord:
    """Registro estructurado de telemetría de un frame evaluado por el validador temporal."""
    frame_id: int
    reason: str
    homography_state: str = "REJECTED_HELD"
    reject_streak: int = 1
    mean_reprojection_error_m: Optional[float] = None
    center_shift_m: Optional[float] = None
    max_center_shift_m: Optional[float] = None
    corner_shift_m: Optional[float] = None
    max_corner_allowed_m: Optional[float] = None
    rotation_deg: Optional[float] = None
    max_rotation_deg: Optional[float] = None
    scale_ratio: Optional[float] = None
    allowed_scale_range: Optional[str] = None
    shear_ratio: Optional[float] = None
    coverage_ratio: Optional[float] = None
    ransac_inliers: Optional[int] = None
    ransac_outliers: Optional[int] = None
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "reason": self.reason,
            "homography_state": self.homography_state,
            "reject_streak": self.reject_streak,
            "mean_reprojection_error_m": f"{self.mean_reprojection_error_m:.2f}" if self.mean_reprojection_error_m is not None else "",
            "center_shift_m": f"{self.center_shift_m:.2f}" if self.center_shift_m is not None else "",
            "max_center_shift_m": f"{self.max_center_shift_m:.2f}" if self.max_center_shift_m is not None else "",
            "corner_shift_m": f"{self.corner_shift_m:.2f}" if self.corner_shift_m is not None else "",
            "max_corner_allowed_m": f"{self.max_corner_allowed_m:.2f}" if self.max_corner_allowed_m is not None else "",
            "rotation_deg": f"{self.rotation_deg:.2f}" if self.rotation_deg is not None else "",
            "max_rotation_deg": f"{self.max_rotation_deg:.2f}" if self.max_rotation_deg is not None else "",
            "scale_ratio": f"{self.scale_ratio:.3f}" if self.scale_ratio is not None else "",
            "allowed_scale_range": self.allowed_scale_range or "",
            "shear_ratio": f"{self.shear_ratio:.3f}" if self.shear_ratio is not None else "",
            "coverage_ratio": f"{self.coverage_ratio * 100.0:.1f}%" if self.coverage_ratio is not None else "",
            "ransac_inliers": self.ransac_inliers if self.ransac_inliers is not None else "",
            "ransac_outliers": self.ransac_outliers if self.ransac_outliers is not None else "",
            "details": self.details or "",
        }


CSV_FIELDNAMES = [
    "frame_id",
    "reason",
    "homography_state",
    "reject_streak",
    "mean_reprojection_error_m",
    "center_shift_m",
    "max_center_shift_m",
    "corner_shift_m",
    "max_corner_allowed_m",
    "rotation_deg",
    "max_rotation_deg",
    "scale_ratio",
    "allowed_scale_range",
    "shear_ratio",
    "coverage_ratio",
    "ransac_inliers",
    "ransac_outliers",
    "details",
]


def get_safe_pitch_roi(
    img_width: int = 1920,
    img_height: int = 1080,
    margin_x_pct: float = 0.10,   # Margen lateral del 10%
    top_y_pct: float = 0.35,      # Margen superior del 35% (evita línea de fuga/horizonte)
    bottom_y_pct: float = 0.95,   # Margen inferior del 95%
) -> np.ndarray:
    """
    Genera 4 esquinas de control dentro de la ROI de césped garantizado en la transmisión.
    Elimina la singularidad proyectiva de (0,0) y (W,0) donde el denominador w -> 0.
    """
    x0 = float(img_width) * margin_x_pct
    x1 = float(img_width) * (1.0 - margin_x_pct)
    y0 = float(img_height) * top_y_pct
    y1 = float(img_height) * bottom_y_pct
    return np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ], dtype=np.float32).reshape(-1, 1, 2)


def decompose_homography_transition(
    H_new: np.ndarray,
    H_prev: np.ndarray,
    img_width: int = 1920,
    img_height: int = 1080,
    margin_x_pct: float = 0.10,
    top_y_pct: float = 0.35,
    bottom_y_pct: float = 0.95,
) -> Dict[str, float]:
    """
    Evalúa cinemáticamente la transición H_new respecto a H_prev utilizando:
    1. Desplazamiento del centro óptico en coordenadas métricas del terreno.
    2. Deriva de las 4 esquinas de la ROI segura de césped (sin singularidad de horizonte).
    3. Descomposición SVD de la transformación afín inducida en el terreno H_delta = inv(H_prev) @ H_new.
       Retorna: rotación euclídea (°), escala geométrica y ratio de deformación (shear).
    """
    center = np.array([
        [[float(img_width) / 2.0, float(img_height) / 2.0]]
    ], dtype=np.float32)

    # 1. Proyección del centro al terreno (metros)
    c_new = cv2.perspectiveTransform(center, H_new).reshape(-1, 2)[0]
    c_prev = cv2.perspectiveTransform(center, H_prev).reshape(-1, 2)[0]
    center_shift_m = float(np.linalg.norm(c_new - c_prev))

    # 2. Proyección de la ROI segura de césped (metros)
    roi = get_safe_pitch_roi(img_width, img_height, margin_x_pct, top_y_pct, bottom_y_pct)
    p_new = cv2.perspectiveTransform(roi, H_new).reshape(-1, 2)
    p_prev = cv2.perspectiveTransform(roi, H_prev).reshape(-1, 2)
    corner_shifts = np.linalg.norm(p_new - p_prev, axis=1)
    max_corner_shift_m = float(corner_shifts.max())

    # 3. Transformación relativa en espacio de terreno: H_delta = inv(H_prev) @ H_new
    try:
        H_p_inv = np.linalg.inv(H_prev)
        H_delta = H_p_inv @ H_new
        if abs(H_delta[2, 2]) > 1e-9:
            H_delta = H_delta / H_delta[2, 2]

        A = H_delta[:2, :2]
        # SVD: A = U @ diag(s) @ V^T
        U, s, Vt = np.linalg.svd(A)
        # Matriz ortogonal de rotación R = U @ Vt
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        rot_rad = np.arctan2(R[1, 0], R[0, 0])
        rotation_deg = float(np.abs(np.rad2deg(rot_rad)))
        scale_ratio = float(np.sqrt(max(1e-9, s[0] * s[1])))
        shear_ratio = float(s[0] / max(1e-6, s[1]))
    except Exception:
        rotation_deg = 0.0
        scale_ratio = 1.0
        shear_ratio = 1.0

    return {
        "center_shift_m": center_shift_m,
        "max_corner_shift_m": max_corner_shift_m,
        "rotation_deg": rotation_deg,
        "scale_ratio": scale_ratio,
        "shear_ratio": shear_ratio,
    }


def normalize_homography(H: np.ndarray) -> np.ndarray:
    """Normaliza H por su norma de Frobenius para comparaciones y blendings estables."""
    f_norm = np.linalg.norm(H, "fro")
    if f_norm > 1e-9:
        H_norm = H / f_norm
    else:
        H_norm = H.copy()
    if H_norm[2, 2] < 0:
        H_norm = -H_norm
    return H_norm


class TemporalHomographyTracker:
    """
    Rastreador temporal robusto v5.1 para homografía deportiva.
    - Utiliza ROI segura de césped (elimina fallos por línea de fuga).
    - Descomposición SVD para rotación, escala y shear en terreno.
    - Prevención del efecto cascada mediante umbrales elásticos adaptativos y fast-recovery.
    - Feature flags para pruebas A/B de métricas.
    """

    def __init__(
        self,
        max_shift_m: float = 3.5,            # Recalibrado: salto base de centro (p95=0.70m, p99=1.24m)
        max_corner_shift_m: float = 6.0,     # Recalibrado: deriva de esquinas en ROI segura (p95=1.40m, p99=2.30m)
        min_scale_ratio: float = 0.75,       # Recalibrado: zoom mínimo tolerado por frame
        max_scale_ratio: float = 1.30,       # Recalibrado: zoom máximo tolerado por frame
        max_rot_deg: float = 5.0,            # Recalibrado: rotación angular SVD máxima por frame (p95=0.89°)
        max_shear_ratio: float = 1.30,       # Ratio de distorsión anarmónica / cizallamiento admisible
        max_reject_streak: int = 10,         # Frames consecutivos antes de forzar re-anclaje
        base_alpha: float = 0.65,            # Inercia de suavizado exponencial
        enable_corner_shift: bool = True,    # Feature flag A/B para evaluación de esquinas
        enable_elastic_thresholds: bool = True,  # Previene cascada al acumular streak de congelamiento
        enable_fast_recovery: bool = True,   # Acepta re-anclaje inmediato si la geometría es impecable
        csv_path: Optional[str] = None,      # Ruta opcional de persistencia CSV
    ):
        self.max_shift_m = max_shift_m
        self.max_corner_shift_m = max_corner_shift_m
        self.min_scale_ratio = min_scale_ratio
        self.max_scale_ratio = max_scale_ratio
        self.max_rot_deg = max_rot_deg
        self.max_shear_ratio = max_shear_ratio
        self.max_reject_streak = max_reject_streak
        self.base_alpha = base_alpha
        self.enable_corner_shift = enable_corner_shift
        self.enable_elastic_thresholds = enable_elastic_thresholds
        self.enable_fast_recovery = enable_fast_recovery
        self.csv_path = csv_path

        self.cached_H: Optional[np.ndarray] = None
        self.cached_mask: Optional[np.ndarray] = None
        self.reject_streak: int = 0
        self.total_accepted: int = 0
        self.total_rejected: int = 0

        # Telemetría e instrumentación
        self.rejection_records: List[RejectionRecord] = []
        self.rejection_counts: Dict[str, int] = {
            "CENTER_SHIFT": 0,
            "ROTATION": 0,
            "SCALE": 0,
            "CORNER_SHIFT": 0,
            "SHEAR": 0,
            "LOW_COVERAGE": 0,
            "COLLINEARITY": 0,
            "NO_HOMOGRAPHY": 0,
        }
        self._frame_seq: int = 0

    def reset(self):
        """Reinicia el historial temporal dinámico (no borra estadísticas globales del video)."""
        self.cached_H = None
        self.cached_mask = None
        self.reject_streak = 0

    def clear_statistics(self):
        """Limpia el histórico de rechazos y contadores."""
        self.rejection_records.clear()
        for k in self.rejection_counts:
            self.rejection_counts[k] = 0

    def _append_to_csv(self, record: RejectionRecord):
        if not self.csv_path:
            return
        try:
            p = Path(self.csv_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            file_exists = p.exists() and p.stat().st_size > 0
            with open(p, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record.to_dict())
        except Exception as e:
            logger.debug(f"No se pudo escribir en CSV de rechazos: {e}")

    def save_rejections_to_csv(self, filepath: Optional[str] = None) -> str:
        """Guarda todos los registros acumulados en el archivo CSV especificado."""
        target = filepath or self.csv_path or "rejections.csv"
        p = Path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for r in self.rejection_records:
                writer.writerow(r.to_dict())
        return str(p)

    def _log_rejection(self, record: RejectionRecord):
        """Emite el log formateado exacto por cada rechazo."""
        lines = [
            "",
            "[REJECTED]",
            f"frame={record.frame_id}",
            f"reason={record.reason}",
            f"homography_state={record.homography_state}",
            f"streak={record.reject_streak}",
            "",
        ]
        if record.reason == "CENTER_SHIFT":
            c_val = f"{record.center_shift_m:.2f}m" if record.center_shift_m is not None else "N/A"
            lines.extend([
                f"center_shift={c_val}",
                f"threshold={record.max_center_shift_m:.2f}m",
                "",
            ])
        elif record.reason == "ROTATION":
            r_val = f"{record.rotation_deg:.1f}°" if record.rotation_deg is not None else "N/A"
            lines.extend([
                f"rotation={r_val}",
                f"threshold={record.max_rotation_deg:.1f}°",
                "",
            ])
        elif record.reason == "CORNER_SHIFT":
            corn_val = f"{record.corner_shift_m:.2f}m" if record.corner_shift_m is not None else "N/A"
            lines.extend([
                f"corner_shift={corn_val}",
                f"threshold={record.max_corner_allowed_m:.2f}m",
                "",
            ])
        elif record.reason == "SCALE":
            s_val = f"{record.scale_ratio:.2f}" if record.scale_ratio is not None else "N/A"
            lines.extend([
                f"scale={s_val}",
                f"threshold={record.allowed_scale_range}",
                "",
            ])
        elif record.reason == "SHEAR":
            sh_val = f"{record.shear_ratio:.2f}" if record.shear_ratio is not None else "N/A"
            lines.extend([
                f"shear_ratio={sh_val}",
                f"threshold={self.max_shear_ratio:.2f}",
                "",
            ])
        elif record.reason in ("LOW_COVERAGE", "COLLINEARITY", "NO_HOMOGRAPHY"):
            if record.details:
                lines.extend([
                    f"details={record.details}",
                    "",
                ])

        if record.coverage_ratio is not None:
            lines.extend([
                f"coverage={record.coverage_ratio * 100.0:.1f}%",
                "",
            ])

        if record.mean_reprojection_error_m is not None:
            lines.extend([
                f"mean_error={record.mean_reprojection_error_m:.2f}m",
                "",
            ])

        if record.ransac_inliers is not None and record.ransac_outliers is not None:
            lines.extend([
                f"inliers={record.ransac_inliers}",
                f"outliers={record.ransac_outliers}",
                "",
            ])

        lines.append("---------------------")
        logger.debug("\n".join(lines))

    def _record_rejection(
        self,
        frame_id: int,
        reason: str,
        homography_state: str = "REJECTED_HELD",
        reject_streak: int = 1,
        center_shift_m: Optional[float] = None,
        effective_center_threshold: Optional[float] = None,
        corner_shift_m: Optional[float] = None,
        effective_corner_threshold: Optional[float] = None,
        rotation_deg: Optional[float] = None,
        effective_rot_threshold: Optional[float] = None,
        scale_ratio: Optional[float] = None,
        shear_ratio: Optional[float] = None,
        coverage_ratio: Optional[float] = None,
        mean_reprojection_error_m: Optional[float] = None,
        ransac_inliers: Optional[int] = None,
        ransac_outliers: Optional[int] = None,
        details: Optional[str] = None,
    ) -> RejectionRecord:
        if reason not in self.rejection_counts:
            self.rejection_counts[reason] = 0
        self.rejection_counts[reason] += 1

        rec = RejectionRecord(
            frame_id=frame_id,
            reason=reason,
            homography_state=homography_state,
            reject_streak=reject_streak,
            center_shift_m=center_shift_m,
            max_center_shift_m=effective_center_threshold or self.max_shift_m,
            corner_shift_m=corner_shift_m,
            max_corner_allowed_m=effective_corner_threshold or self.max_corner_shift_m,
            rotation_deg=rotation_deg,
            max_rotation_deg=effective_rot_threshold or self.max_rot_deg,
            scale_ratio=scale_ratio,
            allowed_scale_range=f"[{self.min_scale_ratio:.2f}, {self.max_scale_ratio:.2f}]",
            shear_ratio=shear_ratio,
            coverage_ratio=coverage_ratio,
            mean_reprojection_error_m=mean_reprojection_error_m,
            ransac_inliers=ransac_inliers,
            ransac_outliers=ransac_outliers,
            details=details,
        )
        self.rejection_records.append(rec)
        self._log_rejection(rec)
        if self.csv_path:
            self._append_to_csv(rec)
        return rec

    def get_rejection_summary(self) -> str:
        """Genera el resumen de texto con alineación de puntos formateada."""
        lines = [
            "Rejection Statistics",
            "",
        ]
        target_keys = [
            "CENTER_SHIFT",
            "ROTATION",
            "SCALE",
        ]
        if self.enable_corner_shift or self.rejection_counts.get("CORNER_SHIFT", 0) > 0:
            target_keys.append("CORNER_SHIFT")
        if self.rejection_counts.get("SHEAR", 0) > 0:
            target_keys.append("SHEAR")
        target_keys.extend([
            "LOW_COVERAGE",
            "COLLINEARITY",
            "NO_HOMOGRAPHY",
        ])
        for k in self.rejection_counts:
            if k not in target_keys:
                target_keys.append(k)

        for key in target_keys:
            cnt = self.rejection_counts.get(key, 0)
            dots_count = max(2, 23 - len(key) - 1)
            lines.append(f"{key} {'.' * dots_count} {cnt}")

        total_rej = sum(self.rejection_counts.values())
        tot_label = "TOTAL REJECTIONS"
        tot_dots = "." * max(2, 23 - len(tot_label) - 1)
        lines.append("")
        lines.append(f"{tot_label} {tot_dots} {total_rej}")
        return "\n".join(lines)

    def log_rejection_summary(self):
        """Imprime el resumen de rechazos en el logger."""
        logger.info("\n" + self.get_rejection_summary())

    def update(
        self,
        H_cand: Optional[np.ndarray],
        mask_cand: Optional[np.ndarray],
        img_width: int = 1920,
        img_height: int = 1080,
        confidence_weight: float = 1.0,
        frame_id: Optional[int] = None,
        diagnostics: Optional[Any] = None,
        candidate_reject_reason: Optional[str] = None,
        details: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, Any]]:
        """
        Procesa una nueva homografía candidata con validación temporal robusta.
        Retorna: (H_efectiva, mask_efectiva, status, transition_metrics)
        """
        self._frame_seq = frame_id if frame_id is not None else (self._frame_seq + 1)
        fid = self._frame_seq

        cov_ratio = getattr(diagnostics, "field_coverage_ratio", None) if diagnostics else None
        mean_err = getattr(diagnostics, "mean_reprojection_error_m", None) if diagnostics else None
        inliers = getattr(diagnostics, "ransac_inliers", None) if diagnostics else None
        outliers = getattr(diagnostics, "ransac_outliers", None) if diagnostics else None

        metrics: Dict[str, Any] = {}

        # ── Caso A: Candidato no generado (fallo de detección previa o RANSAC) ──
        if H_cand is None:
            self.reject_streak += 1
            self.total_rejected += 1
            reason = candidate_reject_reason or "NO_HOMOGRAPHY"
            state = "REJECTED_HELD" if (self.cached_H is not None and self.reject_streak < self.max_reject_streak) else "FAILED_NO_PREV"
            self._record_rejection(
                frame_id=fid,
                reason=reason,
                homography_state=state,
                reject_streak=self.reject_streak,
                coverage_ratio=cov_ratio,
                mean_reprojection_error_m=mean_err,
                ransac_inliers=inliers,
                ransac_outliers=outliers,
                details=details or "Candidato de homografía no generado",
            )
            metrics = {
                "status": state,
                "reject_reason": reason,
                "reject_details": details or "Candidato de homografía no generado",
                "reject_streak": self.reject_streak,
            }
            if self.cached_H is not None:
                if self.reject_streak >= self.max_reject_streak:
                    logger.info(
                        f"Temporal Tracker: auto-reset tras {self.reject_streak} rechazos consecutivos sin homografía (toma no calibrable: repetición o plano cerrado)."
                    )
                    self.reset()
                    metrics["status"] = "RESET_EMPTY"
                    return None, None, "RESET_EMPTY", metrics
                return self.cached_H, self.cached_mask, "REJECTED_HELD", metrics
            return None, None, "FAILED_NO_PREV", metrics

        # ── Caso B: Primera homografía válida o tras reset ──────────────────────
        if self.cached_H is None:
            h22 = H_cand[2, 2] if abs(H_cand[2, 2]) > 1e-9 else 1.0
            self.cached_H = H_cand / h22
            self.cached_mask = mask_cand
            self.reject_streak = 0
            self.total_accepted += 1
            logger.debug(f"[Frame {fid}] Temporal Tracker: Homografía inicial aceptada.")
            metrics = {
                "status": "INITIAL_ACCEPTED",
                "reject_streak": 0,
            }
            return self.cached_H, self.cached_mask, "INITIAL_ACCEPTED", metrics

        # ── Caso C: Evaluar transición cinemática contra H previa ───────────────
        trans = decompose_homography_transition(H_cand, self.cached_H, img_width, img_height)
        metrics = trans

        shift = trans["center_shift_m"]
        corner_shift = trans["max_corner_shift_m"]
        scale = trans["scale_ratio"]
        rot = trans["rotation_deg"]
        shear = trans.get("shear_ratio", 1.0)

        # Cálculo de umbrales elásticos adaptativos (Prevención de Cascada)
        if self.enable_elastic_thresholds:
            dyn_factor = 1.0 + 0.5 * min(self.reject_streak, 6)
            eff_max_shift_m = min(self.max_shift_m * dyn_factor, 8.0)
            eff_max_corner_m = min(self.max_corner_shift_m * dyn_factor, 12.0)
            eff_max_rot_deg = min(self.max_rot_deg * (1.0 + 0.2 * min(self.reject_streak, 6)), 10.0)
        else:
            eff_max_shift_m = self.max_shift_m
            eff_max_corner_m = self.max_corner_shift_m
            eff_max_rot_deg = self.max_rot_deg

        metrics["effective_max_center_shift_m"] = eff_max_shift_m
        metrics["effective_max_corner_shift_m"] = eff_max_corner_m
        metrics["effective_max_rot_deg"] = eff_max_rot_deg
        metrics["max_shear_ratio"] = self.max_shear_ratio
        metrics["allowed_scale_range"] = f"[{self.min_scale_ratio:.2f}, {self.max_scale_ratio:.2f}]"

        is_transition_valid = True
        reject_reason = ""
        reject_details = ""

        # Evaluación jerárquica de estabilidad
        if shift > eff_max_shift_m:
            is_transition_valid = False
            reject_reason = "CENTER_SHIFT"
            reject_details = f"Salto de centro excesivo: {shift:.2f}m > {eff_max_shift_m:.2f}m (streak={self.reject_streak})"
        elif rot > eff_max_rot_deg:
            is_transition_valid = False
            reject_reason = "ROTATION"
            reject_details = f"Rotación SVD excesiva: {rot:.1f}° > {eff_max_rot_deg:.1f}°"
        elif not (self.min_scale_ratio <= scale <= self.max_scale_ratio):
            is_transition_valid = False
            reject_reason = "SCALE"
            reject_details = f"Escala SVD fuera de rango: ratio={scale:.2f} fuera de [{self.min_scale_ratio:.2f}, {self.max_scale_ratio:.2f}]"
        elif self.enable_corner_shift and (corner_shift > eff_max_corner_m):
            is_transition_valid = False
            reject_reason = "CORNER_SHIFT"
            reject_details = f"Deriva de esquinas ROI segura excesiva: {corner_shift:.2f}m > {eff_max_corner_m:.2f}m"
        elif shear > self.max_shear_ratio:
            is_transition_valid = False
            reject_reason = "SHEAR"
            reject_details = f"Deformación anarmónica/cizallamiento excesivo: {shear:.2f} > {self.max_shear_ratio:.2f}"

        # ── Fast-Recovery Bypass para Candidatos de Alta Confianza Geométrica ──
        # Si un candidato tiene sustento RANSAC indiscutible pero falló el umbral
        # temporal por causa de un paneo veloz acumulado, recuperamos sin congelar:
        if not is_transition_valid and self.enable_fast_recovery and self.reject_streak >= 1:
            is_scale_sane = (0.85 <= scale <= 1.18) and (corner_shift < self.max_corner_shift_m * 1.5)
            is_high_conf = (
                is_scale_sane
                and mean_err is not None and mean_err < 0.45
                and inliers is not None and inliers >= 8
                and (cov_ratio is None or cov_ratio >= 0.12)
                and shear <= 1.20
            )
            if is_high_conf:
                logger.info(
                    f"[Frame {fid}] 🔄 Fast-Recovery: Re-anclando en candidato de alta confianza "
                    f"(inliers={inliers}, err={mean_err:.2f}m, dC={shift:.2f}m) tras streak={self.reject_streak}."
                )
                h22 = H_cand[2, 2] if abs(H_cand[2, 2]) > 1e-9 else 1.0
                self.cached_H = H_cand / h22
                self.cached_mask = mask_cand
                self.reject_streak = 0
                self.total_accepted += 1
                metrics["status"] = "FAST_RECOVERY_ACCEPTED"
                metrics["reject_streak"] = 0
                return self.cached_H, self.cached_mask, "FAST_RECOVERY_ACCEPTED", metrics

        if is_transition_valid:
            # ── Transición válida: aplicar Suavizado Adaptativo (EMA) ──────────
            self.reject_streak = 0
            self.total_accepted += 1

            alpha = float(np.clip(self.base_alpha * confidence_weight, 0.40, 0.75))

            H_n = normalize_homography(H_cand)
            H_p = normalize_homography(self.cached_H)

            if np.sign(H_n[2, 2]) != np.sign(H_p[2, 2]):
                H_n = -H_n

            H_smoothed = alpha * H_n + (1.0 - alpha) * H_p
            h22 = H_smoothed[2, 2] if abs(H_smoothed[2, 2]) > 1e-9 else 1.0
            H_final = H_smoothed / h22

            self.cached_H = H_final
            self.cached_mask = mask_cand
            logger.debug(f"[Frame {fid}] H aceptada | α={alpha:.2f} | dC={shift:.2f}m rot={rot:.1f}°")
            metrics["status"] = "ACCEPTED_SMOOTHED"
            metrics["smoothing_alpha"] = alpha
            metrics["reject_streak"] = 0
            return self.cached_H, self.cached_mask, "ACCEPTED_SMOOTHED", metrics

        # ── Transición inválida: registrar y sostener previa ──────────────────
        self.reject_streak += 1
        self.total_rejected += 1
        self._record_rejection(
            frame_id=fid,
            reason=reject_reason,
            homography_state="REJECTED_HELD",
            reject_streak=self.reject_streak,
            center_shift_m=shift,
            effective_center_threshold=eff_max_shift_m,
            corner_shift_m=corner_shift,
            effective_corner_threshold=eff_max_corner_m,
            rotation_deg=rot,
            effective_rot_threshold=eff_max_rot_deg,
            scale_ratio=scale,
            shear_ratio=shear,
            coverage_ratio=cov_ratio,
            mean_reprojection_error_m=mean_err,
            ransac_inliers=inliers,
            ransac_outliers=outliers,
            details=reject_details,
        )

        metrics["status"] = "REJECTED_HELD"
        metrics["reject_reason"] = reject_reason
        metrics["reject_details"] = reject_details
        metrics["reject_streak"] = self.reject_streak

        if self.reject_streak >= self.max_reject_streak and reject_reason not in ("SCALE", "SHEAR"):
            logger.info(
                f"[Frame {fid}] Temporal Tracker: Auto-reset tras {self.reject_streak} rechazos consecutivos. "
                "Re-anclando en nueva posición de cámara."
            )
            h22 = H_cand[2, 2] if abs(H_cand[2, 2]) > 1e-9 else 1.0
            self.cached_H = H_cand / h22
            self.cached_mask = mask_cand
            self.reject_streak = 0
            metrics["status"] = "RESET_ACCEPTED"
            metrics["reject_streak"] = 0
            return self.cached_H, self.cached_mask, "RESET_ACCEPTED", metrics

        return self.cached_H, self.cached_mask, "REJECTED_HELD", metrics
