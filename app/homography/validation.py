"""
ScoutingData v5.0 - Módulo de Validación de Homografía
Validación matricial, evaluación de error de reproyección y puente hacia validadores especializados.
"""

import logging
import cv2
import numpy as np
from typing import Tuple, Dict, Optional, Any
from homography.transforms import pixel_to_field
from homography.temporal_validator import decompose_homography_transition, normalize_homography

_val_logger = logging.getLogger("Homography.debug")


def evaluate_reprojection_error(
    H: np.ndarray, 
    image_pts: np.ndarray, 
    field_pts: np.ndarray
) -> Dict[str, Any]:
    """
    Computes Euclidean error metrics in meters for the homography transformation.
    """
    if H is None or len(image_pts) == 0:
        return {"mean_error_meters": 0.0, "max_error_meters": 0.0, "raw_errors": []}

    projected = cv2.perspectiveTransform(
        image_pts.reshape(-1, 1, 2).astype(np.float32),
        H
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - field_pts, axis=1)

    return {
        "mean_error_meters": float(errors.mean()),
        "max_error_meters": float(errors.max()),
        "raw_errors": [float(e) for e in errors]
    }


def validate_homography_matrix(
    H: np.ndarray,
    min_normalized_det: float = 1e-12,
    max_cond: float = 1e7
) -> bool:
    """
    Verifica que la matriz H sea matemáticamente válida, no singular y con condicionamiento numérico estable.
    Usa la matriz normalizada por norma de Frobenius para que el determinante sea invariante a la escala.
    """
    if H is None or H.shape != (3, 3) or np.isnan(H).any() or np.isinf(H).any():
        return False

    fro = np.linalg.norm(H, "fro")
    if fro < 1e-9:
        return False

    H_norm = H / fro
    det = float(np.linalg.det(H_norm))
    if abs(det) < min_normalized_det:
        _val_logger.debug(f"H rechazada: det normalizado singular ({abs(det):.2e} < {min_normalized_det})")
        return False

    try:
        cond = float(np.linalg.cond(H_norm))
        if cond > max_cond:
            _val_logger.debug(f"H rechazada: número de condición excesivo ({cond:.1e} > {max_cond:.1e})")
            return False
    except np.linalg.LinAlgError:
        return False

    return True


def is_homography_transition_valid(
    H_new: np.ndarray,
    H_prev: Optional[np.ndarray],
    img_width: int = 1920,
    img_height: int = 1080,
    max_shift_m: float = 4.5,            # Umbral físico realista para tracking continuo
    max_corner_shift_m: float = 12.0,
    max_rot_deg: float = 7.0,
    pitch_width: float = 105.0,
    pitch_height: float = 68.0,
    margin_m: float = 25.0
) -> bool:
    """
    Valida la continuidad cinemática entre H_new y H_prev:
    - Verificación matemática intrínseca de H_new
    - Verificación de límites de campo de la proyección del centro
    - Detección de saltos bruscos de traslación, rotación o escala
    """
    if not validate_homography_matrix(H_new):
        _val_logger.debug("  ✖ H_new falla validate_homography_matrix")
        return False

    # 1. Posición proyectada del centro de la imagen en el terreno
    cx, cy = img_width / 2.0, img_height / 2.0
    fx_new, fy_new = pixel_to_field(cx, cy, H_new)

    # Verificar que el centro de cámara proyectado caiga en un rango razonable del terreno
    if not (-margin_m <= fx_new <= pitch_width + margin_m and -margin_m <= fy_new <= pitch_height + margin_m):
        _val_logger.debug(
            f"  ✖ Proyección fuera del campo: fx={fx_new:.1f}m, fy={fy_new:.1f}m "
            f"(límites: [-{margin_m}, {pitch_width + margin_m}] x [-{margin_m}, {pitch_height + margin_m}])"
        )
        return False

    # Si no hay H_prev, H_new es la inicial y es válida
    if H_prev is None or not validate_homography_matrix(H_prev):
        return True

    # 2. Análisis cinemático diferencial
    trans = decompose_homography_transition(H_new, H_prev, img_width=img_width, img_height=img_height)
    shift = trans["center_shift_m"]
    corner_shift = trans["max_corner_shift_m"]
    scale = trans["scale_ratio"]
    rot = trans["rotation_deg"]

    if shift > max_shift_m:
        _val_logger.debug(f"  ✖ Salto de centro excesivo: {shift:.2f}m > {max_shift_m}m")
        return False

    if corner_shift > max_corner_shift_m:
        _val_logger.debug(f"  ✖ Deriva de esquinas excesiva: {corner_shift:.2f}m > {max_corner_shift_m}m")
        return False

    if not (0.70 <= scale <= 1.40):
        _val_logger.debug(f"  ✖ Salto brusco de escala: ratio={scale:.2f}")
        return False

    if rot > max_rot_deg:
        _val_logger.debug(f"  ✖ Salto brusco de rotación: {rot:.1f}° > {max_rot_deg}°")
        return False

    return True


def smooth_homography(
    H_new: np.ndarray,
    H_prev: Optional[np.ndarray],
    alpha: float = 0.65
) -> np.ndarray:
    """
    Aplica suavizado exponencial (EMA) a la matriz de homografía utilizando
    normalización de Frobenius y alineación de signo para evitar singularidades por mezcla de fase.
    """
    if H_prev is None or not validate_homography_matrix(H_prev):
        h22 = H_new[2, 2] if abs(H_new[2, 2]) > 1e-9 else 1.0
        return H_new / h22

    H_n = normalize_homography(H_new)
    H_p = normalize_homography(H_prev)

    # Alinear signo relativo
    if np.sign(H_n[2, 2]) != np.sign(H_p[2, 2]):
        H_n = -H_n

    H_smoothed = alpha * H_n + (1.0 - alpha) * H_p

    if not validate_homography_matrix(H_smoothed):
        _val_logger.debug("  ⚠ H_smoothed degenerada tras EMA — se usa H_new directa")
        h22 = H_new[2, 2] if abs(H_new[2, 2]) > 1e-9 else 1.0
        return H_new / h22

    h22 = H_smoothed[2, 2] if abs(H_smoothed[2, 2]) > 1e-9 else 1.0
    return H_smoothed / h22
