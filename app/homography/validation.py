import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from homography.transforms import pixel_to_field

def evaluate_reprojection_error(
    H: np.ndarray, 
    image_pts: np.ndarray, 
    field_pts: np.ndarray
) -> Dict[str, float]:
    """
    Computes Euclidean error metrics in meters for the homography transformation.
    """
    if len(image_pts) == 0:
        return {"mean_error_meters": 0.0, "max_error_meters": 0.0, "raw_errors": []}

    projected = cv2.perspectiveTransform(
        image_pts.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - field_pts, axis=1)

    return {
        "mean_error_meters": float(errors.mean()),
        "max_error_meters": float(errors.max()),
        "raw_errors": errors.tolist()
    }

def validate_homography_matrix(
    H: np.ndarray,
    min_det: float = 1e-7,
    max_cond: float = 1e7
) -> bool:
    """
    Verifica que la matriz H sea matemáticamente válida y no tenga inversiones ni singularidades.
    """
    if H is None or H.shape != (3, 3) or np.isnan(H).any() or np.isinf(H).any():
        return False

    det = np.linalg.det(H)
    if abs(det) < min_det:
        return False

    try:
        cond = np.linalg.cond(H)
        if cond > max_cond:
            return False
    except np.linalg.LinAlgError:
        return False

    return True

def is_homography_transition_valid(
    H_new: np.ndarray,
    H_prev: Optional[np.ndarray],
    img_width: int = 1920,
    img_height: int = 1080,
    max_shift_m: float = 15.0,
    pitch_width: float = 105.0,
    pitch_height: float = 68.0,
    margin_m: float = 25.0
) -> bool:
    """
    Valida que H_new no sufra saltos bruscos respecto a H_prev ni invierta el lado del campo.
    """
    if not validate_homography_matrix(H_new):
        return False

    # 1. Posición proyectada del centro de la imagen en el terreno
    cx, cy = img_width / 2.0, img_height / 2.0
    fx_new, fy_new = pixel_to_field(cx, cy, H_new)

    # Verificar que el centro de cámara proyectado caiga en un rango razonable del terreno
    if not (-margin_m <= fx_new <= pitch_width + margin_m and -margin_m <= fy_new <= pitch_height + margin_m):
        return False

    # 2. Coherencia de orientación Horizontal (Evitar giros/inversiones Espejo Izquierda ↔ Derecha)
    left_x, _ = pixel_to_field(0.0, cy, H_new)
    right_x, _ = pixel_to_field(float(img_width), cy, H_new)

    # En una toma normal de fútbol, conforme avanzamos a la derecha en la imagen (pixel X sube),
    # las coordenadas de terreno también deben cambiar consistentemente en orientación.
    # Si la orientación cambia bruscamente (ej. left_x > right_x cuando antes era left_x < right_x), rechazar.
    if H_prev is not None and validate_homography_matrix(H_prev):
        prev_left_x, _ = pixel_to_field(0.0, cy, H_prev)
        prev_right_x, _ = pixel_to_field(float(img_width), cy, H_prev)
        
        orient_new = right_x - left_x
        orient_prev = prev_right_x - prev_left_x
        
        if (orient_new * orient_prev) < 0:
            # Inversión de orientación izquierda/derecha respecto al frame anterior
            return False

        # 3. Control de Distancia Máxima entre frames (Rechazo de saltos bruscos)
        fx_prev, fy_prev = pixel_to_field(cx, cy, H_prev)
        shift_dist = np.sqrt((fx_new - fx_prev) ** 2 + (fy_new - fy_prev) ** 2)
        if shift_dist > max_shift_m:
            return False

    return True

def smooth_homography(
    H_new: np.ndarray,
    H_prev: Optional[np.ndarray],
    alpha: float = 0.3
) -> np.ndarray:
    """
    Aplica suavizado exponencial (EMA) a la matriz de homografía para eliminar micro-vibraciones.
    """
    if H_prev is None or not validate_homography_matrix(H_prev):
        return H_new

    # Normalizar ambas matrices por H[2,2] para que sean comparables
    H_n = H_new / (H_new[2, 2] if H_new[2, 2] != 0 else 1.0)
    H_p = H_prev / (H_prev[2, 2] if H_prev[2, 2] != 0 else 1.0)

    H_smoothed = alpha * H_n + (1.0 - alpha) * H_p
    return H_smoothed

