"""
ScoutingData - Métricas de Diagnóstico de Homografía
Auditoría y Monitoreo de Calidad de Transformación Proyectiva.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np
import cv2


@dataclass
class HomographyDiagnostics:
    """Métricas cuantitativas del cálculo de la homografía en un frame."""
    
    # Conteo de puntos
    kps_detected: int = 0
    kps_used: int = 0
    ransac_inliers: int = 0
    ransac_outliers: int = 0
    inlier_ratio: float = 0.0
    
    # Errores de reproyección (en metros de terreno de juego)
    mean_reprojection_error_m: float = 0.0
    max_reprojection_error_m: float = 0.0
    raw_errors_m: List[float] = field(default_factory=list)
    
    # Cobertura geométrica (Convex Hull)
    hull_area_image_px2: float = 0.0
    hull_area_field_m2: float = 0.0
    image_coverage_ratio: float = 0.0
    field_coverage_ratio: float = 0.0
    
    # Análisis de degeneración espacial
    collinearity_score: float = 0.0  # sqrt(lambda_min / lambda_max) de PCA (0=colineal, 1=circular)
    condition_number: float = 0.0
    is_geometrically_degenerate: bool = False
    is_low_longitudinal_span: bool = False
    
    # Validación temporal
    center_shift_m: Optional[float] = None
    max_corner_shift_m: Optional[float] = None
    scale_ratio: Optional[float] = None
    rotation_deg: Optional[float] = None
    temporal_status: str = "INITIAL"  # "ACCEPTED", "SMOOTHED", "REJECTED_HELD", "RESET"
    
    # Estado general
    is_valid: bool = False
    status_reason: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario excluyendo arrays muy largos si se desea."""
        d = asdict(self)
        d["mean_reprojection_error_m"] = round(self.mean_reprojection_error_m, 3)
        d["max_reprojection_error_m"] = round(self.max_reprojection_error_m, 3)
        d["inlier_ratio"] = round(self.inlier_ratio, 3)
        d["hull_area_field_m2"] = round(self.hull_area_field_m2, 1)
        d["field_coverage_ratio"] = round(self.field_coverage_ratio, 4)
        d["image_coverage_ratio"] = round(self.image_coverage_ratio, 4)
        d["collinearity_score"] = round(self.collinearity_score, 3)
        d["condition_number"] = round(self.condition_number, 2)
        if self.center_shift_m is not None:
            d["center_shift_m"] = round(self.center_shift_m, 2)
        if self.max_corner_shift_m is not None:
            d["max_corner_shift_m"] = round(self.max_corner_shift_m, 2)
        if self.scale_ratio is not None:
            d["scale_ratio"] = round(self.scale_ratio, 3)
        if self.rotation_deg is not None:
            d["rotation_deg"] = round(self.rotation_deg, 2)
        return d


def compute_reprojection_errors(
    H: np.ndarray,
    image_pts: np.ndarray,
    field_pts: np.ndarray
) -> Dict[str, Any]:
    """
    Calcula los errores euclidianos punto a punto en metros (espacio de campo).
    """
    if H is None or len(image_pts) == 0:
        return {
            "mean_error_m": 0.0,
            "max_error_m": 0.0,
            "raw_errors_m": []
        }

    projected = cv2.perspectiveTransform(
        image_pts.reshape(-1, 1, 2).astype(np.float32),
        H
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - field_pts, axis=1)

    return {
        "mean_error_m": float(errors.mean()),
        "max_error_m": float(errors.max()),
        "raw_errors_m": [float(e) for e in errors]
    }
