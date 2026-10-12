"""
ScoutingData - Validador Geométrico de Homografía
Convex Hull, Área de Cobertura y Detección de Degeneraciones Colineales.
"""

import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

logger = logging.getLogger("homography.geometric_validator")


def compute_convex_hull_and_coverage(
    image_pts: np.ndarray,
    field_pts: np.ndarray,
    img_width: int = 1920,
    img_height: int = 1080,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Dict[str, float]:
    """
    Calcula el Convex Hull y las áreas de cobertura tanto en píxeles como en metros de cancha.
    """
    if len(image_pts) < 3:
        return {
            "hull_area_img_px2": 0.0,
            "image_coverage_ratio": 0.0,
            "hull_area_field_m2": 0.0,
            "field_coverage_ratio": 0.0,
        }

    # Hull en espacio de imagen
    pts_img_f32 = image_pts.reshape(-1, 2).astype(np.float32)
    hull_img = cv2.convexHull(pts_img_f32)
    area_img = float(cv2.contourArea(hull_img))
    total_img_area = float(img_width * img_height) if (img_width > 0 and img_height > 0) else 1.0
    coverage_img = area_img / total_img_area

    # Hull en espacio de terreno (metros)
    pts_fld_f32 = field_pts.reshape(-1, 2).astype(np.float32)
    hull_fld = cv2.convexHull(pts_fld_f32)
    area_fld = float(cv2.contourArea(hull_fld))
    total_fld_area = float(pitch_length * pitch_width) if (pitch_length > 0 and pitch_width > 0) else 7140.0
    coverage_fld = area_fld / total_fld_area

    return {
        "hull_area_img_px2": area_img,
        "image_coverage_ratio": coverage_img,
        "hull_area_field_m2": area_fld,
        "field_coverage_ratio": coverage_fld,
    }


def compute_collinearity_score(pts: np.ndarray) -> float:
    """
    Calcula el ratio de dispersión 2D vía PCA/Valores propios de la matriz de covarianza.
    Retorna sqrt(lambda_min / lambda_max) en [0.0, 1.0]:
      - 0.0: Puntos perfectamente colineales (degeneración proyectiva fatal).
      - ~1.0: Puntos distribuidos uniformemente en 2D.
    """
    if len(pts) < 3:
        return 0.0

    pts_2d = pts.reshape(-1, 2).astype(np.float64)
    centered = pts_2d - pts_2d.mean(axis=0)
    cov = np.cov(centered, rowvar=False)

    if cov.shape != (2, 2) or np.isnan(cov).any():
        return 0.0

    eigvals = np.linalg.eigvalsh(cov)
    # eigvalsh retorna [lambda_min, lambda_max] ordenados ascendentemente
    l_min = max(0.0, float(eigvals[0]))
    l_max = max(1e-9, float(eigvals[1]))

    return float(np.sqrt(l_min / l_max))


def validate_keypoint_geometry(
    image_pts: np.ndarray,
    field_pts: np.ndarray,
    img_width: int = 1920,
    img_height: int = 1080,
    min_points: int = 4,
    min_hull_area_m2: float = 80.0,      # Mínimo ~1.1% del campo de 7140m²
    min_image_coverage: float = 0.015,   # Mínimo 1.5% del viewport de cámara
    min_collinearity_score: float = 0.08, # Ratio mínimo de ancho/largo 2D
    min_span_x_m: float = 6.0,           # Dispersión mínima en X (largo)
    min_span_y_m: float = 5.0,           # Dispersión mínima en Y (ancho)
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evalúa la validez geométrica del conjunto de puntos antes o después de RANSAC.
    Rechaza configuraciones colineales, hiper-concentradas o insuficientes.
    """
    n_pts = len(image_pts)
    if n_pts < min_points:
        return False, f"Puntos insuficientes: {n_pts} < {min_points}", {}

    coverage_info = compute_convex_hull_and_coverage(
        image_pts, field_pts, img_width=img_width, img_height=img_height
    )
    collinearity_fld = compute_collinearity_score(field_pts)
    collinearity_img = compute_collinearity_score(image_pts)
    min_collin = min(collinearity_fld, collinearity_img)

    fld_2d = field_pts.reshape(-1, 2)
    span_x_m = float(fld_2d[:, 0].max() - fld_2d[:, 0].min())
    span_y_m = float(fld_2d[:, 1].max() - fld_2d[:, 1].min())
    is_low_longitudinal_span = bool(span_x_m < 25.0)

    geom_metrics = {
        **coverage_info,
        "collinearity_score": min_collin,
        "span_x_m": span_x_m,
        "span_y_m": span_y_m,
        "is_low_longitudinal_span": is_low_longitudinal_span,
    }

    # 1. Colinealidad severa
    if min_collin < min_collinearity_score:
        return False, (
            f"Degeneración geométrica: puntos colineales (score={min_collin:.3f} < {min_collinearity_score})"
        ), geom_metrics

    # 2. Área en campo demasiado pequeña (puntos agrupados en una zona diminuta)
    if coverage_info["hull_area_field_m2"] < min_hull_area_m2:
        return False, (
            f"Área cubierta en campo insuficiente: {coverage_info['hull_area_field_m2']:.1f}m² < {min_hull_area_m2}m²"
        ), geom_metrics

    # 3. Cobertura de imagen insuficiente
    if coverage_info["image_coverage_ratio"] < min_image_coverage:
        return False, (
            f"Cobertura en imagen insuficiente: {coverage_info['image_coverage_ratio']*100:.2f}% < {min_image_coverage*100:.2f}%"
        ), geom_metrics

    # 4. Falta de apertura espacial en uno de los ejes
    if span_x_m < min_span_x_m or span_y_m < min_span_y_m:
        return False, (
            f"Dispersión dimensional insuficiente: span_x={span_x_m:.1f}m (min {min_span_x_m}m), "
            f"span_y={span_y_m:.1f}m (min {min_span_y_m}m)"
        ), geom_metrics

    return True, "OK", geom_metrics


def filter_border_keypoints(
    candidates: list,
    img_width: int = 1920,
    img_height: int = 1080,
    border_margin_px: float = 4.0,
) -> list:
    """
    Descarta keypoints pegados al borde de la imagen causados por clamping/truncamiento de la red neuronal.
    candidates es una lista de tuplas: (idx, xy, conf, label) o similar donde xy tiene coordenadas [x, y].
    """
    filtered = []
    for item in candidates:
        xy = item[1]
        x, y = float(xy[0]), float(xy[1])
        if (
            x <= border_margin_px
            or y <= border_margin_px
            or x >= (img_width - border_margin_px)
            or y >= (img_height - border_margin_px)
        ):
            logger.debug(f"Keypoint descartado por borde de imagen: {item[3]} en ({x:.1f}, {y:.1f})")
            continue
        filtered.append(item)
    return filtered
