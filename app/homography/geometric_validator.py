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


CORNER_LABELS = {"01", "06", "27", "32"}


def filter_border_keypoints(
    candidates: list,
    img_width: int = 1920,
    img_height: int = 1080,
    border_margin_px: float = 12.0,
    frame_bgr: Optional[np.ndarray] = None,
    patch_radius: int = 15,
    min_grass_ratio: float = 0.25,
) -> list:
    """
    1. Descarta keypoints pegados al borde físico de la imagen (clamping de la CNN).
    2. Si se provee frame_bgr, verifica presencia de césped en HSV alrededor del punto (30x30 px)
       para descartar detecciones sobre banners publicitarios intermitentes u overlays de TV.
    """
    filtered = []
    for item in candidates:
        xy = item[1]
        x, y = float(xy[0]), float(xy[1])
        lbl = item[3]

        if (
            x <= border_margin_px
            or y <= border_margin_px
            or x >= (img_width - border_margin_px)
            or y >= (img_height - border_margin_px)
        ):
            logger.debug(f"Keypoint descartado por borde de imagen: {lbl} en ({x:.1f}, {y:.1f})")
            continue

        # 2. Verificar presencia de césped solo en zonas periféricas/vallas publicitarias.
        # Un keypoint dentro del campo no puede ser un banner comercial y no debe descartarse
        # por variaciones de iluminación o césped desaturado en estadios nublados.
        top_banner_limit = float(img_height) * 0.22
        is_peripheral = (y <= top_banner_limit) or (x <= (img_width * 0.05)) or (x >= (img_width * 0.95))

        if frame_bgr is not None and is_peripheral:
            ix, iy = int(round(x)), int(round(y))
            y0, y1 = max(0, iy - patch_radius), min(img_height, iy + patch_radius)
            x0, x1 = max(0, ix - patch_radius), min(img_width, ix + patch_radius)
            patch = frame_bgr[y0:y1, x0:x1]
            if patch.size > 0:
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                # Rango ampliado de césped deportivo: H [22, 105], S [10, 255] (soporta césped desaturado/nublado), V [25, 255]
                grass_mask = cv2.inRange(hsv_patch, (22, 10, 25), (105, 255, 255))
                grass_ratio = float(np.count_nonzero(grass_mask)) / float(grass_mask.size)
                if grass_ratio < min_grass_ratio:
                    logger.debug(
                        f"Keypoint {lbl} en ({x:.1f}, {y:.1f}) descartado por banner/overlay "
                        f"(grass_ratio={grass_ratio:.2f} < {min_grass_ratio})"
                    )
                    continue

        filtered.append(item)
    return filtered


def filter_spatial_swaps_and_hallucinations(
    candidates: list,
    field_coords: dict,
    min_field_dist_m: float = 12.0,
    min_pixel_dist_for_far_points: float = 120.0,
) -> list:
    if len(candidates) < 4:
        return candidates

    bad_indices = set()
    n = len(candidates)

    def _eff_conf(conf: float, lbl: str) -> float:
        return conf * 0.85 if lbl in CORNER_LABELS else conf

    for i in range(n):
        idx_i, xy_i, conf_i, lbl_i = candidates[i]
        px_i, py_i = float(xy_i[0]), float(xy_i[1])
        if lbl_i not in field_coords:
            continue
        fx_i, fy_i = field_coords[lbl_i]
        eff_i = _eff_conf(conf_i, lbl_i)

        conflicts = 0
        for j in range(n):
            if i == j:
                continue
            idx_j, xy_j, conf_j, lbl_j = candidates[j]
            px_j, py_j = float(xy_j[0]), float(xy_j[1])
            if lbl_j not in field_coords:
                continue
            fx_j, fy_j = field_coords[lbl_j]
            eff_j = _eff_conf(conf_j, lbl_j)

            dist_field = float(np.hypot(fx_i - fx_j, fy_i - fy_j))
            dist_px = float(np.hypot(px_i - px_j, py_i - py_j))

            # Regla 1: Colapso espacial (puntos lejanos en campo pero muy juntos en imagen)
            if dist_field >= min_field_dist_m and dist_px < min_pixel_dist_for_far_points:
                if eff_i <= eff_j:
                    conflicts += 2 if (lbl_i in CORNER_LABELS and conf_j > 0.70) else 1

            # Regla 2: Monotonía vertical en Y (misma línea longitudinal X en campo)
            if abs(fx_i - fx_j) <= 3.0 and abs(fy_i - fy_j) >= 5.0:
                if (fy_i < fy_j and py_i > py_j - 15.0) or (fy_i > fy_j and py_i < py_j + 15.0):
                    if eff_i <= eff_j:
                        conflicts += 1

            # Regla 3: Monotonía horizontal en X (profundidad Y similar en campo)
            if abs(fy_i - fy_j) <= 6.0 and abs(fx_i - fx_j) >= 10.0:
                if (fx_i < fx_j and px_i > px_j - 15.0) or (fx_i > fx_j and px_i < px_j + 15.0):
                    if eff_i <= eff_j:
                        conflicts += 1

        if conflicts >= 2:
            logger.debug(
                f"Keypoint {lbl_i} descartado por incoherencia topológica ({conflicts} conflictos)."
            )
            bad_indices.add(i)

    return [c for i, c in enumerate(candidates) if i not in bad_indices]

