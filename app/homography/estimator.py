"""
ScoutingData v5.0 - Estimador de Homografía (Robustecido con USAC_MAGSAC y Diagnóstico Integral)
"""
import logging
import cv2
import numpy as np
from typing import Tuple, List, Optional
from config.field_dimensions import FieldDimensions
from config.field_points import FIELD_COORDS
from homography.mappings import KP_MAP, LEFT_KEYPOINTS, RIGHT_KEYPOINTS, CENTER_KEYPOINTS
from homography.validation import validate_homography_matrix
from homography.geometric_validator import (
    validate_keypoint_geometry,
    filter_border_keypoints,
    filter_spatial_swaps_and_hallucinations,
    compute_convex_hull_and_coverage,
    compute_collinearity_score,
)
from homography.metrics import HomographyDiagnostics, compute_reprojection_errors
from homography.diagnostics_tool import compute_point_residuals

logger = logging.getLogger("homography.estimator")

# Seleccionar el método más robusto disponible en la versión de OpenCV
if hasattr(cv2, "USAC_MAGSAC"):
    DEFAULT_HOMOGRAPHY_METHOD = cv2.USAC_MAGSAC
else:
    DEFAULT_HOMOGRAPHY_METHOD = cv2.RANSAC


class HomographyEstimateResult(tuple):
    """
    Tupla enriquecida compatible con desempaquetado de 6 valores clásico:
    H, image_pts, field_pts, mask, inliers, outliers = compute_homography(...)
    Además expone los atributos .diagnostics, .valid_candidates y .residuals para telemetría avanzada.
    """
    H: np.ndarray
    image_pts: np.ndarray
    field_pts: np.ndarray
    mask: np.ndarray
    inliers_count: int
    outliers_count: int
    diagnostics: HomographyDiagnostics
    valid_candidates: list
    residuals: list

    def __new__(cls, H, image_pts, field_pts, mask, inliers_count, outliers_count, diagnostics=None, valid_candidates=None, residuals=None):
        instance = super().__new__(cls, (H, image_pts, field_pts, mask, inliers_count, outliers_count))
        instance.H = H
        instance.image_pts = image_pts
        instance.field_pts = field_pts
        instance.mask = mask
        instance.inliers_count = inliers_count
        instance.outliers_count = outliers_count
        instance.diagnostics = diagnostics
        instance.valid_candidates = valid_candidates if valid_candidates is not None else []
        instance.residuals = residuals if residuals is not None else []
        return instance


def filter_keypoints_by_pitch_side(
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_threshold: float = 0.35,
    min_dominant_points: int = 4
) -> Tuple[List[Tuple[int, np.ndarray, float, str]], bool, bool]:
    """
    Filtra keypoints descartando alucinaciones del arco opuesto cuando una mitad del campo es claramente dominante.
    Retorna: (candidates, discarded_left, discarded_right)
    donde candidates es una lista de (idx, xy, conf, label) ordenados por idx.
    """
    candidates: List[Tuple[int, np.ndarray, float, str]] = []
    left_cands: List[Tuple[int, np.ndarray, float, str]] = []
    right_cands: List[Tuple[int, np.ndarray, float, str]] = []
    center_cands: List[Tuple[int, np.ndarray, float, str]] = []

    for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
        if conf < conf_threshold:
            continue
        if idx not in KP_MAP:
            continue
        label = KP_MAP[idx]
        if label not in FIELD_COORDS:
            continue

        cand = (idx, xy, float(conf), label)
        if label in LEFT_KEYPOINTS:
            left_cands.append(cand)
        elif label in RIGHT_KEYPOINTS:
            right_cands.append(cand)
        else:
            center_cands.append(cand)

    n_left = len(left_cands)
    n_right = len(right_cands)
    sum_left = sum(c[2] for c in left_cands)
    sum_right = sum(c[2] for c in right_cands)

    discard_left = False
    discard_right = False

    # Criterio de dominancia espacial: si una mitad del campo tiene >= 4 puntos válidos
    # y supera contundentemente a la otra, la cámara está en plano corto sobre ese arco.
    if n_right >= min_dominant_points and (n_left == 0 or sum_right >= 2.0 * sum_left or n_right >= n_left + 4):
        discard_left = True
    elif n_left >= min_dominant_points and (n_right == 0 or sum_left >= 2.0 * sum_right or n_left >= n_right + 4):
        discard_right = True

    if not discard_left:
        candidates.extend(left_cands)
    if not discard_right:
        candidates.extend(right_cands)
    candidates.extend(center_cands)

    candidates.sort(key=lambda x: x[0])
    return candidates, discard_left, discard_right


def compute_homography(
    result,
    conf_threshold: float = 0.35,
    ransac_threshold: float = 0.8,       # Umbral calibrado en metros (0.8m en terreno)
    enable_side_filter: bool = True,
    filter_border_margin_px: float = 12.0, # Descartar keypoints truncados en bordes de imagen
    img_width: int = 1920,
    img_height: int = 1080,
    validate_geometry: bool = True,
    method: int = DEFAULT_HOMOGRAPHY_METHOD,
    frame_bgr: Optional[np.ndarray] = None,
) -> HomographyEstimateResult:
    """
    Estima la homografía (imagen -> terreno) con validación geométrica previa,
    algoritmo USAC_MAGSAC robusto y telemetría de diagnóstico.
    """
    if (
        result is None
        or result.keypoints is None
        or len(result.keypoints) == 0
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        raise ValueError("YOLO keypoints data is empty/None.")

    kps_xy = result.keypoints.xy[0].cpu().numpy()
    kps_conf = result.keypoints.conf[0].cpu().numpy()

    if len(kps_xy) == 0:
        raise ValueError("YOLO keypoints array is empty.")

    # 1. Filtro de lado dominante de campo
    if enable_side_filter:
        valid_candidates, disc_l, disc_r = filter_keypoints_by_pitch_side(
            kps_xy, kps_conf, conf_threshold=conf_threshold
        )
        if disc_l:
            logger.debug("Pitch side filter: mitad derecha dominante, descartados puntos del arco izquierdo.")
        elif disc_r:
            logger.debug("Pitch side filter: mitad izquierda dominante, descartados puntos del arco derecho.")
    else:
        valid_candidates = []
        for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
            if conf < conf_threshold or idx not in KP_MAP:
                continue
            lbl = KP_MAP[idx]
            if lbl not in FIELD_COORDS:
                continue
            valid_candidates.append((idx, xy, float(conf), lbl))

    # 2. Filtro de keypoints pegados al borde de pantalla y banners dinámicos
    if filter_border_margin_px > 0 or frame_bgr is not None:
        valid_candidates = filter_border_keypoints(
            valid_candidates,
            img_width=img_width,
            img_height=img_height,
            border_margin_px=filter_border_margin_px,
            frame_bgr=frame_bgr,
        )
    valid_candidates = filter_spatial_swaps_and_hallucinations(valid_candidates, FIELD_COORDS)

    image_pts_list = [xy for _, xy, _, _ in valid_candidates]
    field_pts_list = [FIELD_COORDS[lbl] for _, _, _, lbl in valid_candidates]

    image_pts = np.array(image_pts_list, dtype=np.float32)
    field_pts = np.array(field_pts_list, dtype=np.float32)

    if len(image_pts) < 4:
        raise ValueError(f"At least 4 keypoints required to compute homography. Found: {len(image_pts)}")

    # 3. Validación geométrica previa (Colinealidad, Convex Hull, Dispersión)
    geom_meta = {}
    if validate_geometry:
        is_geom_ok, reason, geom_meta = validate_keypoint_geometry(
            image_pts, field_pts, img_width=img_width, img_height=img_height
        )
        if not is_geom_ok:
            raise ValueError(f"Configuración geométrica degenerada: {reason}")

    # 4. Cálculo de Homografía Robusta (USAC_MAGSAC / RANSAC)
    try:
        H, mask = cv2.findHomography(
            image_pts,
            field_pts,
            method,
            ransac_threshold
        )
    except Exception as exc:
        raise ValueError(f"OpenCV findHomography failed with exception: {exc}")

    if H is None or not validate_homography_matrix(H):
        raise ValueError("Homography calculation failed (returned None or singular/invalid matrix).")

    # Asegurar mask como array 1D booleano/entero
    if mask is None:
        mask = np.ones((len(image_pts), 1), dtype=np.uint8)

    inliers_count = int(mask.sum())
    outliers_count = len(mask) - inliers_count

    # Si RANSAC dejó menos de 4 inliers, la homografía no está sustentada
    if inliers_count < 4:
        raise ValueError(f"Insufficient RANSAC inliers: {inliers_count} < 4")

    # 5. Cálculo de métricas de diagnóstico sobre inliers
    inlier_mask = mask.ravel().astype(bool)
    inlier_img_pts = image_pts[inlier_mask]
    inlier_fld_pts = field_pts[inlier_mask]

    if validate_geometry:
        is_inlier_geom_ok, inlier_reason, geom_meta = validate_keypoint_geometry(
            inlier_img_pts, inlier_fld_pts, img_width=img_width, img_height=img_height
        )
        if not is_inlier_geom_ok:
            raise ValueError(f"Configuración de inliers post-RANSAC degenerada: {inlier_reason}")

    err_info = compute_reprojection_errors(H, inlier_img_pts, inlier_fld_pts)
    cov_info = compute_convex_hull_and_coverage(
        inlier_img_pts, inlier_fld_pts, img_width=img_width, img_height=img_height
    )
    collin_score = compute_collinearity_score(inlier_fld_pts)

    # Condición de la matriz normalizada
    h_norm = H / (H[2, 2] if abs(H[2, 2]) > 1e-9 else 1.0)
    try:
        cond_num = float(np.linalg.cond(h_norm))
    except Exception:
        cond_num = 1e9

    diag = HomographyDiagnostics(
        kps_detected=int((kps_conf >= conf_threshold).sum()),
        kps_used=len(image_pts),
        ransac_inliers=inliers_count,
        ransac_outliers=outliers_count,
        inlier_ratio=float(inliers_count / len(image_pts)),
        mean_reprojection_error_m=err_info["mean_error_m"],
        max_reprojection_error_m=err_info["max_error_m"],
        raw_errors_m=err_info["raw_errors_m"],
        hull_area_image_px2=cov_info["hull_area_img_px2"],
        hull_area_field_m2=cov_info["hull_area_field_m2"],
        image_coverage_ratio=cov_info["image_coverage_ratio"],
        field_coverage_ratio=cov_info["field_coverage_ratio"],
        collinearity_score=collin_score,
        condition_number=cond_num,
        is_geometrically_degenerate=False,
        is_low_longitudinal_span=bool(geom_meta.get("is_low_longitudinal_span", False)) if validate_geometry else False,
        is_valid=True,
        status_reason="OK",
    )

    # 6. Post-filtro: Si el error medio de reproyección es inaceptablemente alto (>2.5m)
    if diag.mean_reprojection_error_m > 2.5:
        raise ValueError(
            f"Mean reprojection error too high ({diag.mean_reprojection_error_m:.2f}m > 2.5m). Rejecting H."
        )

    # 7. Residuales punto a punto para telemetría forense
    residuals = compute_point_residuals(H, valid_candidates, mask)

    logger.debug(
        f"Homography calculated: inliers={inliers_count}/{len(image_pts)} | "
        f"mean_err={diag.mean_reprojection_error_m:.2f}m | max_err={diag.max_reprojection_error_m:.2f}m | "
        f"hull={diag.hull_area_field_m2:.0f}m²"
    )

    return HomographyEstimateResult(
        H, image_pts, field_pts, mask, inliers_count, outliers_count,
        diagnostics=diag, valid_candidates=valid_candidates, residuals=residuals
    )
