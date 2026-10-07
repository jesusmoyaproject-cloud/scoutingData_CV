"""
ScoutingData v5.0 - Estimador de Homografía (Robustecido)
"""
import logging
import cv2
import numpy as np
from typing import Tuple, List
from config.field_dimensions import FieldDimensions
from config.field_points import FIELD_COORDS
from homography.mappings import KP_MAP, LEFT_KEYPOINTS, RIGHT_KEYPOINTS, CENTER_KEYPOINTS
from homography.validation import validate_homography_matrix

logger = logging.getLogger("homography.estimator")

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
    conf_threshold: float = 0.5,
    ransac_threshold: float = 5.0,
    enable_side_filter: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
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

    image_pts_list = [xy for _, xy, _, _ in valid_candidates]
    field_pts_list = [FIELD_COORDS[lbl] for _, _, _, lbl in valid_candidates]

    image_pts = np.array(image_pts_list, dtype=np.float32)
    field_pts = np.array(field_pts_list, dtype=np.float32)

    if len(image_pts) < 4:
        raise ValueError(f"At least 4 keypoints required to compute homography. Found: {len(image_pts)}")

    H, mask = cv2.findHomography(
        image_pts,
        field_pts,
        cv2.RANSAC,
        ransac_threshold
    )

    if H is None or not validate_homography_matrix(H):
        raise ValueError("Homography calculation failed (returned None or invalid matrix).")

    inliers_count = int(mask.sum())
    outliers_count = len(mask) - inliers_count

    logger.info(f"Homography calculated. Inliers: {inliers_count}/{len(image_pts)} (Outliers: {outliers_count})")

    return H, image_pts, field_pts, mask, inliers_count, outliers_count
