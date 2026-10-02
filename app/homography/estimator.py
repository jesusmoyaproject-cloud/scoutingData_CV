"""
ScoutingData v5.0 - Estimador de Homografía (Robustecido)
"""
import logging
import cv2
import numpy as np
from typing import Tuple
from config.field_dimensions import FieldDimensions
from config.field_points import FIELD_COORDS
from homography.mappings import KP_MAP

logger = logging.getLogger("homography.estimator")

def compute_homography(
    result,
    conf_threshold: float = 0.5,
    ransac_threshold: float = 5.0
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

    image_pts_list = []
    field_pts_list = []

    for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
        if conf < conf_threshold:
            continue
        if idx not in KP_MAP:
            continue
        label = KP_MAP[idx]
        if label not in FIELD_COORDS:
            continue

        image_pts_list.append(xy)
        field_pts_list.append(FIELD_COORDS[label])

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

    if H is None:
        raise ValueError("Homography calculation failed (returned None).")

    inliers_count = int(mask.sum())
    outliers_count = len(mask) - inliers_count

    logger.info(f"Homography calculated. Inliers: {inliers_count}/{len(image_pts)} (Outliers: {outliers_count})")

    return H, image_pts, field_pts, mask, inliers_count, outliers_count
