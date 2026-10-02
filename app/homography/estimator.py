import logging
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from homography.mappings import KP_MAP
from config.field_points import FIELD_COORDS

logger = logging.getLogger(__name__)

def compute_homography(
    result, 
    conf_threshold: float = 0.5, 
    ransac_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Computes camera homography using detected keypoints and their real field coordinates.
    
    Args:
        result: Ultralytics YOLO inference result object for a single frame.
        conf_threshold: Threshold to ignore keypoints with low detection confidence.
        ransac_threshold: RANSAC threshold parameter (in meters) for cv2.findHomography.
        
    Returns:
        H: The 3x3 homography transformation matrix.
        image_pts: Numpy array (N, 2) of source image coordinates (pixels).
        field_pts: Numpy array (N, 2) of target field coordinates (meters).
        mask: Mask array representing inliers/outliers.
        inliers_count: Total inlier points.
        outliers_count: Total outlier points.
    """
    if result.keypoints is None:
        raise ValueError("YOLO keypoints data is empty/None.")

    kps_xy = result.keypoints.xy[0].cpu().numpy()
    kps_conf = result.keypoints.conf[0].cpu().numpy()

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
        raise ValueError(
            f"At least 4 keypoints are required to compute homography. Found: {len(image_pts)}"
        )

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
