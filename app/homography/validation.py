import cv2
import numpy as np
from typing import Tuple, Dict

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
