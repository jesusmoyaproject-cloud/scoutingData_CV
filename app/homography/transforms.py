import cv2
import numpy as np
from typing import Tuple

def pixel_to_field(x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
    """
    Transforms pixel coordinates (image space) into field coordinates in meters.
    """
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    return float(transformed[0][0][0]), float(transformed[0][0][1])

def field_to_pixel(x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
    """
    Transforms field coordinates in meters into pixel coordinates (image space).
    """
    _, H_inv = cv2.invert(H)
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H_inv)
    return float(transformed[0][0][0]), float(transformed[0][0][1])
