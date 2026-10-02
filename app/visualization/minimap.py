"""
ScoutingData v5.0 - Minimap (Optimizado)

OPT-5: get_cached_pitch_template() — renderiza el fondo de la cancha una sola vez.
"""

import cv2
import numpy as np
from typing import Optional
from config.field_dimensions import FieldDimensions
from config.field_points import FIELD_COORDS
from homography.mappings import KP_MAP
from homography.transforms import pixel_to_field
from visualization.pitch import draw_pitch_template

# OPT-5: Cache del template base de la cancha
_MINIMAP_TEMPLATE_CACHE: dict = {}


def get_cached_pitch_template(
    dims: FieldDimensions = FieldDimensions(),
    scale: float = 8.0,
    padding_meters: float = 5.0,
) -> np.ndarray:
    """
    Renderiza el template base de la cancha UNA SOLA VEZ y lo cachea.
    Cada frame, basta con hacer .copy() del template y dibujar encima.
    """
    key = (dims.length, dims.width, scale, padding_meters)
    if key not in _MINIMAP_TEMPLATE_CACHE:
        img, _, _ = draw_pitch_template(dims=dims, scale=scale, padding_meters=padding_meters)
        _MINIMAP_TEMPLATE_CACHE[key] = img
    return _MINIMAP_TEMPLATE_CACHE[key]


def draw_minimap_with_projections(
    H: np.ndarray,
    result,
    conf_threshold: float = 0.5,
    dims: FieldDimensions = FieldDimensions(),
    scale: float = 8.0,
    padding_meters: float = 5.0,
    mask: Optional[np.ndarray] = None,
    cached_template: Optional[np.ndarray] = None,   # OPT-5
) -> np.ndarray:
    """
    Genera el minimap proyectando los keypoints detectados.
    OPT-5: Si se proporciona cached_template, no re-renderiza la cancha.
    """
    if cached_template is not None:
        minimap_img = cached_template.copy()
        pad_px = int(padding_meters * scale)
        scale_factor = scale
    else:
        minimap_img, pad_px, scale_factor = draw_pitch_template(
            dims=dims, scale=scale, padding_meters=padding_meters
        )
        pad_px = int(pad_px)

    def f2px(x_m, y_m):
        return (int(x_m * scale_factor) + pad_px, int(y_m * scale_factor) + pad_px)

    # Puntos GT detectados
    detected_labels = set()
    if (
        result is not None
        and result.keypoints is not None
        and len(result.keypoints) > 0
        and result.keypoints.conf is not None
        and len(result.keypoints.conf) > 0
    ):
        kps_conf = result.keypoints.conf[0].cpu().numpy()
        for idx, conf in enumerate(kps_conf):
            if conf >= conf_threshold and idx in KP_MAP:
                detected_labels.add(KP_MAP[idx])

    for label, (gt_x, gt_y) in FIELD_COORDS.items():
        gt_px = f2px(gt_x, gt_y)
        if label in detected_labels:
            cv2.circle(minimap_img, gt_px, 10, (0,255,255), 1, cv2.LINE_AA)
            cv2.circle(minimap_img, gt_px, 4,  (0,255,255), -1, cv2.LINE_AA)
        else:
            cv2.circle(minimap_img, gt_px, 4, (255,100,0), -1, cv2.LINE_AA)

    if (
        result is None
        or result.keypoints is None
        or len(result.keypoints) == 0
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        return minimap_img

    kps_xy   = result.keypoints.xy[0].cpu().numpy()
    kps_conf = result.keypoints.conf[0].cpu().numpy()
    if len(kps_xy) == 0:
        return minimap_img
    used = 0
    for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
        if conf < conf_threshold:
            continue
        label = KP_MAP.get(idx)
        if label is None:
            continue
        fx, fy = pixel_to_field(xy[0], xy[1], H)
        px_x, px_y = f2px(fx, fy)
        color = (0,255,0) if (mask is not None and bool(mask[used][0])) else (255,128,0)
        if mask is not None:
            used += 1
        cv2.circle(minimap_img, (px_x, px_y), 9, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(minimap_img, (px_x, px_y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(minimap_img, label, (px_x+10, px_y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    return minimap_img


def draw_player_projections_on_minimap(
    minimap_img: np.ndarray,
    player_result,
    H: np.ndarray,
    conf_threshold: float = 0.3,
    dims: FieldDimensions = FieldDimensions(),
    scale: float = 8.0,
    padding_meters: float = 5.0,
    player_color: tuple = (255, 0, 255),
) -> np.ndarray:
    """OPT-6: in-place sobre minimap_img."""
    if player_result.boxes is None or len(player_result.boxes) == 0:
        return minimap_img
    pad_px = int(padding_meters * scale)
    def f2px(x_m, y_m):
        return (int(x_m * scale) + pad_px, int(y_m * scale) + pad_px)

    boxes     = player_result.boxes.xyxy.cpu().numpy()
    confs     = player_result.boxes.conf.cpu().numpy()
    track_ids = (
        player_result.boxes.id.cpu().numpy().astype(int)
        if player_result.boxes.id is not None else None
    )
    for idx, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < conf_threshold:
            continue
        player_id = track_ids[idx] if track_ids is not None else idx
        x1,y1,x2,y2 = box
        xc = float((x1+x2)/2)
        yb = float(y2)
        fx, fy = pixel_to_field(xc, yb, H)
        if not (-5 <= fx <= dims.length+5 and -5 <= fy <= dims.width+5):
            continue
        px_x, px_y = f2px(fx, fy)
        cv2.circle(minimap_img, (px_x, px_y), 8, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(minimap_img, (px_x, px_y), 5, player_color, -1, cv2.LINE_AA)
        cv2.putText(minimap_img, f"P{player_id}", (px_x+8, px_y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
    return minimap_img


def draw_ball_projection_on_minimap(
    minimap_img: np.ndarray,
    ball_detection: Optional[dict] = None,
    scale: float = 8.0,
    padding_meters: float = 5.0,
    ball_color: tuple = (0, 255, 255),
) -> np.ndarray:
    """OPT-6: in-place sobre minimap_img."""
    if ball_detection is None:
        return minimap_img
    px = ball_detection.get("pitch_x_m")
    py = ball_detection.get("pitch_y_m")
    if px is None or py is None:
        return minimap_img
    pad_px = int(padding_meters * scale)
    px_x = int(px * scale) + pad_px
    px_y = int(py * scale) + pad_px
    cv2.circle(minimap_img, (px_x, px_y), 7, (0,0,0), 1, cv2.LINE_AA)
    cv2.circle(minimap_img, (px_x, px_y), 5, ball_color, -1, cv2.LINE_AA)
    cv2.circle(minimap_img, (px_x, px_y), 2, (255,255,255), -1, cv2.LINE_AA)
    cv2.putText(minimap_img, "BALL", (px_x+7, px_y+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1, cv2.LINE_AA)
    return minimap_img
