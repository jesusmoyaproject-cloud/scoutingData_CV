"""
ScoutingData v5.0 - Overlays de Visualización (Optimizados)

OPT-4: get_pitch_field_points() — pre-calcula puntos de cancha una sola vez.
OPT-6: Renders in-place (sin .copy() en cada función).
"""

import cv2
import numpy as np
from typing import Optional, List
from homography.mappings import KP_MAP
from config.field_dimensions import FieldDimensions

# ── OPT-4: Caché de puntos de campo ──────────────────────────────────────────
_FIELD_POINTS_CACHE: dict = {}


def get_pitch_field_points(dims: FieldDimensions = FieldDimensions()) -> List[np.ndarray]:
    """
    Calcula UNA VEZ los segmentos de líneas de la cancha en coordenadas de campo (metros).
    Retorna lista de arrays float32 para cv2.perspectiveTransform.
    Solo se recalcula si cambian las dimensiones del campo.
    """
    key = (dims.length, dims.width)
    if key in _FIELD_POINTS_CACHE:
        return _FIELD_POINTS_CACHE[key]

    segments: List[np.ndarray] = []

    def seg(*pts):
        segments.append(np.array(pts, dtype=np.float32).reshape(-1, 1, 2))

    L, W = dims.length, dims.width
    PBL  = dims.penalty_box_length
    PBW  = dims.penalty_box_width
    GBL  = dims.goal_box_length
    GBW  = dims.goal_box_width
    PSD  = dims.penalty_spot_distance
    CR   = dims.centre_circle_radius

    # Borde exterior
    seg((0,0),(L,0),(L,W),(0,W),(0,0))
    # Línea de medio campo
    seg((L/2,0),(L/2,W))
    # Círculo central (36 puntos → triángulos suaves, más rápido que 360)
    angles = np.linspace(0, 2*np.pi, 73)
    circle_pts = np.column_stack([
        L/2 + CR*np.cos(angles),
        W/2 + CR*np.sin(angles),
    ]).astype(np.float32).reshape(-1,1,2)
    segments.append(circle_pts)
    # Áreas de penalti
    seg((0, W/2-PBW/2),(PBL,W/2-PBW/2),(PBL,W/2+PBW/2),(0,W/2+PBW/2))
    seg((L, W/2-PBW/2),(L-PBL,W/2-PBW/2),(L-PBL,W/2+PBW/2),(L,W/2+PBW/2))
    # Áreas de meta
    seg((0,W/2-GBW/2),(GBL,W/2-GBW/2),(GBL,W/2+GBW/2),(0,W/2+GBW/2))
    seg((L,W/2-GBW/2),(L-GBL,W/2-GBW/2),(L-GBL,W/2+GBW/2),(L,W/2+GBW/2))
    # Arco de penalti izquierdo
    left_arc_a = np.deg2rad(np.arange(-53, 54))
    left_arc   = np.column_stack([
        PSD + CR*np.cos(left_arc_a),
        W/2 + CR*np.sin(left_arc_a),
    ]).astype(np.float32).reshape(-1,1,2)
    segments.append(left_arc)
    # Arco de penalti derecho
    right_arc_a = np.deg2rad(np.arange(127, 234))
    right_arc   = np.column_stack([
        L-PSD + CR*np.cos(right_arc_a),
        W/2   + CR*np.sin(right_arc_a),
    ]).astype(np.float32).reshape(-1,1,2)
    segments.append(right_arc)

    _FIELD_POINTS_CACHE[key] = segments
    return segments


def draw_pitch_lines_on_image(
    img: np.ndarray,
    H: np.ndarray,
    dims: FieldDimensions = FieldDimensions(),
    precomputed_segments: Optional[List[np.ndarray]] = None,
    color: tuple = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Proyecta líneas de la cancha sobre la imagen de cámara usando H_inv.
    OPT-4: Si se pasan `precomputed_segments`, evita recalcular la geometría.
    OPT-6: Opera in-place sobre `img` (sin crear una copia).
    """
    _, H_inv = cv2.invert(H)
    segs = precomputed_segments if precomputed_segments is not None else get_pitch_field_points(dims)
    for pts in segs:
        projected = cv2.perspectiveTransform(pts, H_inv)
        cv2.polylines(img, [projected.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return img


def draw_calibration_overlay(
    img: np.ndarray,
    result,
    mask: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """Dibuja keypoints detectados. OPT-6: in-place."""
    if result.keypoints is None:
        return img
    kps_xy   = result.keypoints.xy[0].cpu().numpy()
    kps_conf = result.keypoints.conf[0].cpu().numpy()
    used = 0
    for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
        if conf < conf_threshold:
            continue
        x, y = int(xy[0]), int(xy[1])
        label = KP_MAP.get(idx, "??")
        if mask is not None:
            color = (0, 255, 0) if bool(mask[used][0]) else (0, 0, 255)
            used += 1
        else:
            color = (255, 0, 0)
        cv2.circle(img, (x, y), 8, color, -1)
        cv2.circle(img, (x, y), 9, (255, 255, 255), 1)
        text = f"Y:{idx} R:{label} ({conf:.2f})"
        cv2.putText(img, text, (x+12, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x+12, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return img


def draw_player_detections(
    img: np.ndarray,
    player_result,
    conf_threshold: float = 0.3,
    ellipse_color: tuple = (255, 0, 255),
    text_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Dibuja jugadores detectados. OPT-6: in-place."""
    if player_result.boxes is None or len(player_result.boxes) == 0:
        return img
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
        x1, y1, x2, y2 = box
        xc = int((x1+x2)/2)
        yb = int(y2)
        bw = int(x2-x1)
        aw = max(8, int(bw/2.5))
        ah = max(4, int(aw/2.0))
        cv2.ellipse(img, (xc, yb), (aw, ah), 0, 0, 360, ellipse_color, 2, cv2.LINE_AA)
        cv2.circle(img, (xc, yb), 3, (0, 255, 255), -1, cv2.LINE_AA)
        label = f"ID:{player_id} ({int(conf*100)}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (xc-tw//2-4, yb+6), (xc+tw//2+4, yb+6+th+6), (0,0,0), -1)
        cv2.rectangle(img, (xc-tw//2-4, yb+6), (xc+tw//2+4, yb+6+th+6), ellipse_color, 1)
        cv2.putText(img, label, (xc-tw//2, yb+6+th+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    return img


def draw_ball_detections(
    img: np.ndarray,
    ball_detection: Optional[dict] = None,
    ball_history: Optional[list] = None,
    ball_color: tuple = (0, 255, 255),
    trail_color: tuple = (0, 200, 255),
) -> np.ndarray:
    """Dibuja balón y trayectoria. OPT-6: in-place."""
    if ball_history and len(ball_history) > 1:
        valid = [p for p in ball_history if p is not None]
        for i in range(1, len(valid)):
            pt1 = (int(valid[i-1][0]), int(valid[i-1][1]))
            pt2 = (int(valid[i][0]),   int(valid[i][1]))
            alpha = float(i) / float(len(valid))
            cv2.line(img, pt1, pt2, trail_color, max(1, int(alpha*3)), cv2.LINE_AA)
    if ball_detection is not None:
        px = int(ball_detection["pixel_x"])
        py = int(ball_detection["pixel_y"])
        conf = ball_detection["confidence"]
        is_interp = ball_detection.get("interpolated", False)
        color  = (0, 165, 255) if is_interp else ball_color
        radius = 8
        cv2.circle(img, (px, py), radius+4, (0,0,0), 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), 2, (255,255,255), -1, cv2.LINE_AA)
        tag   = " (est)" if is_interp else ""
        label = f"Balon{tag} {int(conf*100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (px-tw//2-3, py-radius-th-8), (px+tw//2+3, py-radius-2), (0,0,0), -1)
        cv2.rectangle(img, (px-tw//2-3, py-radius-th-8), (px+tw//2+3, py-radius-2), color, 1)
        cv2.putText(img, label, (px-tw//2, py-radius-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    return img
