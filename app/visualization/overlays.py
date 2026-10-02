import cv2
import numpy as np
from typing import Optional
from homography.mappings import KP_MAP
from config.field_dimensions import FieldDimensions

def draw_pitch_lines_on_image(
    img: np.ndarray,
    H: np.ndarray,
    dims: FieldDimensions = FieldDimensions(),
    color: tuple = (0, 255, 255),  # Cyan/yellow overlay
    thickness: int = 2
) -> np.ndarray:
    """
    Projects the entire 2D soccer field markings (lines, box, circles) 
    onto the camera image using the homography matrix H.
    """
    annotated_img = img.copy()
    _, H_inv = cv2.invert(H)

    def draw_poly(pts_m, is_closed=False):
        pts_arr = np.array(pts_m, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts_arr, H_inv).reshape(-1, 2)
        # Filter points that project behind the camera if any (we check if we get extreme values)
        # Draw as polyline
        cv2.polylines(annotated_img, [projected.astype(np.int32)], is_closed, color, thickness, cv2.LINE_AA)

    # 1. Outer boundary
    boundary = [(0.0, 0.0), (dims.length, 0.0), (dims.length, dims.width), (0.0, dims.width)]
    draw_poly(boundary, is_closed=True)

    # 2. Midfield line
    midfield = [(dims.length / 2.0, 0.0), (dims.length / 2.0, dims.width)]
    draw_poly(midfield, is_closed=False)

    # 3. Center circle
    center_circle = []
    for deg in range(361):
        rad = np.deg2rad(deg)
        cx = dims.length / 2.0 + dims.centre_circle_radius * np.cos(rad)
        cy = dims.width / 2.0 + dims.centre_circle_radius * np.sin(rad)
        center_circle.append((cx, cy))
    draw_poly(center_circle, is_closed=True)

    # 4. Left areas
    l_penalty = [
        (0.0, dims.width / 2.0 - dims.penalty_box_width / 2.0),
        (dims.penalty_box_length, dims.width / 2.0 - dims.penalty_box_width / 2.0),
        (dims.penalty_box_length, dims.width / 2.0 + dims.penalty_box_width / 2.0),
        (0.0, dims.width / 2.0 + dims.penalty_box_width / 2.0)
    ]
    draw_poly(l_penalty, is_closed=False)

    l_goal = [
        (0.0, dims.width / 2.0 - dims.goal_box_width / 2.0),
        (dims.goal_box_length, dims.width / 2.0 - dims.goal_box_width / 2.0),
        (dims.goal_box_length, dims.width / 2.0 + dims.goal_box_width / 2.0),
        (0.0, dims.width / 2.0 + dims.goal_box_width / 2.0)
    ]
    draw_poly(l_goal, is_closed=False)

    # 5. Right areas
    r_penalty = [
        (dims.length, dims.width / 2.0 - dims.penalty_box_width / 2.0),
        (dims.length - dims.penalty_box_length, dims.width / 2.0 - dims.penalty_box_width / 2.0),
        (dims.length - dims.penalty_box_length, dims.width / 2.0 + dims.penalty_box_width / 2.0),
        (dims.length, dims.width / 2.0 + dims.penalty_box_width / 2.0)
    ]
    draw_poly(r_penalty, is_closed=False)

    r_goal = [
        (dims.length, dims.width / 2.0 - dims.goal_box_width / 2.0),
        (dims.length - dims.goal_box_length, dims.width / 2.0 - dims.goal_box_width / 2.0),
        (dims.length - dims.goal_box_length, dims.width / 2.0 + dims.goal_box_width / 2.0),
        (dims.length, dims.width / 2.0 + dims.goal_box_width / 2.0)
    ]
    draw_poly(r_goal, is_closed=False)

    # 6. Penalty arcs
    left_arc = []
    for deg in range(-53, 54):
        rad = np.deg2rad(deg)
        ax = dims.penalty_spot_distance + dims.centre_circle_radius * np.cos(rad)
        ay = dims.width / 2.0 + dims.centre_circle_radius * np.sin(rad)
        left_arc.append((ax, ay))
    draw_poly(left_arc, is_closed=False)

    right_arc = []
    for deg in range(127, 234):
        rad = np.deg2rad(deg)
        ax = dims.length - dims.penalty_spot_distance + dims.centre_circle_radius * np.cos(rad)
        ay = dims.width / 2.0 + dims.centre_circle_radius * np.sin(rad)
        right_arc.append((ax, ay))
    draw_poly(right_arc, is_closed=False)

    return annotated_img

def draw_calibration_overlay(
    img: np.ndarray,
    result,
    mask: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5
) -> np.ndarray:
    """
    Draws the detected keypoints and their details on the original camera image.
    """
    annotated_img = img.copy()
    
    if result.keypoints is None:
        return annotated_img
        
    kps_xy = result.keypoints.xy[0].cpu().numpy()
    kps_conf = result.keypoints.conf[0].cpu().numpy()
    
    used_counter = 0
    for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
        if conf < conf_threshold:
            continue
            
        x, y = int(xy[0]), int(xy[1])
        label_rf = KP_MAP.get(idx, "??")
        
        if mask is not None:
            is_inlier = bool(mask[used_counter][0])
            color = (0, 255, 0) if is_inlier else (0, 0, 255)
            used_counter += 1
        else:
            color = (255, 0, 0)
            
        cv2.circle(annotated_img, (x, y), 8, color, -1)
        cv2.circle(annotated_img, (x, y), 9, (255, 255, 255), 1)
        
        text = f"Y:{idx} R:{label_rf} ({conf:.2f})"
        
        cv2.putText(
            annotated_img,
            text,
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )
        
        cv2.putText(
            annotated_img,
            text,
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated_img

def draw_player_detections(
    img: np.ndarray,
    player_result,
    conf_threshold: float = 0.3,
    ellipse_color: tuple = (255, 0, 255),  # Magenta/Purple BGR
    text_color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    Draws player foot circles/ellipses with persistent track ID and confidence percentage 
    at the bottom-center of each player's bounding box.
    """
    annotated_img = img.copy()
    if player_result.boxes is None or len(player_result.boxes) == 0:
        return annotated_img

    boxes = player_result.boxes.xyxy.cpu().numpy()
    confs = player_result.boxes.conf.cpu().numpy()
    classes = player_result.boxes.cls.cpu().numpy()
    
    # Extract persistent tracking IDs if available (from model.track)
    track_ids = (
        player_result.boxes.id.cpu().numpy().astype(int)
        if player_result.boxes.id is not None
        else None
    )

    for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
        if conf < conf_threshold:
            continue

        player_id = track_ids[idx] if track_ids is not None else idx

        x1, y1, x2, y2 = box
        x_center = int((x1 + x2) / 2.0)
        y_bottom = int(y2)
        box_width = int(x2 - x1)

        # Foot ellipse radius based on box width
        axes_width = max(8, int(box_width / 2.5))
        axes_height = max(4, int(axes_width / 2.0))

        # Draw foot ellipse
        cv2.ellipse(
            annotated_img,
            (x_center, y_bottom),
            (axes_width, axes_height),
            0, 0, 360,
            ellipse_color,
            2,
            cv2.LINE_AA
        )

        # Draw foot center point
        cv2.circle(annotated_img, (x_center, y_bottom), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # Label: Persistent Track ID and confidence percentage
        label_text = f"ID:{player_id} ({int(conf * 100)}%)"

        # Text background badge
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx_bg_top_left = (x_center - tw // 2 - 4, y_bottom + 6)
        tx_bg_bot_right = (x_center + tw // 2 + 4, y_bottom + 6 + th + 6)

        cv2.rectangle(annotated_img, tx_bg_top_left, tx_bg_bot_right, (0, 0, 0), -1)
        cv2.rectangle(annotated_img, tx_bg_top_left, tx_bg_bot_right, ellipse_color, 1)

        # Text label
        cv2.putText(
            annotated_img,
            label_text,
            (x_center - tw // 2, y_bottom + 6 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA
        )

    return annotated_img


def draw_ball_detections(
    img: np.ndarray,
    ball_detection: Optional[dict] = None,
    ball_history: Optional[list] = None,
    ball_color: tuple = (0, 255, 255),  # Amarillo brillante BGR
    trail_color: tuple = (0, 200, 255)
) -> np.ndarray:
    """
    Dibuja la detección del balón con marcador animado/resaltado y su trayectoria (trail) de frames previos.
    """
    annotated_img = img.copy()

    # 1. Dibujar trayectoria del balón si se proporciona historia
    if ball_history is not None and len(ball_history) > 1:
        valid_points = [p for p in ball_history if p is not None]
        for i in range(1, len(valid_points)):
            pt1 = (int(valid_points[i - 1][0]), int(valid_points[i - 1][1]))
            pt2 = (int(valid_points[i][0]), int(valid_points[i][1]))
            alpha = float(i) / float(len(valid_points))
            thickness = max(1, int(alpha * 3))
            cv2.line(annotated_img, pt1, pt2, trail_color, thickness, cv2.LINE_AA)

    # 2. Dibujar la ubicación actual del balón
    if ball_detection is not None:
        px = int(ball_detection["pixel_x"])
        py = int(ball_detection["pixel_y"])
        conf = ball_detection["confidence"]
        is_interpolated = ball_detection.get("interpolated", False)

        # Estilo de circulo según si es detectado u obtenido por rastreador/interpolador
        radius = 8
        color = (0, 165, 255) if is_interpolated else ball_color  # Naranja si es interpolado, amarillo si es directo

        cv2.circle(annotated_img, (px, py), radius + 4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(annotated_img, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(annotated_img, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Etiqueta de Balón
        status_tag = " (est)" if is_interpolated else ""
        label_text = f"Balon{status_tag} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        tx_bg_top_left = (px - tw // 2 - 3, py - radius - th - 8)
        tx_bg_bot_right = (px + tw // 2 + 3, py - radius - 2)

        cv2.rectangle(annotated_img, tx_bg_top_left, tx_bg_bot_right, (0, 0, 0), -1)
        cv2.rectangle(annotated_img, tx_bg_top_left, tx_bg_bot_right, color, 1)
        cv2.putText(
            annotated_img,
            label_text,
            (px - tw // 2, py - radius - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated_img


def draw_event_banner_overlay(
    img: np.ndarray,
    active_events: list
) -> np.ndarray:
    """
    Dibuja un banner visual destacado en la parte superior del fotograma cuando se detecta un evento.
    Muestra el tipo de evento, confianza y si es un pase, el ID del emisor y receptor.
    """
    if not active_events:
        return img

    annotated_img = img.copy()
    h, w, _ = annotated_img.shape

    # Colores por tipo de evento (BGR)
    COLOR_MAP = {
        "Gol": (0, 215, 255),          # Dorado / Amarillo brillante
        "Remate/Disparo": (0, 0, 255), # Rojo
        "Pase": (255, 191, 0),         # Cyan / Azul claro
        "Atajada": (0, 255, 0),        # Verde
        "Falta": (0, 140, 255),        # Naranja
        "Tarjeta": (0, 255, 255),      # Amarillo
        "Fuera de Juego": (128, 0, 128)# Púrpura
    }

    y_offset = 20
    for evt in active_events[-2:]: # Mostrar como máximo 2 eventos recientes simultáneos
        event_name = evt.get("event_type", "Evento")
        conf = evt.get("confidence", 0.0)
        p_from = evt.get("pass_from_player_id")
        p_to = evt.get("pass_to_player_id")

        theme_color = COLOR_MAP.get(event_name, (0, 255, 255))

        line1 = f"EVENTO: {event_name.upper()} ({int(conf * 100)}%)"
        if event_name == "Pase" and (p_from is not None or p_to is not None):
            line2 = f"Pase: Jugador #{p_from if p_from is not None else '?'} -> Jugador #{p_to if p_to is not None else '?'}"
        else:
            line2 = None

        (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        if line2:
            (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            box_w = max(w1, w2) + 30
            box_h = h1 + h2 + 25
        else:
            box_w = w1 + 30
            box_h = h1 + 20

        x1 = (w - box_w) // 2
        y1 = y_offset
        x2 = x1 + box_w
        y2 = y1 + box_h

        # Fondo semitransparente
        overlay = annotated_img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, annotated_img, 0.25, 0, annotated_img)

        # Borde brillante y acento lateral
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), theme_color, 2)
        cv2.rectangle(annotated_img, (x1, y1), (x1 + 8, y2), theme_color, -1)

        # Texto Línea 1
        cv2.putText(
            annotated_img,
            line1,
            (x1 + 18, y1 + h1 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Texto Línea 2 si aplica
        if line2:
            cv2.putText(
                annotated_img,
                line2,
                (x1 + 18, y1 + h1 + h2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 230, 255),
                1,
                cv2.LINE_AA
            )

        y_offset += box_h + 10

    return annotated_img





