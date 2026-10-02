import cv2
import numpy as np
from typing import Optional
from config.field_dimensions import FieldDimensions
from config.field_points import FIELD_COORDS
from homography.mappings import KP_MAP
from homography.transforms import pixel_to_field
from visualization.pitch import draw_pitch_template

def draw_minimap_with_projections(
    H: np.ndarray,
    result,
    conf_threshold: float = 0.5,
    dims: FieldDimensions = FieldDimensions(),
    scale: float = 8.0,
    padding_meters: float = 5.0,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generates a minimap and projects detected image keypoints onto it.
    Highlights detected/matched keypoints with larger rings to visually stand out.
    """
    minimap_img, pad_px, scale_factor = draw_pitch_template(
        dims=dims,
        scale=scale,
        padding_meters=padding_meters
    )
    
    def field_to_minimap_px(x_m: float, y_m: float) -> tuple[int, int]:
        px_x = int(x_m * scale_factor) + int(pad_px)
        px_y = int(y_m * scale_factor) + int(pad_px)
        return (px_x, px_y)

    # Detect which labels are present in the image
    detected_labels = set()
    if result.keypoints is not None:
        kps_conf = result.keypoints.conf[0].cpu().numpy()
        for idx, conf in enumerate(kps_conf):
            if conf >= conf_threshold and idx in KP_MAP:
                detected_labels.add(KP_MAP[idx])

    # Draw all Ground Truth locations (highlight the detected ones)
    for label, (gt_x, gt_y) in FIELD_COORDS.items():
        gt_px = field_to_minimap_px(gt_x, gt_y)
        if label in detected_labels:
            # Highlight detected GT point with a yellow ring
            cv2.circle(minimap_img, gt_px, 10, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(minimap_img, gt_px, 4, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            # Normal GT point
            cv2.circle(minimap_img, gt_px, 4, (255, 100, 0), -1, cv2.LINE_AA)
        
    if result.keypoints is None:
        return minimap_img
        
    kps_xy = result.keypoints.xy[0].cpu().numpy()
    kps_conf = result.keypoints.conf[0].cpu().numpy()
    
    used_counter = 0
    for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
        if conf < conf_threshold:
            continue
            
        label = KP_MAP.get(idx, None)
        if label is None:
            continue
            
        field_x, field_y = pixel_to_field(xy[0], xy[1], H)
        px_x, px_y = field_to_minimap_px(field_x, field_y)
        
        if mask is not None:
            is_inlier = bool(mask[used_counter][0])
            color = (0, 255, 0) if is_inlier else (0, 0, 255)
            used_counter += 1
        else:
            color = (255, 128, 0)
            
        # Draw projected point with a double-ring highlight
        cv2.circle(minimap_img, (px_x, px_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(minimap_img, (px_x, px_y), 5, color, -1, cv2.LINE_AA)
        
        cv2.putText(
            minimap_img,
            label,
            (px_x + 10, px_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
    return minimap_img

def draw_player_projections_on_minimap(
    minimap_img: np.ndarray,
    player_result,
    H: np.ndarray,
    conf_threshold: float = 0.3,
    dims: FieldDimensions = FieldDimensions(),
    scale: float = 8.0,
    padding_meters: float = 5.0,
    player_color: tuple = (255, 0, 255)
) -> np.ndarray:
    """
    Projects detected player bottom-center coordinates onto the 2D minimap
    using persistent track IDs if available.
    """
    annotated_minimap = minimap_img.copy()
    if player_result.boxes is None or len(player_result.boxes) == 0:
        return annotated_minimap

    pad_px = int(padding_meters * scale)

    def field_to_minimap_px(x_m: float, y_m: float) -> tuple[int, int]:
        px_x = int(x_m * scale) + pad_px
        px_y = int(y_m * scale) + pad_px
        return (px_x, px_y)

    boxes = player_result.boxes.xyxy.cpu().numpy()
    confs = player_result.boxes.conf.cpu().numpy()
    track_ids = (
        player_result.boxes.id.cpu().numpy().astype(int)
        if player_result.boxes.id is not None
        else None
    )

    for idx, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < conf_threshold:
            continue

        player_id = track_ids[idx] if track_ids is not None else idx

        x1, y1, x2, y2 = box
        x_center = float((x1 + x2) / 2.0)
        y_bottom = float(y2)

        field_x, field_y = pixel_to_field(x_center, y_bottom, H)

        if not (-5.0 <= field_x <= dims.length + 5.0 and -5.0 <= field_y <= dims.width + 5.0):
            continue

        px_x, px_y = field_to_minimap_px(field_x, field_y)

        # Draw player marker on minimap
        cv2.circle(annotated_minimap, (px_x, px_y), 8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(annotated_minimap, (px_x, px_y), 5, player_color, -1, cv2.LINE_AA)

        label_text = f"P{player_id}"
        cv2.putText(
            annotated_minimap,
            label_text,
            (px_x + 8, px_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated_minimap


def draw_ball_projection_on_minimap(
    minimap_img: np.ndarray,
    ball_detection: Optional[dict] = None,
    scale: float = 8.0,
    padding_meters: float = 5.0,
    ball_color: tuple = (0, 255, 255)
) -> np.ndarray:
    """
    Proyecta la ubicación 2D del balón en el minimapa de la cancha.
    """
    annotated_minimap = minimap_img.copy()
    if ball_detection is None:
        return annotated_minimap

    pitch_x = ball_detection.get("pitch_x_m")
    pitch_y = ball_detection.get("pitch_y_m")

    if pitch_x is None or pitch_y is None:
        return annotated_minimap

    pad_px = int(padding_meters * scale)
    px_x = int(pitch_x * scale) + pad_px
    px_y = int(pitch_y * scale) + pad_px

    # Dibujar marcador del balón en el minimapa
    cv2.circle(annotated_minimap, (px_x, px_y), 7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(annotated_minimap, (px_x, px_y), 5, ball_color, -1, cv2.LINE_AA)
    cv2.circle(annotated_minimap, (px_x, px_y), 2, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(
        annotated_minimap,
        "BALL",
        (px_x + 7, px_y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    return annotated_minimap



