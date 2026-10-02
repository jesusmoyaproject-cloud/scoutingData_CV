import cv2
import numpy as np
from typing import Tuple
from config.field_dimensions import FieldDimensions

def draw_pitch_template(
    dims: FieldDimensions = FieldDimensions(),
    scale: float = 10.0,
    padding_meters: float = 5.0,
    background_color: tuple = (34, 112, 34),
    line_color: tuple = (255, 255, 255),
    line_thickness_px: int = 2
) -> Tuple[np.ndarray, float, float]:
    """
    Renders a standardized 2D top-down canvas of a football pitch.
    """
    pad_px = int(padding_meters * scale)
    pitch_w_px = int(dims.length * scale)
    pitch_h_px = int(dims.width * scale)
    
    img_w = pitch_w_px + 2 * pad_px
    img_h = pitch_h_px + 2 * pad_px
    
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img[:] = background_color
    
    def to_px(x_m: float, y_m: float) -> Tuple[int, int]:
        return (int(x_m * scale) + pad_px, int(y_m * scale) + pad_px)
    
    tl = to_px(0, 0)
    br = to_px(dims.length, dims.width)
    
    cv2.rectangle(img, tl, br, line_color, line_thickness_px)
    
    mid_top = to_px(dims.length / 2.0, 0)
    mid_bottom = to_px(dims.length / 2.0, dims.width)
    cv2.line(img, mid_top, mid_bottom, line_color, line_thickness_px)
    
    center = to_px(dims.length / 2.0, dims.width / 2.0)
    cv2.circle(img, center, int(dims.centre_circle_radius * scale), line_color, line_thickness_px)
    cv2.circle(img, center, int(0.5 * scale), line_color, -1)
    
    # Left goal & penalty areas
    cv2.rectangle(img, to_px(0.0, dims.width / 2.0 - dims.goal_box_width / 2.0), to_px(dims.goal_box_length, dims.width / 2.0 + dims.goal_box_width / 2.0), line_color, line_thickness_px)
    cv2.rectangle(img, to_px(0.0, dims.width / 2.0 - dims.penalty_box_width / 2.0), to_px(dims.penalty_box_length, dims.width / 2.0 + dims.penalty_box_width / 2.0), line_color, line_thickness_px)
    l_penalty_spot = to_px(dims.penalty_spot_distance, dims.width / 2.0)
    cv2.circle(img, l_penalty_spot, int(0.3 * scale), line_color, -1)
    cv2.ellipse(img, l_penalty_spot, (int(dims.centre_circle_radius * scale), int(dims.centre_circle_radius * scale)), 0.0, -53.0, 53.0, line_color, line_thickness_px)
    
    # Right goal & penalty areas
    cv2.rectangle(img, to_px(dims.length - dims.goal_box_length, dims.width / 2.0 - dims.goal_box_width / 2.0), to_px(dims.length, dims.width / 2.0 + dims.goal_box_width / 2.0), line_color, line_thickness_px)
    cv2.rectangle(img, to_px(dims.length - dims.penalty_box_length, dims.width / 2.0 - dims.penalty_box_width / 2.0), to_px(dims.length, dims.width / 2.0 + dims.penalty_box_width / 2.0), line_color, line_thickness_px)
    r_penalty_spot = to_px(dims.length - dims.penalty_spot_distance, dims.width / 2.0)
    cv2.circle(img, r_penalty_spot, int(0.3 * scale), line_color, -1)
    cv2.ellipse(img, r_penalty_spot, (int(dims.centre_circle_radius * scale), int(dims.centre_circle_radius * scale)), 0.0, 127.0, 233.0, line_color, line_thickness_px)
    
    return img, float(pad_px), scale
