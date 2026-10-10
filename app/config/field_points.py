from typing import Dict, Tuple
import numpy as np
from config.field_dimensions import FieldDimensions

def get_field_coords(dims: FieldDimensions = FieldDimensions()) -> Dict[str, Tuple[float, float]]:
    """
    Generates the physical coordinates (in meters) for all 30 standard pitch keypoints.
    The origin (0, 0) is the top-left corner of the field boundary.
    x increases to the right (length), y increases downwards (width).
    """
    
    half_len = dims.length / 2.0
    half_wid = dims.width / 2.0
    
    # Distancia vertical del arco de penalti (R=9.15m desde punto de penal x=11.0m)
    # En la línea del área grande (x=16.5m), dx = 16.5 - 11.0 = 5.5m:
    # dy = sqrt(9.15^2 - 5.5^2) = sqrt(53.4725) ≈ 7.3125m
    dx_penalty = dims.penalty_box_length - dims.penalty_spot_distance
    dy_arc = float(np.sqrt(max(0.0, dims.centre_circle_radius**2 - dx_penalty**2)))

    # Vertices definitions based on standard geometry and configuration
    coords = {
        # 1-6: Left boundary line (x=0)
        "01": (0.0, 0.0),
        "02": (0.0, half_wid - dims.penalty_box_width / 2.0),
        "03": (0.0, half_wid - dims.goal_box_width / 2.0),
        "04": (0.0, half_wid + dims.goal_box_width / 2.0),
        "05": (0.0, half_wid + dims.penalty_box_width / 2.0),
        "06": (0.0, dims.width),

        # 7-8: Goal box left corners
        "07": (dims.goal_box_length, half_wid - dims.goal_box_width / 2.0),
        "08": (dims.goal_box_length, half_wid + dims.goal_box_width / 2.0),

        # 10-13: Penalty box left corners and penalty arc (medialuna) intersections
        "10": (dims.penalty_box_length, half_wid - dims.penalty_box_width / 2.0),
        "11": (dims.penalty_box_length, half_wid - dy_arc),
        "12": (dims.penalty_box_length, half_wid + dy_arc),
        "13": (dims.penalty_box_length, half_wid + dims.penalty_box_width / 2.0),

        # 15-18: Center line (x = length/2)
        "15": (half_len, 0.0),
        "16": (half_len, half_wid - dims.centre_circle_radius),
        "17": (half_len, half_wid + dims.centre_circle_radius),
        "18": (half_len, dims.width),

        # 20-23: Penalty box right corners and penalty arc (medialuna) intersections
        "20": (dims.length - dims.penalty_box_length, half_wid - dims.penalty_box_width / 2.0),
        "21": (dims.length - dims.penalty_box_length, half_wid - dy_arc),
        "22": (dims.length - dims.penalty_box_length, half_wid + dy_arc),
        "23": (dims.length - dims.penalty_box_length, half_wid + dims.penalty_box_width / 2.0),

        # 25-26: Goal box right corners
        "25": (dims.length - dims.goal_box_length, half_wid - dims.goal_box_width / 2.0),
        "26": (dims.length - dims.goal_box_length, half_wid + dims.goal_box_width / 2.0),

        # 27-32: Right boundary line (x=length)
        "27": (dims.length, 0.0),
        "28": (dims.length, half_wid - dims.penalty_box_width / 2.0),
        "29": (dims.length, half_wid - dims.goal_box_width / 2.0),
        "30": (dims.length, half_wid + dims.goal_box_width / 2.0),
        "31": (dims.length, half_wid + dims.penalty_box_width / 2.0),
        "32": (dims.length, dims.width),

        # 14 & 19: Left and right intersections of center circle with center line
        "14": (half_len - dims.centre_circle_radius, half_wid),
        "19": (half_len + dims.centre_circle_radius, half_wid),
    }
    
    return coords

FIELD_COORDS = get_field_coords()
