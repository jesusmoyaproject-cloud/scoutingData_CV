from typing import Dict, Tuple
from config.field_dimensions import FieldDimensions

def get_field_coords(dims: FieldDimensions = FieldDimensions()) -> Dict[str, Tuple[float, float]]:
    """
    Generates the physical coordinates (in meters) for all 30 standard pitch keypoints.
    The origin (0, 0) is the top-left corner of the field boundary.
    x increases to the right (length), y increases downwards (width).
    """
    
    half_len = dims.length / 2.0
    half_wid = dims.width / 2.0
    
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

        # 10-13: Penalty box left corners
        "10": (dims.penalty_box_length, half_wid - dims.penalty_box_width / 2.0),
        "11": (dims.penalty_box_length, half_wid - dims.goal_box_width / 2.0),
        "12": (dims.penalty_box_length, half_wid + dims.goal_box_width / 2.0),
        "13": (dims.penalty_box_length, half_wid + dims.penalty_box_width / 2.0),

        # 15-18: Center line (x = length/2)
        "15": (half_len, 0.0),
        "16": (half_len, half_wid - dims.centre_circle_radius),
        "17": (half_len, half_wid + dims.centre_circle_radius),
        "18": (half_len, dims.width),

        # 20-23: Penalty box right corners
        "20": (dims.length - dims.penalty_box_length, half_wid - dims.penalty_box_width / 2.0),
        "21": (dims.length - dims.penalty_box_length, half_wid - dims.goal_box_width / 2.0),
        "22": (dims.length - dims.penalty_box_length, half_wid + dims.goal_box_width / 2.0),
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
