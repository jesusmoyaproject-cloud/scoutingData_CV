from typing import List, Tuple, Optional, Dict
import numpy as np

class BallTracker:
    def __init__(self, max_history: int = 10, max_missing_frames: int = 5, max_distance_jump: float = 200.0):
        self.max_history = max_history
        self.max_missing_frames = max_missing_frames
        self.max_distance_jump = max_distance_jump
        self.history: List[Optional[Tuple[float, float]]] = []
        self.missing_count = 0
        self.last_known_detection: Optional[Dict] = None

    def update(self, current_detection: Optional[Dict]) -> Tuple[Optional[Dict], List[Tuple[float, float]]]:
        if current_detection is not None:
            curr_x = current_detection["pixel_x"]
            curr_y = current_detection["pixel_y"]

            if self.last_known_detection is not None and self.missing_count == 0:
                prev_x = self.last_known_detection["pixel_x"]
                prev_y = self.last_known_detection["pixel_y"]
                dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                if dist > self.max_distance_jump:
                    current_detection["confidence"] *= 0.7

            self.missing_count = 0
            self.last_known_detection = current_detection.copy()
            self.history.append((curr_x, curr_y))
            if len(self.history) > self.max_history:
                self.history.pop(0)

            return current_detection, [p for p in self.history if p is not None]

        self.missing_count += 1
        if self.missing_count <= self.max_missing_frames and self.last_known_detection is not None and len(self.history) >= 2:
            valid_pts = [p for p in self.history if p is not None]
            if len(valid_pts) >= 2:
                p2 = valid_pts[-1]
                p1 = valid_pts[-2]
                vx = p2[0] - p1[0]
                vy = p2[1] - p1[1]

                est_x = p2[0] + vx * 0.5
                est_y = p2[1] + vy * 0.5

                interpolated_det = self.last_known_detection.copy()
                interpolated_det["pixel_x"] = round(est_x, 2)
                interpolated_det["pixel_y"] = round(est_y, 2)
                interpolated_det["confidence"] = round(max(0.1, interpolated_det["confidence"] * 0.85), 2)
                interpolated_det["interpolated"] = True

                self.history.append((est_x, est_y))
                if len(self.history) > self.max_history:
                    self.history.pop(0)

                return interpolated_det, [p for p in self.history if p is not None]

        return None, [p for p in self.history if p is not None]
