from typing import List, Optional
from pydantic import BaseModel

class KeypointItem(BaseModel):
    id: int
    x: float
    y: float
    conf: float

class KeypointResponse(BaseModel):
    keypoints: List[KeypointItem]

class DetectionBox(BaseModel):
    box: List[float]
    confidence: float
    class_id: int
    track_id: Optional[int] = None

class PlayerResponse(BaseModel):
    detections: List[DetectionBox]

class BallDetection(BaseModel):
    pixel_x: float
    pixel_y: float
    confidence: float
    interpolated: bool = False
    bbox: Optional[List[float]] = None

class BallResponse(BaseModel):
    ball_detection: Optional[BallDetection] = None

class EventItem(BaseModel):
    event_type: str
    confidence: float
    timestamp_sec: float
    frame_idx: int
    spatial_location: Optional[List[float]] = None
    pass_from_player_id: Optional[int] = None
    pass_to_player_id: Optional[int] = None

class EventResponse(BaseModel):
    events: List[EventItem]
    buffer_size: int = 0


class HealthResponse(BaseModel):
    status: str
    service: str
    model_path: str

