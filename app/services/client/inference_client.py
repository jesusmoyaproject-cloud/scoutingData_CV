import io
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import httpx

logger = logging.getLogger("AsyncInferenceClient")

class MockArrayWrapper:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, item):
        res = self._arr[item]
        if isinstance(res, np.ndarray):
            return MockArrayWrapper(res)
        return res

    def __len__(self):
        return len(self._arr)

    def __iter__(self):
        return iter(self._arr)

    @property
    def shape(self):
        return self._arr.shape

class MockKeypoints:
    def __init__(self, keypoints_list: List[Dict]):
        if keypoints_list:
            xy_arr = np.array([[kp["x"], kp["y"]] for kp in keypoints_list], dtype=np.float32)
            conf_arr = np.array([kp["conf"] for kp in keypoints_list], dtype=np.float32)
            self.xy = [MockArrayWrapper(xy_arr)]
            self.conf = [MockArrayWrapper(conf_arr)]
        else:
            self.xy = [MockArrayWrapper(np.empty((0, 2), dtype=np.float32))]
            self.conf = [MockArrayWrapper(np.empty((0,), dtype=np.float32))]

class MockKeypointResult:
    def __init__(self, keypoints_list: List[Dict]):
        self.keypoints = MockKeypoints(keypoints_list)

class MockBoxes:
    def __init__(self, detections: List[Dict]):
        if detections:
            boxes_list = [d["box"] for d in detections]
            confs_list = [d["confidence"] for d in detections]
            cls_list = [d["class_id"] for d in detections]
            t_ids = [d["track_id"] for d in detections]

            self.xyxy = MockArrayWrapper(np.array(boxes_list, dtype=np.float32))
            self.conf = MockArrayWrapper(np.array(confs_list, dtype=np.float32))
            self.cls = MockArrayWrapper(np.array(cls_list, dtype=np.float32))

            if any(tid is not None for tid in t_ids):
                id_arr = np.array([tid if tid is not None else -1 for tid in t_ids], dtype=np.int32)
                self.id = MockArrayWrapper(id_arr)
            else:
                self.id = None
        else:
            self.xyxy = MockArrayWrapper(np.empty((0, 4), dtype=np.float32))
            self.conf = MockArrayWrapper(np.empty((0,), dtype=np.float32))
            self.cls = MockArrayWrapper(np.empty((0,), dtype=np.float32))
            self.id = None

    def __len__(self):
        return len(self.xyxy)

class MockPlayerResult:
    def __init__(self, detections: List[Dict]):
        self.boxes = MockBoxes(detections)

class AsyncInferenceClient:
    def __init__(
        self,
        keypoint_url: str = "http://localhost:8001",
        player_url: str = "http://localhost:8002",
        ball_url: str = "http://localhost:8003",
        event_url: str = "http://localhost:8004",
        timeout: float = 30.0
    ):
        self.keypoint_url = keypoint_url.rstrip("/")
        self.player_url = player_url.rstrip("/")
        self.ball_url = ball_url.rstrip("/")
        self.event_url = event_url.rstrip("/")
        self.timeout = timeout
        
        # Persistent Connection Pool for high performance HTTP Keep-Alive
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        self._http_client = httpx.AsyncClient(limits=limits, timeout=self.timeout)

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def predict_keypoints(self, img_bytes: bytes, conf: float = 0.25) -> MockKeypointResult:
        files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
        params = {"conf": conf}
        resp = await self._http_client.post(f"{self.keypoint_url}/predict", files=files, params=params)
        resp.raise_for_status()
        data = resp.json()
        return MockKeypointResult(data.get("keypoints", []))

    async def track_players(self, img_bytes: bytes, conf: float = 0.35, is_video: bool = True) -> MockPlayerResult:
        files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
        params = {"conf": conf, "is_video": is_video}
        resp = await self._http_client.post(f"{self.player_url}/track", files=files, params=params)
        resp.raise_for_status()
        data = resp.json()
        return MockPlayerResult(data.get("detections", []))

    async def predict_ball(self, img_bytes: bytes, conf: float = 0.15) -> Optional[Dict]:
        files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
        params = {"conf": conf}
        resp = await self._http_client.post(f"{self.ball_url}/predict", files=files, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("ball_detection")

    async def predict_events(
        self,
        img_bytes: bytes,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
        conf: float = 0.45,
        ball_x: Optional[float] = None,
        ball_y: Optional[float] = None,
        pass_from_player_id: Optional[int] = None,
        pass_to_player_id: Optional[int] = None
    ) -> List[Dict]:
        files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
        params = {
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "conf": conf
        }
        if ball_x is not None and ball_y is not None:
            params["ball_x"] = ball_x
            params["ball_y"] = ball_y
        if pass_from_player_id is not None:
            params["pass_from_player_id"] = pass_from_player_id
        if pass_to_player_id is not None:
            params["pass_to_player_id"] = pass_to_player_id

        resp = await self._http_client.post(f"{self.event_url}/predict", files=files, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("events", [])

    async def process_frame_parallel(
        self,
        frame_bgr: np.ndarray,
        conf_keypoint: float = 0.25,
        conf_player: float = 0.35,
        conf_ball: float = 0.15,
        conf_event: float = 0.45,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
        pass_from_player_id: Optional[int] = None,
        pass_to_player_id: Optional[int] = None,
        is_video: bool = True
    ) -> Tuple[MockKeypointResult, MockPlayerResult, Optional[Dict], List[Dict]]:
        img_bytes = await self.encode_image_async(frame_bgr)
        kp_task = self.predict_keypoints(img_bytes, conf=conf_keypoint)
        player_task = self.track_players(img_bytes, conf=conf_player, is_video=is_video)
        ball_task = self.predict_ball(img_bytes, conf=conf_ball)
        event_task = self.predict_events(
            img_bytes,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            conf=conf_event,
            pass_from_player_id=pass_from_player_id,
            pass_to_player_id=pass_to_player_id
        )

        kp_res, player_res, ball_res, event_res = await asyncio.gather(
            kp_task, player_task, ball_task, event_task
        )
        return kp_res, player_res, ball_res, event_res


    async def encode_image_async(self, frame_bgr: np.ndarray) -> bytes:
        return await asyncio.to_thread(self._encode_image_sync, frame_bgr)

    def _encode_image_sync(self, frame_bgr: np.ndarray) -> bytes:
        _, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return buffer.tobytes()

