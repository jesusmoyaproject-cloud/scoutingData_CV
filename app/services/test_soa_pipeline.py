import os
import sys
import time
import asyncio
import logging
from pathlib import Path
import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.client.inference_client import AsyncInferenceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestSOAPipeline")

async def test_microservices():
    client = AsyncInferenceClient(
        keypoint_url="http://localhost:8001",
        player_url="http://localhost:8002",
        ball_url="http://localhost:8003",
        event_url="http://localhost:8004"
    )

    img_path = APP_DIR / "img" / "frame6.png"
    if not img_path.exists():
        logger.warning(f"Test image not found at {img_path}, generating synthetic frame for testing.")
        frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(frame_bgr, (100, 100), (300, 400), (0, 255, 0), -1)
    else:
        frame_bgr = cv2.imread(str(img_path))
        logger.info(f"Loaded test frame: {img_path} ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})")

    logger.info("Executing PARALLEL inference request across 4 microservices (8001, 8002, 8003, 8004)...")
    start_t = time.time()

    kp_res, player_res, ball_res, events_res = await client.process_frame_parallel(
        frame_bgr,
        frame_idx=1,
        timestamp_sec=3.5
    )
    total_ms = (time.time() - start_t) * 1000

    kps_count = len(kp_res.keypoints.xy[0]) if kp_res.keypoints and len(kp_res.keypoints.xy) > 0 else 0
    players_count = len(player_res.boxes.xyxy) if player_res.boxes else 0
    ball_found = ball_res is not None

    logger.info("================ SOA Test Results (4 Services) ================")
    logger.info(f"Total Parallel Response Time: {total_ms:.2f} ms")
    logger.info(f"Keypoints Detected: {kps_count}")
    logger.info(f"Players Detected: {players_count}")
    logger.info(f"Ball Detected: {ball_found} -> {ball_res}")
    logger.info(f"Events Detected: {len(events_res)} -> {events_res}")
    logger.info("===============================================================")

if __name__ == "__main__":
    asyncio.run(test_microservices())
