import os
import sys
import io
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO

SERVICE_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICE_DIR.parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings

DEFAULT_MODEL_PATH = settings.MODEL_PLAYER_PATH

logger = logging.getLogger('PlayerService')
logging.basicConfig(level=logging.INFO)

model = None
model_path_str = str(DEFAULT_MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_path_str
    model_env = os.getenv('MODEL_PATH', str(DEFAULT_MODEL_PATH))
    logger.info(f'Loading Player Model (OpenVINO): {model_env}')
    if os.path.exists(model_env):
        model = YOLO(model_env, task='detect')
        logger.info('Player Model loaded successfully.')
    else:
        logger.error(f'Model path not found: {model_env}')
    yield

app = FastAPI(title='Player Tracking Microservice', version='4.0', lifespan=lifespan)

@app.get('/health')
async def health():
    return {
        'status': 'healthy' if model is not None else 'degraded',
        'service': 'player_service',
        'model_path': model_path_str
    }

@app.post('/track')
@app.post('/predict')
async def track_players(
    file: UploadFile = File(...),
    conf: float = Query(0.35, ge=0.0, le=1.0),
    is_video: bool = Query(True)
):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise HTTPException(status_code=400, detail='Invalid image file')

        if is_video:
            try:
                results = model.track(
                    source=frame_bgr,
                    conf=conf,
                    persist=True,
                    tracker='bytetrack.yaml',
                    verbose=False
                )
            except Exception:
                results = model.predict(source=frame_bgr, conf=conf, save=False, verbose=False)
        else:
            results = model.predict(source=frame_bgr, conf=conf, save=False, verbose=False)

        result = results[0]
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else None
            )

            for idx, (box, c, cls_id) in enumerate(zip(boxes, confs, classes)):
                t_id = int(track_ids[idx]) if track_ids is not None else None
                detections.append({
                    'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    'confidence': float(c),
                    'class_id': int(cls_id),
                    'track_id': t_id
                })

        return {'detections': detections}
    except Exception as e:
        logger.error(f'Error processing player tracking: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8002)
