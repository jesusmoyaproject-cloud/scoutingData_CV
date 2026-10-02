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

DEFAULT_MODEL_PATH = settings.MODEL_KEYPOINT_PATH

logger = logging.getLogger('KeypointService')
logging.basicConfig(level=logging.INFO)

model = None
model_path_str = str(DEFAULT_MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_path_str
    model_env = os.getenv('MODEL_PATH', str(DEFAULT_MODEL_PATH))
    logger.info(f'Loading Keypoint Model (OpenVINO): {model_env}')
    if os.path.exists(model_env):
        model = YOLO(model_env, task='pose')
        logger.info('Keypoint Model loaded successfully.')
    else:
        logger.error(f'Model path not found: {model_env}')
    yield

app = FastAPI(title='Keypoint Detection Microservice', version='4.0', lifespan=lifespan)

@app.get('/health')
async def health():
    return {
        'status': 'healthy' if model is not None else 'degraded',
        'service': 'keypoint_service',
        'model_path': model_path_str
    }

@app.post('/predict')
@app.post('/track')
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0)
):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise HTTPException(status_code=400, detail='Invalid image file')

        results = model.predict(source=frame_bgr, conf=conf, save=False, verbose=False)
        result = results[0]

        keypoints_list = []
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kps_xy = result.keypoints.xy[0].cpu().numpy()
            kps_conf = result.keypoints.conf[0].cpu().numpy()

            for idx, (xy, c) in enumerate(zip(kps_xy, kps_conf)):
                keypoints_list.append({
                    'id': int(idx),
                    'x': float(xy[0]),
                    'y': float(xy[1]),
                    'conf': float(c)
                })

        return {'keypoints': keypoints_list}
    except Exception as e:
        logger.error(f'Error processing keypoints: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8001)
