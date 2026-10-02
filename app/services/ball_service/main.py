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

DEFAULT_MODEL_PATH = settings.MODEL_BALL_PATH

logger = logging.getLogger('BallService')
logging.basicConfig(level=logging.INFO)

model = None
model_path_str = str(DEFAULT_MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_path_str
    model_env = os.getenv('MODEL_PATH', str(DEFAULT_MODEL_PATH))
    logger.info(f'Loading Ball Model (OpenVINO): {model_env}')
    if os.path.exists(model_env):
        model = YOLO(model_env, task='detect')
        logger.info('Ball Model loaded successfully.')
    else:
        logger.error(f'Model path not found: {model_env}')
    yield

app = FastAPI(title='Ball Detection Microservice', version='4.0', lifespan=lifespan)

@app.get('/health')
async def health():
    return {
        'status': 'healthy' if model is not None else 'degraded',
        'service': 'ball_service',
        'model_path': model_path_str
    }

@app.post('/predict')
@app.post('/track')
async def predict_ball(
    file: UploadFile = File(...),
    conf: float = Query(0.15, ge=0.0, le=1.0)
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

        ball_det = None
        if result.boxes is not None and len(result.boxes) > 0:
            b_boxes = result.boxes.xyxy.cpu().numpy()
            b_confs = result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(b_confs))

            if float(b_confs[best_idx]) >= conf:
                bx1, by1, bx2, by2 = b_boxes[best_idx]
                bcx = float((bx1 + bx2) / 2.0)
                bcy = float((by1 + by2) / 2.0)
                ball_det = {
                    'pixel_x': round(bcx, 2),
                    'pixel_y': round(bcy, 2),
                    'confidence': float(b_confs[best_idx]),
                    'interpolated': False,
                    'bbox': [float(bx1), float(by1), float(bx2), float(by2)]
                }

        return {'ball_detection': ball_det}
    except Exception as e:
        logger.error(f'Error processing ball detection: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8003)
