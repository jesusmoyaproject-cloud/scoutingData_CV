import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
import cv2
import numpy as np

# Registrar ruta app en sys.path
SERVICE_DIR = Path(__file__).resolve().parent
SERVICES_DIR = SERVICE_DIR.parent
APP_DIR = SERVICES_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.schemas import EventResponse, EventItem, HealthResponse
from services.event_service.event_detector import SoccerNetEventDetector, PRIORITY_EVENT_CLASSES

logger = logging.getLogger("EventService")
logging.basicConfig(level=logging.INFO)

detector: Optional[SoccerNetEventDetector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("Cargando Servicio de Detección de Eventos (PyTorch / SoccerNet)...")
    try:
        detector = SoccerNetEventDetector(confidence_threshold=0.45)
        logger.info("Modelo SoccerNet inicializado correctamente.")
    except Exception as e:
        logger.error(f"Error cargando modelo SoccerNet: {e}")
    yield

app = FastAPI(
    title="SoccerNet Event Detection Microservice",
    description="Microservicio SOA para la detección temporal de eventos futbolísticos (Pases, Remates, Gol, Atajadas, Faltas, Tarjetas, Fuera de Juego).",
    version="2.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy" if detector is not None else "degraded",
        "service": "event_service",
        "model_path": "SoccerNet_PyTorch_ResNet18_Temporal"
    }

@app.get("/supported_events")
async def supported_events():
    return {
        "priority_events": PRIORITY_EVENT_CLASSES,
        "total": len(PRIORITY_EVENT_CLASSES)
    }

@app.post("/predict", response_model=EventResponse)
@app.post("/predict_frame", response_model=EventResponse)
async def predict_frame(
    file: UploadFile = File(...),
    frame_idx: int = Query(0, ge=0),
    timestamp_sec: float = Query(0.0, ge=0.0),
    conf: float = Query(0.45, ge=0.0, le=1.0),
    ball_x: Optional[float] = Query(None),
    ball_y: Optional[float] = Query(None),
    pass_from_player_id: Optional[int] = Query(None),
    pass_to_player_id: Optional[int] = Query(None)
):
    if detector is None:
        raise HTTPException(status_code=503, detail="Servicio de modelos SoccerNet no cargado")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="Formato de imagen inválido")

        detector.confidence_threshold = conf
        ball_pos = (ball_x, ball_y) if (ball_x is not None and ball_y is not None) else None

        raw_events = detector.process_frame(
            frame_bgr=frame_bgr,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            ball_pos=ball_pos,
            pass_from_player_id=pass_from_player_id,
            pass_to_player_id=pass_to_player_id
        )

        event_items = [EventItem(**evt) for evt in raw_events]
        return EventResponse(
            events=event_items,
            buffer_size=len(detector.feature_buffer)
        )


    except Exception as e:
        logger.error(f"Error procesando fotograma en event_service: {e}")
        raise HTTPException(status_code=500, detail=str(e))
