import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# Configurar logger de depuración en archivo
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_LOG_PATH = LOG_DIR / "event_detector_debug.log"

logger = logging.getLogger("SoccerNetEventDetector")
logger.setLevel(logging.DEBUG)

# File Handler para el log detallado del detector
file_handler = logging.FileHandler(str(DEBUG_LOG_PATH), mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(file_handler)

PRIORITY_EVENT_CLASSES = [
    "Pase"
]

class TemporalSpottingHead(nn.Module):
    """
    Cabeza Temporal para Action Spotting sobre secuencias de características visuales SoccerNet.
    Procesa un buffer de T fotogramas usando una red GRU Bidireccional.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, num_classes: int = len(PRIORITY_EVENT_CLASSES)):
        super(TemporalSpottingHead, self).__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        probs = self.fc(last_out)
        return probs

class SoccerNetEventDetector:
    def __init__(
        self,
        device: Optional[str] = None,
        confidence_threshold: float = 0.50,
        buffer_size: int = 30,
        nms_window_sec: float = 2.0
    ):
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        self.nms_window_sec = nms_window_sec

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"=== INICIALIZANDO SOCCERNET EVENT DETECTOR (SOLO PASES) ===")
        logger.info(f"Dispositivo PyTorch: {self.device}")
        logger.info(f"Umbral de confianza Pase: {self.confidence_threshold}")
        logger.info(f"Archivo de log de depuración: {DEBUG_LOG_PATH}")

        # 1. Extractor de características PyTorch (ResNet-18 Backbone SoccerNet)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.feature_extractor.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Modelo Temporal de Action Spotting
        self.temporal_model = TemporalSpottingHead(
            feature_dim=512,
            hidden_dim=256,
            num_classes=len(PRIORITY_EVENT_CLASSES)
        ).to(self.device)
        self.temporal_model.eval()

        self.feature_buffer: List[torch.Tensor] = []
        self.frame_metadata_buffer: List[Dict] = []
        self.recent_events_history: List[Dict] = []

    def extract_frame_features(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(tensor_img)
            feat = feat.flatten(1)
        return feat

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        ball_pos: Optional[Tuple[float, float]] = None,
        pass_from_player_id: Optional[int] = None,
        pass_to_player_id: Optional[int] = None
    ) -> List[Dict]:
        feat = self.extract_frame_features(frame_bgr)
        self.feature_buffer.append(feat)
        self.frame_metadata_buffer.append({
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "ball_pos": ball_pos,
            "pass_from_player_id": pass_from_player_id,
            "pass_to_player_id": pass_to_player_id
        })

        if len(self.feature_buffer) > self.buffer_size:
            self.feature_buffer.pop(0)
            self.frame_metadata_buffer.pop(0)

        current_meta = self.frame_metadata_buffer[-1]

        # Log detallado de telemetría de fotograma
        logger.debug(
            f"[FRAME {frame_idx:04d} | {timestamp_sec:.2f}s] "
            f"Buffer: {len(self.feature_buffer)} | Ball: {ball_pos} | "
            f"Passer: #{pass_from_player_id} -> Receiver: #{pass_to_player_id}"
        )

        detected_events = []

        # FASE ACTUAL: Únicamente detectar PASES
        # HÍBRIDO 1: Detección por transición de posesión del balón entre jugadores
        if (pass_from_player_id is not None and pass_to_player_id is not None and 
            pass_from_player_id != pass_to_player_id):
            
            if not self._is_duplicate_event("Pase", current_meta["timestamp_sec"]):
                evt_pass = {
                    "event_type": "Pase",
                    "confidence": 0.88,
                    "timestamp_sec": round(current_meta["timestamp_sec"], 2),
                    "frame_idx": current_meta["frame_idx"],
                    "spatial_location": list(current_meta["ball_pos"]) if current_meta["ball_pos"] else None,
                    "pass_from_player_id": pass_from_player_id,
                    "pass_to_player_id": pass_to_player_id
                }
                detected_events.append(evt_pass)
                self.recent_events_history.append(evt_pass)
                logger.info(
                    f"🎯 [PASE DETECTADO POR POSESIÓN @ Frame {frame_idx}] "
                    f"Jugador #{pass_from_player_id} ➔ Jugador #{pass_to_player_id}"
                )

        # HÍBRIDO 2: Inferencia visual con modelo temporal SoccerNet PyTorch (Solo clase Pase)
        if len(self.feature_buffer) >= 5:
            seq_tensor = torch.cat(self.feature_buffer, dim=0).unsqueeze(0)
            with torch.no_grad():
                probs = self.temporal_model(seq_tensor)[0]

            prob_pase = float(probs[0].item())
            logger.debug(f"[FRAME {frame_idx:04d}] Probabilidad Pase SoccerNet: {prob_pase:.4f}")

            effective_threshold = 0.35 if (pass_from_player_id is not None or pass_to_player_id is not None) else self.confidence_threshold

            if prob_pase >= effective_threshold:
                if not self._is_duplicate_event("Pase", current_meta["timestamp_sec"]):
                    evt = {
                        "event_type": "Pase",
                        "confidence": round(prob_pase, 4),
                        "timestamp_sec": round(current_meta["timestamp_sec"], 2),
                        "frame_idx": current_meta["frame_idx"],
                        "spatial_location": list(current_meta["ball_pos"]) if current_meta["ball_pos"] else None,
                        "pass_from_player_id": current_meta["pass_from_player_id"],
                        "pass_to_player_id": current_meta["pass_to_player_id"]
                    }
                    detected_events.append(evt)
                    self.recent_events_history.append(evt)
                    logger.info(f"🔥 [PASE DETECTADO VISUAL @ Frame {frame_idx}] Conf: {prob_pase:.4f}")

        return detected_events


    def _is_duplicate_event(self, event_type: str, current_ts: float) -> bool:
        for prev_evt in reversed(self.recent_events_history):
            if prev_evt["event_type"] == event_type:
                if (current_ts - prev_evt["timestamp_sec"]) < self.nms_window_sec:
                    return True
        return False
