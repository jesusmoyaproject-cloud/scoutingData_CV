"""
ScoutingData v5.0 - Motor de Inferencia Directa en Proceso

Carga los 3 modelos (Keypoint, Player, Ball) directamente en memoria, eliminando
el overhead HTTP/SOA (JPEG encode × 3, HTTP POST × 3, JSON serialize × 3).

Selección automática de modelo y device por entorno:
  LOCAL  → OpenVINO IR  + CPU Intel  (2-4x vs .pt CPU)
  KAGGLE → PyTorch .pt  + CUDA T4    (6-10x vs OpenVINO CPU)
"""

import logging
from typing import Tuple, Optional
from pathlib import Path
import torch
from ultralytics import YOLO
from config import settings

logger = logging.getLogger("DirectInferenceEngine")


class DirectInferenceEngine:
    """
    Motor de inferencia sin capa HTTP.
    Expone un único método `infer()` que retorna los mismos tipos
    que usaban los microservicios SOA (compatible con el pipeline existente).
    """

    def __init__(self):
        self.device = settings.INFERENCE_DEVICE
        self.format = settings.MODEL_FORMAT

        logger.info(
            f"[Engine] Inicializando DirectInferenceEngine — "
            f"format={self.format}, device={self.device}, "
            f"CUDA={settings.CUDA_AVAILABLE}"
        )

        if self.format == "pt":
            self._load_pt_models()
        else:
            self._load_openvino_models()

        logger.info("[Engine] ✅ Los 3 modelos están cargados en memoria.")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_pt_models(self):
        """Carga modelos .pt originales para inferencia CUDA (Kaggle T4)."""
        kp_path  = settings.MODEL_KEYPOINT_PT_PATH
        pl_path  = settings.MODEL_PLAYER_PT_PATH
        ba_path  = settings.MODEL_BALL_PT_PATH

        for label, path in [("Keypoint", kp_path), ("Player", pl_path), ("Ball", ba_path)]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Modelo .pt '{label}' no encontrado: {path}. "
                    f"Coloca los .pt en app/model/modeloOriginal/"
                )

        logger.info(f"  Keypoint .pt : {kp_path}")
        logger.info(f"  Player   .pt : {pl_path}")
        logger.info(f"  Ball     .pt : {ba_path}")

        self.keypoint_model = YOLO(str(kp_path))
        self.player_model   = YOLO(str(pl_path))
        self.ball_model     = YOLO(str(ba_path))

    def _load_openvino_models(self):
        """Carga modelos OpenVINO IR para inferencia CPU Intel (Local)."""
        kp_path  = settings.MODEL_KEYPOINT_PATH
        pl_path  = settings.MODEL_PLAYER_PATH
        ba_path  = settings.MODEL_BALL_PATH

        for label, path in [("Keypoint", kp_path), ("Player", pl_path), ("Ball", ba_path)]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Modelo OpenVINO '{label}' no encontrado: {path}. "
                    f"Verifica la carpeta openVino/ en la raíz del proyecto."
                )

        logger.info(f"  Keypoint OpenVINO : {kp_path}")
        logger.info(f"  Player   OpenVINO : {pl_path}")
        logger.info(f"  Ball     OpenVINO : {ba_path}")

        self.keypoint_model = YOLO(str(kp_path), task="pose")
        self.player_model   = YOLO(str(pl_path), task="detect")
        self.ball_model     = YOLO(str(ba_path), task="detect")

    # ── Inferencia ───────────────────────────────────────────────────────────

    def infer_keypoints(self, frame_bgr, conf: float = 0.25):
        """Retorna resultado de keypoints compatible con MockKeypointResult."""
        return self.keypoint_model.predict(
            source=frame_bgr,
            conf=conf,
            device=self.device,
            verbose=False,
            save=False,
        )[0]

    def infer_players(self, frame_bgr, conf: float = 0.35, is_video: bool = True):
        """Retorna resultado de player tracking compatible con MockPlayerResult."""
        if is_video:
            try:
                return self.player_model.track(
                    source=frame_bgr,
                    conf=conf,
                    device=self.device,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                    save=False,
                )[0]
            except Exception:
                pass
        return self.player_model.predict(
            source=frame_bgr,
            conf=conf,
            device=self.device,
            verbose=False,
            save=False,
        )[0]

    def infer_ball(self, frame_bgr, conf: float = 0.15):
        """Retorna resultado de detección de balón."""
        return self.ball_model.predict(
            source=frame_bgr,
            conf=conf,
            device=self.device,
            verbose=False,
            save=False,
        )[0]

    def infer(
        self,
        frame_bgr,
        conf_kp: float = 0.25,
        conf_player: float = 0.35,
        conf_ball: float = 0.15,
        is_video: bool = True,
    ) -> Tuple:
        """
        Inferencia de los 3 modelos de forma secuencial.
        Retorna (kp_result, player_result, ball_result) —
        objetos nativos de Ultralytics YOLO, sin wrappers Mock.
        """
        kp_result     = self.infer_keypoints(frame_bgr, conf=conf_kp)
        player_result = self.infer_players(frame_bgr, conf=conf_player, is_video=is_video)
        ball_result   = self.infer_ball(frame_bgr, conf=conf_ball)
        return kp_result, player_result, ball_result
