import logging
import warnings
from typing import Tuple, Optional
from pathlib import Path
import time
import torch
from ultralytics import YOLO
from config import settings

warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("DirectInferenceEngine")
perf_logger = logging.getLogger("DirectInferenceEngine.perf")


class DirectInferenceEngine:
    """
    Motor de inferencia sin capa HTTP.
    Expone un único método `infer()` que retorna los mismos tipos
    que usaban los microservicios SOA (compatible con el pipeline existente).
    """

    def __init__(self):
        self.device = settings.INFERENCE_DEVICE
        self.format = settings.MODEL_FORMAT
        self.use_half = (self.device == "cuda")

        gpu_name = getattr(settings, "GPU_DEVICE_NAME", "")
        is_l4 = getattr(settings, "GPU_IS_L4", False)
        gpu_str = f", GPU={gpu_name} (L4={is_l4})" if gpu_name else ""

        logger.info(
            f"[Engine] Inicializando DirectInferenceEngine — "
            f"format={self.format}, device={self.device}, "
            f"CUDA={settings.CUDA_AVAILABLE}{gpu_str}, "
            f"FP16={self.use_half}"
        )

        if self.format == "pt":
            self._load_pt_models()
        else:
            self._load_openvino_models()

        logger.info("[Engine] ✅ Los 3 modelos están cargados en memoria.")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_pt_models(self):
        """Carga modelos .pt originales para inferencia CUDA (Colab L4 / Kaggle T4)."""
        kp_path  = settings.MODEL_KEYPOINT_PT_PATH
        pl_path  = settings.MODEL_PLAYER_PT_PATH
        ba_path  = settings.MODEL_BALL_PT_PATH

        for label, path in [("Keypoint", kp_path), ("Player", pl_path), ("Ball", ba_path)]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Modelo .pt '{label}' no encontrado en: {path}. "
                    f"Coloca los .pt en app/model/modeloOriginal/ o define MODEL_{label.upper()}_PT_PATH"
                )

        logger.info(f"  Keypoint .pt : {kp_path}")
        logger.info(f"  Player   .pt : {pl_path}")
        logger.info(f"  Ball     .pt : {ba_path}")

        self.keypoint_model = YOLO(str(kp_path))
        self.player_model   = YOLO(str(pl_path))
        self.ball_model     = YOLO(str(ba_path))

        if self.device == "cuda":
            gpu_name = getattr(settings, "GPU_DEVICE_NAME", "CUDA")
            logger.info(f"  [Engine] ⚡ Modelos preparados para aceleración en {gpu_name} (FP16={self.use_half})...")
            try:
                self.keypoint_model.to("cuda")
                self.player_model.to("cuda")
                self.ball_model.to("cuda")
                if self.use_half:
                    self.keypoint_model.model.half()
                    self.player_model.model.half()
                    self.ball_model.model.half()
            except Exception as e:
                logger.warning(f"  [Engine] Advertencia al mover modelos a CUDA: {e}")

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
        pred_kwargs = {
            "source": frame_bgr,
            "conf": conf,
            "device": self.device,
            "verbose": False,
            "save": False,
        }
        if self.use_half and self.format == "pt":
            pred_kwargs["half"] = True
        return self.keypoint_model.predict(**pred_kwargs)[0]

    def infer_players(self, frame_bgr, conf: float = 0.35, is_video: bool = True, iou: Optional[float] = None):
        """Retorna resultado de player tracking con BoTSORT."""
        nms_iou = iou if iou is not None else getattr(settings, "PLAYER_NMS_IOU", 0.45)
        pred_kwargs = {
            "source": frame_bgr,
            "conf": conf,
            "iou": nms_iou,
            "device": self.device,
            "verbose": False,
            "save": False,
        }
        if self.use_half and self.format == "pt":
            pred_kwargs["half"] = True

        if is_video:
            tracker_cfg = str(settings.TRACKER_CONFIG_PATH) if hasattr(settings, "TRACKER_CONFIG_PATH") and Path(settings.TRACKER_CONFIG_PATH).exists() else "botsort.yaml"
            try:
                return self.player_model.track(
                    persist=True,
                    tracker=tracker_cfg,
                    **pred_kwargs,
                )[0]
            except Exception as e:
                logger.warning(f"[DirectInferenceEngine] Fallback en track({tracker_cfg}): {e}, intentando botsort.yaml...")
                try:
                    return self.player_model.track(
                        persist=True,
                        tracker="botsort.yaml",
                        **pred_kwargs,
                    )[0]
                except Exception:
                    pass
        return self.player_model.predict(**pred_kwargs)[0]


    def infer_ball(self, frame_bgr, conf: float = 0.15):
        """Retorna resultado de detección de balón."""
        pred_kwargs = {
            "source": frame_bgr,
            "conf": conf,
            "device": self.device,
            "verbose": False,
            "save": False,
        }
        if self.use_half and self.format == "pt":
            pred_kwargs["half"] = True
        return self.ball_model.predict(**pred_kwargs)[0]

    def infer(
        self,
        frame_bgr,
        conf_kp: float = 0.25,
        conf_player: float = 0.35,
        conf_ball: float = 0.15,
        is_video: bool = True,
        iou_player: Optional[float] = None,
    ) -> Tuple:
        """
        Inferencia de los 3 modelos de forma secuencial con profiling de tiempos.
        Retorna (kp_result, player_result, ball_result) —
        objetos nativos de Ultralytics YOLO, sin wrappers Mock.
        """
        t0 = time.perf_counter()
        kp_result = self.infer_keypoints(frame_bgr, conf=conf_kp)
        t1 = time.perf_counter()

        player_result = self.infer_players(frame_bgr, conf=conf_player, is_video=is_video, iou=iou_player)
        t2 = time.perf_counter()

        ball_result = self.infer_ball(frame_bgr, conf=conf_ball)
        t3 = time.perf_counter()

        ms_kp     = (t1 - t0) * 1000
        ms_player = (t2 - t1) * 1000
        ms_ball   = (t3 - t2) * 1000
        ms_total  = (t3 - t0) * 1000

        perf_logger.debug(
            f"⏱  Inferencia — "
            f"Keypoint: {ms_kp:6.1f}ms | "
            f"Player:   {ms_player:6.1f}ms | "
            f"Ball:     {ms_ball:6.1f}ms | "
            f"TOTAL:    {ms_total:6.1f}ms"
        )

        return kp_result, player_result, ball_result

