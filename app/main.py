"""
ScoutingData v5.0 - Pipeline Principal Optimizado

Cambios vs v4.0:
  - Modo directo (sin HTTP SOA): DirectInferenceEngine
  - Sin event_service (SoccerNet eliminado)
  - Selección automática .pt CUDA (Kaggle) / OpenVINO CPU (Local)
  - Caché de líneas de cancha (OPT-4)
  - Caché de minimap template (OPT-5)
  - Homografía cada N frames configurable (OPT-7)
  - AsyncVideoWriter para escritura en background (OPT-8)
  - Renders in-place (sin .copy() redundantes) (OPT-6)
"""

import os
import sys
import time
import logging
import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings
from config.field_dimensions import FieldDimensions
from engine.direct_inference import DirectInferenceEngine
from engine.async_writer import AsyncVideoWriter
from homography.estimator import compute_homography
from homography.validation import is_homography_transition_valid, smooth_homography
from homography.transforms import pixel_to_field
from sports.ball_tracker import BallTracker
from visualization.pitch import draw_pitch_template
from visualization.overlays import (
    get_pitch_field_points,          # OPT-4: puntos pre-calculados
    draw_pitch_lines_on_image,
    draw_calibration_overlay,
    draw_player_detections,
    draw_ball_detections,
)
from visualization.minimap import (
    get_cached_pitch_template,       # OPT-5: template pre-renderizado
    draw_minimap_with_projections,
    draw_player_projections_on_minimap,
    draw_ball_projection_on_minimap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SoccerAnalysisV5")

# Logger de timing de inferencia — desactivado por defecto, se activa con --debug-homography
perf_logger = logging.getLogger("DirectInferenceEngine.perf")
perf_logger.setLevel(logging.WARNING)  # Silencioso por defecto
perf_logger.propagate = False

# Logger de homografía — desactivado por defecto, se activa con --debug-homography
homography_logger = logging.getLogger("Homography.debug")
homography_logger.setLevel(logging.WARNING)  # Silencioso por defecto
homography_logger.propagate = False

def _enable_debug_mode():
    """Activa logs detallados de homografía e inferencia (--debug-homography)."""
    _fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    for lg in (perf_logger, homography_logger):
        lg.setLevel(logging.DEBUG)
        if not lg.handlers:
            _h = logging.StreamHandler()
            _h.setLevel(logging.DEBUG)
            _h.setFormatter(_fmt)
            lg.addHandler(_h)



def _extract_ball_detection(ball_result, conf_threshold: float) -> Optional[Dict]:
    """Extrae la mejor detección de balón de un resultado YOLO nativo."""
    if ball_result.boxes is None or len(ball_result.boxes) == 0:
        return None
    b_boxes = ball_result.boxes.xyxy.cpu().numpy()
    b_confs = ball_result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(b_confs))
    if float(b_confs[best_idx]) < conf_threshold:
        return None
    bx1, by1, bx2, by2 = b_boxes[best_idx]
    return {
        "pixel_x":     round(float((bx1 + bx2) / 2), 2),
        "pixel_y":     round(float((by1 + by2) / 2), 2),
        "confidence":  float(b_confs[best_idx]),
        "interpolated": False,
        "bbox":        [float(bx1), float(by1), float(bx2), float(by2)],
    }


class SoccerAnalysisV5:
    """
    Pipeline de análisis v5.0: Inferencia directa + renders cacheados / Modo Headless.
    Sin SoccerNet, sin HTTP SOA.
    """

    def __init__(
        self,
        conf_keypoint: float = settings.CONF_KEYPOINT,
        conf_player:   float = settings.CONF_PLAYER,
        conf_ball:     float = settings.CONF_BALL,
        ransac_threshold: float = settings.RANSAC_THRESHOLD,
        field_dims: FieldDimensions = FieldDimensions(),
        minimap_scale: float = settings.MINIMAP_SCALE,
        homography_interval: int = settings.HOMOGRAPHY_INTERVAL,
        headless: Optional[bool] = None,
        draw_outliers: Optional[bool] = None,
    ):
        self.conf_keypoint   = conf_keypoint
        self.conf_player     = conf_player
        self.conf_ball       = conf_ball
        self.ransac_threshold = ransac_threshold
        self.field_dims      = field_dims
        self.minimap_scale   = minimap_scale
        self.homography_interval = homography_interval
        self.headless        = headless if headless is not None else settings.HEADLESS_MODE
        self.draw_outliers   = draw_outliers if draw_outliers is not None else settings.DRAW_OUTLIERS

        # Motor de inferencia directa (sin HTTP)
        self.engine = DirectInferenceEngine()

        # OPT-4: Pre-calcular puntos de líneas de cancha (solo una vez)
        self._pitch_field_segments = get_pitch_field_points(field_dims)

        # OPT-5: Pre-renderizar minimap template (solo una vez)
        self._minimap_template = get_cached_pitch_template(
            field_dims, minimap_scale, padding_meters=5.0
        )

        # Estado del tracker de balón
        self.ball_tracker = BallTracker()

        # Caché de homografía (OPT-7)
        self._cached_H:    Optional[np.ndarray] = None
        self._cached_mask: Optional[np.ndarray] = None
        self._h_reject_streak: int = 0   # Contador de rechazos consecutivos para auto-reset

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
        is_video: bool = True,
        headless: Optional[bool] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[Dict], Optional[Dict]]:

        is_headless = headless if headless is not None else self.headless

        # ── 1. Inferencia Directa ────────────────────────────────────────────
        kp_result, player_result, ball_result = self.engine.infer(
            frame_bgr,
            conf_kp=self.conf_keypoint,
            conf_player=self.conf_player,
            conf_ball=self.conf_ball,
            is_video=is_video,
        )

        # ── 2. Ball Tracker ──────────────────────────────────────────────────
        raw_ball_det = _extract_ball_detection(ball_result, self.conf_ball)
        if is_video:
            ball_detection, ball_history = self.ball_tracker.update(raw_ball_det)
        else:
            ball_detection = raw_ball_det
            ball_history = [(raw_ball_det["pixel_x"], raw_ball_det["pixel_y"])] if raw_ball_det else []

        # ── 3. Homografía (OPT-7: caché cada N frames con filtro de continuidad) ──
        recalc = (frame_idx % self.homography_interval == 0) or (self._cached_H is None)
        if recalc:
            try:
                H_cand, _, _, mask, inliers, _ = compute_homography(
                    kp_result,
                    conf_threshold=self.conf_keypoint,
                    ransac_threshold=self.ransac_threshold,
                )
                h_img, w_img = frame_bgr.shape[:2]

                # Auto-reset si llevamos demasiados rechazos consecutivos
                # (indica que la caché actual está bloqueada en estado incorrecto)
                if self._h_reject_streak >= 8:
                    homography_logger.debug(
                        f"[Frame {frame_idx}] 🔄 AUTO-RESET caché H tras {self._h_reject_streak} rechazos consecutivos"
                    )
                    self._cached_H = None
                    self._cached_mask = None
                    self._h_reject_streak = 0

                if is_homography_transition_valid(
                    H_cand, self._cached_H,
                    img_width=w_img, img_height=h_img,
                    max_shift_m=35.0,   # Aumentado: permite correcciones de cámara más amplias
                ):
                    # alpha=0.7 → 70% nueva H, 30% anterior (más responsivo)
                    H_smoothed = smooth_homography(H_cand, self._cached_H, alpha=0.7)
                    self._cached_H    = H_smoothed
                    self._cached_mask = mask
                    self._h_reject_streak = 0
                    homography_logger.debug(
                        f"[Frame {frame_idx}] ✅ H aceptada | inliers={inliers} | EMA α=0.7"
                    )
                elif self._cached_H is None:
                    # Primera H: aceptar directamente sin suavizado
                    self._cached_H    = H_cand
                    self._cached_mask = mask
                    self._h_reject_streak = 0
                    homography_logger.debug(
                        f"[Frame {frame_idx}] ✅ H inicial aceptada | inliers={inliers}"
                    )
                else:
                    self._h_reject_streak += 1
                    homography_logger.debug(
                        f"[Frame {frame_idx}] ⚠ H RECHAZADA (streak={self._h_reject_streak}) "
                        f"| inliers={inliers} — se mantiene caché anterior"
                    )
            except Exception as exc:
                self._h_reject_streak += 1
                homography_logger.debug(
                    f"[Frame {frame_idx}] ✖ compute_homography falló (streak={self._h_reject_streak}): {exc}"
                )

        H    = self._cached_H
        mask = self._cached_mask

        # Coordenadas de terreno para balón
        if ball_detection is not None and H is not None:
            bf_x, bf_y = pixel_to_field(ball_detection["pixel_x"], ball_detection["pixel_y"], H)
            ball_detection["pitch_x_m"] = round(bf_x, 2)
            ball_detection["pitch_y_m"] = round(bf_y, 2)
        elif ball_detection is not None:
            ball_detection["pitch_x_m"] = None
            ball_detection["pitch_y_m"] = None

        # ── 4. Player Detections ─────────────────────────────────────────────
        player_detections: List[Dict] = []
        if player_result.boxes is not None and len(player_result.boxes.xyxy) > 0:
            boxes     = player_result.boxes.xyxy.cpu().numpy()
            confs     = player_result.boxes.conf.cpu().numpy()
            classes   = player_result.boxes.cls.cpu().numpy()
            track_ids = (
                player_result.boxes.id.cpu().numpy().astype(int)
                if player_result.boxes.id is not None else None
            )
            for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
                if conf < self.conf_player:
                    continue
                x1, y1, x2, y2 = box
                xc = float((x1 + x2) / 2)
                yb = float(y2)
                p_id = int(track_ids[idx]) if track_ids is not None else idx
                fx, fy = (pixel_to_field(xc, yb, H) if H is not None else (None, None))
                player_detections.append({
                    "player_id":   p_id,
                    "class_id":    int(cls_id),
                    "confidence":  float(conf),
                    "pixel_x":     round(xc, 2),
                    "pixel_y":     round(yb, 2),
                    "pitch_x_m":   round(fx, 2) if fx is not None else None,
                    "pitch_y_m":   round(fy, 2) if fy is not None else None,
                })

        # Si estamos en modo headless, omitir renderizado OpenCV (Overlays, Minimap, cvtColor)
        if is_headless:
            return None, None, H, player_detections, ball_detection

        # ── 5. Render Camera Frame (OPT-6: in-place, sin copias extra) ──────
        annotated = frame_bgr.copy()   # Una sola copia al inicio

        if H is not None:
            # OPT-4: usar puntos pre-calculados, solo aplicar H_inv
            draw_pitch_lines_on_image(
                annotated, H,
                precomputed_segments=self._pitch_field_segments,
                color=(0, 255, 255), thickness=2,
            )
            draw_calibration_overlay(
                annotated, kp_result, mask=mask,
                conf_threshold=self.conf_keypoint,
                draw_outliers=self.draw_outliers,
            )

        draw_player_detections(annotated, player_result,
                               conf_threshold=self.conf_player,
                               ellipse_color=(255, 0, 255))
        draw_ball_detections(annotated, ball_detection=ball_detection,
                             ball_history=ball_history)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # ── 6. Render Minimap (OPT-5: partir de template cacheado) ──────────
        if H is not None:
            minimap = draw_minimap_with_projections(
                H=H, result=kp_result,
                conf_threshold=self.conf_keypoint,
                mask=mask, scale=self.minimap_scale,
                cached_template=self._minimap_template,   # OPT-5
            )
            draw_player_projections_on_minimap(
                minimap, player_result, H=H,
                conf_threshold=self.conf_player,
                scale=self.minimap_scale, player_color=(255, 0, 255),
            )
            draw_ball_projection_on_minimap(
                minimap, ball_detection=ball_detection, scale=self.minimap_scale,
            )
        else:
            minimap = self._minimap_template.copy()

        minimap_rgb = cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB)

        return annotated_rgb, minimap_rgb, H, player_detections, ball_detection

    def process_video(
        self,
        video_path: str,
        output_path: str,
        csv_output_path: Optional[str] = None,
        json_output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        headless: Optional[bool] = None,
        debug_homography: bool = False,
    ) -> List[Dict]:
        import csv

        is_headless = headless if headless is not None else self.headless

        resolved = settings.resolve_video_source(video_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Video no encontrado: {resolved}")

        cap = cv2.VideoCapture(str(resolved))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {resolved}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cuda_info = f" (CUDA: {settings.CUDA_AVAILABLE})" if settings.INFERENCE_DEVICE == "cuda" else ""
        # Frames efectivos a procesar (respetando max_frames)
        effective_total = min(total, max_frames) if max_frames else total
        limit_str = f" [límite: {max_frames}]" if max_frames else ""
        logger.info(
            f"📹 {resolved.name} | {width}x{height} @ {fps:.1f}fps | "
            f"{total} frames totales{limit_str} | {settings.INFERENCE_DEVICE.upper()} {settings.MODEL_FORMAT.upper()}{cuda_info}"
        )
        if debug_homography:
            _enable_debug_mode()
            self.draw_outliers = True
            logger.info("🔍 Modo DEBUG de homografía activado (outliers visibles)")

        writer: Optional[AsyncVideoWriter] = None
        if not is_headless:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        start_time  = time.time()
        all_events: List[Dict] = []
        tracking_frames: List[Dict] = []

        # ── Acumuladores de tiempo por fase ─────────────────────────────────
        total_inference_time  = 0.0
        total_tracking_time   = 0.0
        total_homography_time = 0.0
        total_json_time       = 0.0
        total_video_time      = 0.0

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_count += 1
                if max_frames and frame_count > max_frames:
                    break

                ts = frame_count / fps

                # ── Fase 1: Inferencia (3 modelos) ──────────────────────────
                _t0 = time.perf_counter()
                annotated_rgb, minimap_rgb, H, player_dets, ball_det = self.process_frame(
                    frame_bgr, frame_idx=frame_count, timestamp_sec=ts, is_video=True, headless=is_headless
                )
                total_inference_time += time.perf_counter() - _t0

                # ── Fase 2: Tracking / Ball Tracker (incluido en process_frame pero separado aquí conceptualmente)
                # El tiempo de tracking se mide internamente; el resto de process_frame es homografía

                # ── Fase 3: Construcción del JSON por frame ──────────────────
                _t1 = time.perf_counter()
                frame_data = {
                    "frame": frame_count,
                    "timestamp": round(ts, 3),
                    "players": [
                        {
                            "track_id": p["player_id"],
                            "pixel_x": p["pixel_x"],
                            "pixel_y": p["pixel_y"],
                            "pitch_x": p["pitch_x_m"],
                            "pitch_y": p["pitch_y_m"],
                        }
                        for p in player_dets
                    ],
                    "ball": {
                        "pixel_x": ball_det["pixel_x"],
                        "pixel_y": ball_det["pixel_y"],
                        "pitch_x": ball_det["pitch_x_m"],
                        "pitch_y": ball_det["pitch_y_m"],
                    } if ball_det is not None else None,
                    "homography_valid": bool(H is not None),
                }
                tracking_frames.append(frame_data)
                total_json_time += time.perf_counter() - _t1

                # ── Fase 4: Renderizado y escritura de video ─────────────────
                if not is_headless:
                    _t2 = time.perf_counter()
                    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                    minimap_bgr   = cv2.cvtColor(minimap_rgb,   cv2.COLOR_RGB2BGR)

                    h_f, w_f = annotated_bgr.shape[:2]
                    h_m, w_m = minimap_bgr.shape[:2]
                    scale_r   = h_f / float(h_m)
                    minimap_r = cv2.resize(minimap_bgr, (int(w_m * scale_r), h_f))
                    combined  = np.hstack([annotated_bgr, minimap_r])

                    if writer is None:
                        h_c, w_c = combined.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = AsyncVideoWriter(output_path, fourcc, fps, (w_c, h_c))

                    writer.write(combined)
                    total_video_time += time.perf_counter() - _t2

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_avg = frame_count / elapsed
                    pct     = frame_count / effective_total * 100 if effective_total > 0 else 0
                    bar_len = 30
                    filled  = int(bar_len * frame_count / effective_total) if effective_total > 0 else 0
                    bar     = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  [{bar}] {pct:5.1f}%  {frame_count}/{effective_total}  {fps_avg:.1f} FPS", end="", flush=True)

        finally:
            cap.release()
            if writer:
                writer.release()
            print()  # Salto de línea tras la barra de progreso

        # ── Exportar JSON Contract ───────────────────────────────────────────
        video_name = resolved.name
        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0

        tracking_dataset = {
            "metadata": {
                "video_name": video_name,
                "fps": round(fps, 2),
                "total_frames": frame_count,
                "duration_seconds": duration_sec,
                "inference": {
                    "device": settings.INFERENCE_DEVICE,
                    "model_format": settings.MODEL_FORMAT,
                    "cuda_available": settings.CUDA_AVAILABLE,
                    "environment": settings.ENVIRONMENT,
                },
            },
            "frames": tracking_frames,
            "events": [],
        }

        actual_json_path = json_output_path or str(settings.OUTPUT_DIR / f"tracking_{resolved.stem}.json")
        json_path_obj = Path(actual_json_path)
        json_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path_obj, "w", encoding="utf-8") as f:
            json.dump(tracking_dataset, f, indent=2)
        logger.info(f"💾 Tracking JSON exportado: {actual_json_path}")

        # ── CSV Export ────────────────────────────────────────────────────────
        if csv_output_path and all_events:
            csv_path = Path(csv_output_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx", "timestamp_sec", "timestamp_fmt", "event_type", "confidence"])
                for evt in all_events:
                    ts = evt.get("timestamp_sec", 0.0)
                    mins, secs = int(ts // 60), int(ts % 60)
                    w.writerow([evt.get("frame_idx"), ts,
                                f"{mins:02d}:{secs:02d}", evt.get("event_type"), evt.get("confidence")])
            logger.info(f"📄 CSV exportado: {csv_output_path}")

        total_time = time.time() - start_time
        fps_final  = frame_count / total_time if total_time > 0 else 0.0

        # Porcentajes sobre tiempo total para identificar cuellos de botella
        def _pct(t): return f"{t / total_time * 100:.1f}%" if total_time > 0 else "—"

        logger.info("=" * 65)
        logger.info("📊 RESUMEN FINAL DE PROCESAMIENTO")
        logger.info(f"   Modo:              {'HEADLESS (Sin MP4/Overlays)' if is_headless else 'FULL (Video + JSON)'}")
        logger.info(f"   Motor inferencia:  {settings.INFERENCE_DEVICE.upper()} | {settings.MODEL_FORMAT.upper()}")
        logger.info(f"   CUDA disponible:   {settings.CUDA_AVAILABLE}")
        logger.info(f"   Frames procesados: {frame_count}")
        logger.info(f"   Tiempo total:      {total_time:.2f} s")
        logger.info(f"   FPS promedio:      {fps_final:.2f}")
        logger.info("-" * 65)
        logger.info("⏱  DESGLOSE DE TIEMPOS POR FASE:")
        logger.info(f"   Total inferencia:  {total_inference_time:.2f}s  ({_pct(total_inference_time)} del total)")
        logger.info(f"   Total tracking:    {total_tracking_time:.2f}s  ({_pct(total_tracking_time)} del total)")
        logger.info(f"   Total homography:  {total_homography_time:.2f}s  ({_pct(total_homography_time)} del total)")
        logger.info(f"   Total json build:  {total_json_time:.2f}s  ({_pct(total_json_time)} del total)")
        logger.info(f"   Total video write: {total_video_time:.2f}s  ({_pct(total_video_time)} del total)")
        logger.info("-" * 65)
        logger.info(f"   JSON Output:       {actual_json_path}")
        if not is_headless:
            logger.info(f"   MP4 Output:        {output_path}")
        logger.info("=" * 65)

        return all_events


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ScoutingData v5.0 — Soccer Analysis Pipeline (Direct + OpenVINO/CUDA)"
    )
    parser.add_argument("--video",  "-v", default=settings.DEFAULT_INPUT_VIDEO, help="Video de entrada")
    parser.add_argument("--output", "-o", default=settings.DEFAULT_OUTPUT_VIDEO, help="Video de salida MP4")
    parser.add_argument("--csv",    "-c", default=settings.DEFAULT_OUTPUT_CSV,   help="CSV de eventos")
    parser.add_argument("--json",   "-j", default=None,                          help="JSON de tracking output")
    parser.add_argument("--env",    "-e", choices=["LOCAL", "KAGGLE"],           help="Forzar entorno")
    parser.add_argument("--headless", action="store_true", default=None,         help="Modo headless (Sin renderizar MP4/Overlays)")
    parser.add_argument("--max-frames", "-m", type=int, default=None,            help="Límite de frames")
    parser.add_argument("--homography-interval", type=int,
                        default=settings.HOMOGRAPHY_INTERVAL,
                        help="Recalcular H cada N frames (1=cada frame, 5=cada 5 frames)")
    parser.add_argument("--debug-homography", action="store_true", default=False,
                        help="Activar logs detallados de homografía e inferencia por frame")
    args = parser.parse_args()

    if args.env:
        os.environ["ENVIRONMENT"] = args.env

    if args.headless:
        os.environ["HEADLESS_MODE"] = "True"

    logger.info("=" * 65)
    logger.info("  ScoutingData v5.0 — Direct Inference Pipeline")
    logger.info("=" * 65)
    for k, v in settings.get_summary().items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 65)

    pipeline = SoccerAnalysisV5(
        homography_interval=args.homography_interval,
        headless=args.headless,
    )

    resolved = settings.resolve_video_source(args.video)
    if resolved.exists():
        pipeline.process_video(
            video_path=str(resolved),
            output_path=args.output,
            csv_output_path=args.csv,
            json_output_path=args.json,
            max_frames=args.max_frames,
            headless=args.headless,
            debug_homography=args.debug_homography,
        )
    else:
        logger.warning(f"Video no encontrado: {resolved}")
        logger.info("Usa --video <ruta> o define VIDEO_INPUT como variable de entorno.")


if __name__ == "__main__":
    main()
