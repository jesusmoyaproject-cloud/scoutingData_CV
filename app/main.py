import os
import sys
import time
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import cv2
import numpy as np

# Ensure app root is in sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import settings
from config.field_dimensions import FieldDimensions
from homography.estimator import compute_homography
from homography.transforms import pixel_to_field
from sports.ball_tracker import BallTracker
from visualization.pitch import draw_pitch_template
from visualization.overlays import (
    draw_pitch_lines_on_image,
    draw_calibration_overlay,
    draw_player_detections,
    draw_ball_detections,
    draw_event_banner_overlay
)
from visualization.minimap import (
    draw_minimap_with_projections,
    draw_player_projections_on_minimap,
    draw_ball_projection_on_minimap
)
from services.client.inference_client import AsyncInferenceClient, MockKeypointResult, MockPlayerResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SoccerAnalysisSOA")

class SoccerAnalysisSOA:
    def __init__(
        self,
        keypoint_url: str = settings.KEYPOINT_SERVICE_URL,
        player_url: str = settings.PLAYER_SERVICE_URL,
        ball_url: str = settings.BALL_SERVICE_URL,
        event_url: str = settings.EVENT_SERVICE_URL,
        conf_keypoint: float = settings.CONF_KEYPOINT,
        conf_player: float = settings.CONF_PLAYER,
        conf_ball: float = settings.CONF_BALL,
        conf_event: float = settings.CONF_EVENT,
        ransac_threshold: float = settings.RANSAC_THRESHOLD,
        field_dims: FieldDimensions = FieldDimensions(),
        minimap_scale: float = settings.MINIMAP_SCALE
    ):
        self.conf_keypoint = conf_keypoint
        self.conf_player = conf_player
        self.conf_ball = conf_ball
        self.conf_event = conf_event
        self.ransac_threshold = ransac_threshold
        self.field_dims = field_dims
        self.minimap_scale = minimap_scale

        self.client = AsyncInferenceClient(
            keypoint_url=keypoint_url,
            player_url=player_url,
            ball_url=ball_url,
            event_url=event_url
        )
        self.ball_tracker = BallTracker()
        self.last_possessor_id: Optional[int] = None
        self.current_possessor_id: Optional[int] = None
        self.active_event_banners: List[Dict] = []

    async def process_frame_async(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
        is_video: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[Dict], Optional[Dict], List[Dict]]:
        # 1. Ejecución paralela vía Microservicios SOA (AsyncInferenceClient)
        kp_result, player_result, raw_ball_det, events_result = await self.client.process_frame_parallel(
            frame_bgr,
            conf_keypoint=self.conf_keypoint,
            conf_player=self.conf_player,
            conf_ball=self.conf_ball,
            conf_event=self.conf_event,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            pass_from_player_id=self.last_possessor_id,
            pass_to_player_id=self.current_possessor_id,
            is_video=is_video
        )

        # 2. Actualizar Ball Tracker
        if is_video:
            ball_detection, ball_history = self.ball_tracker.update(raw_ball_det)
        else:
            ball_detection = raw_ball_det
            ball_history = [(raw_ball_det["pixel_x"], raw_ball_det["pixel_y"])] if raw_ball_det else []

        # 3. Compute Homography
        try:
            H, image_pts, field_pts, mask, inliers, outliers = compute_homography(
                kp_result,
                conf_threshold=self.conf_keypoint,
                ransac_threshold=self.ransac_threshold
            )
        except Exception:
            H, mask = None, None

        # Coordenadas de terreno para balón
        if ball_detection is not None and H is not None:
            bf_x, bf_y = pixel_to_field(ball_detection["pixel_x"], ball_detection["pixel_y"], H)
            ball_detection["pitch_x_m"] = round(bf_x, 2)
            ball_detection["pitch_y_m"] = round(bf_y, 2)
        elif ball_detection is not None:
            ball_detection["pitch_x_m"] = None
            ball_detection["pitch_y_m"] = None

        # 4. Extract Player Field Detections & Calcular Jugador más cercano al Balón
        player_detections: List[Dict] = []
        closest_player_id = None
        min_ball_dist = float("inf")

        if player_result.boxes is not None and len(player_result.boxes.xyxy) > 0:
            boxes = player_result.boxes.xyxy
            confs = player_result.boxes.conf
            classes = player_result.boxes.cls
            track_ids = player_result.boxes.id

            for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
                if conf < self.conf_player:
                    continue

                x1, y1, x2, y2 = box
                x_center = float((x1 + x2) / 2.0)
                y_bottom = float(y2)
                p_id = int(track_ids[idx]) if track_ids is not None else idx

                field_x, field_y = None, None
                if H is not None:
                    field_x, field_y = pixel_to_field(x_center, y_bottom, H)

                player_detections.append({
                    "player_id": p_id,
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "pixel_x": round(x_center, 2),
                    "pixel_y": round(y_bottom, 2),
                    "pitch_x_m": round(field_x, 2) if field_x is not None else None,
                    "pitch_y_m": round(field_y, 2) if field_y is not None else None
                })

                if ball_detection is not None:
                    dist = np.hypot(x_center - ball_detection["pixel_x"], y_bottom - ball_detection["pixel_y"])
                    if dist < min_ball_dist:
                        min_ball_dist = dist
                        closest_player_id = p_id

        # Actualizar IDs de poseedor del balón (Emisor / Receptor)
        if closest_player_id is not None and min_ball_dist < 150.0:
            if self.current_possessor_id is not None and closest_player_id != self.current_possessor_id:
                self.last_possessor_id = self.current_possessor_id
                logger.info(f"🔄 Transición de balón detectada: Jugador #{self.last_possessor_id} ➔ Jugador #{closest_player_id} (Dist: {min_ball_dist:.1f}px)")
            self.current_possessor_id = closest_player_id

        # 5. Render Camera Frame Overlays
        annotated_img = frame_bgr.copy()
        if H is not None:
            annotated_img = draw_pitch_lines_on_image(annotated_img, H, color=(0, 255, 255), thickness=2)
            annotated_img = draw_calibration_overlay(annotated_img, kp_result, mask=mask, conf_threshold=self.conf_keypoint)

        annotated_img = draw_player_detections(
            annotated_img,
            player_result,
            conf_threshold=self.conf_player,
            ellipse_color=(255, 0, 255)
        )

        annotated_img = draw_ball_detections(
            annotated_img,
            ball_detection=ball_detection,
            ball_history=ball_history
        )

        # Administrar visibilidad de banners de eventos en video (duración: 45 fotogramas / 1.5s)
        if events_result:
            for evt in events_result:
                self.active_event_banners.append({"evt": evt, "expiry_frame": frame_idx + 45})

        self.active_event_banners = [b for b in self.active_event_banners if b["expiry_frame"] >= frame_idx]

        if self.active_event_banners:
            active_evts_list = [b["evt"] for b in self.active_event_banners]
            annotated_img = draw_event_banner_overlay(annotated_img, active_evts_list)

        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # 6. Render 2D Minimap
        if H is not None:
            minimap_bgr = draw_minimap_with_projections(
                H=H,
                result=kp_result,
                conf_threshold=self.conf_keypoint,
                mask=mask,
                scale=self.minimap_scale
            )
            minimap_bgr = draw_player_projections_on_minimap(
                minimap_bgr,
                player_result,
                H=H,
                conf_threshold=self.conf_player,
                scale=self.minimap_scale,
                player_color=(255, 0, 255)
            )
            minimap_bgr = draw_ball_projection_on_minimap(
                minimap_bgr,
                ball_detection=ball_detection,
                scale=self.minimap_scale
            )
        else:
            minimap_bgr, _, _ = draw_pitch_template(scale=self.minimap_scale)

        minimap_rgb = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2RGB)

        return annotated_rgb, minimap_rgb, H, mask, player_detections, ball_detection, events_result

    async def process_video_async(
        self,
        video_path: str,
        output_path: str,
        csv_output_path: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        import csv
        
        resolved_input = settings.resolve_video_source(video_path)
        if not resolved_input.exists():
            raise FileNotFoundError(f"Cannot open video source: {resolved_input}")

        cap = cv2.VideoCapture(str(resolved_input))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {resolved_input}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing Video (SOA Parallel + SoccerNet Events): {resolved_input} ({width}x{height} @ {fps:.1f}fps, {total_frames} frames)")

        out_path_obj = Path(output_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)
        out_writer = None

        frame_count = 0
        start_time = time.time()
        all_events = []

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            timestamp_sec = frame_count / fps

            annotated_rgb, minimap_rgb, H, _, _, _, events = await self.process_frame_async(
                frame_bgr,
                frame_idx=frame_count,
                timestamp_sec=timestamp_sec,
                is_video=True
            )

            if events:
                for evt in events:
                    p_from = evt.get("pass_from_player_id")
                    p_to = evt.get("pass_to_player_id")
                    pass_info = f" [Pase: #{p_from} -> #{p_to}]" if (p_from is not None or p_to is not None) else ""
                    logger.info(f"⚽ [EVENTO DETECTADO @ {evt['timestamp_sec']}s] {evt['event_type']} (Conf: {evt['confidence']:.2f}){pass_info}")
                    all_events.append(evt)

            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            minimap_bgr = cv2.cvtColor(minimap_rgb, cv2.COLOR_RGB2BGR)

            h_frame, w_frame, _ = annotated_bgr.shape
            h_mini, w_mini, _ = minimap_bgr.shape
            scale_ratio = h_frame / float(h_mini)
            w_mini_resized = int(w_mini * scale_ratio)
            minimap_resized = cv2.resize(minimap_bgr, (w_mini_resized, h_frame))

            combined_bgr = np.hstack([annotated_bgr, minimap_resized])

            if out_writer is None:
                h_comb, w_comb, _ = combined_bgr.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(str(out_path_obj), fourcc, fps, (w_comb, h_comb))

            out_writer.write(combined_bgr)

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_avg = frame_count / elapsed
                logger.info(f"Processed {frame_count}/{total_frames} frames ({fps_avg:.2f} FPS) - Eventos detectados: {len(all_events)}")

        cap.release()
        if out_writer:
            out_writer.release()

        # Exportar eventos a CSV con IDs de Pase
        if csv_output_path:
            csv_path = Path(csv_output_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_idx",
                    "timestamp_sec",
                    "timestamp_formatted",
                    "event_type",
                    "confidence",
                    "pass_from_player_id",
                    "pass_to_player_id",
                    "spatial_location"
                ])
                for evt in all_events:
                    ts = evt.get("timestamp_sec", 0.0)
                    mins = int(ts // 60)
                    secs = int(ts % 60)
                    ms = int((ts - int(ts)) * 1000)
                    ts_fmt = f"{mins:02d}:{secs:02d}.{ms:03d}"
                    writer.writerow([
                        evt.get("frame_idx", 0),
                        ts,
                        ts_fmt,
                        evt.get("event_type", ""),
                        evt.get("confidence", 0.0),
                        evt.get("pass_from_player_id", ""),
                        evt.get("pass_to_player_id", ""),
                        str(evt.get("spatial_location", ""))
                    ])
            logger.info(f"📄 Eventos exportados a CSV: {csv_path} ({len(all_events)} eventos)")

        total_time = time.time() - start_time
        logger.info(f"Video processing finished in {total_time:.2f}s ({frame_count / total_time:.2f} FPS). Output saved to: {output_path}")
        await self.client.close()
        return all_events


async def main():
    parser = argparse.ArgumentParser(description="ScoutingData v4.0 - Soccer Video Analysis Pipeline (SOA + OpenVINO)")
    parser.add_argument("--video", "-v", type=str, default=settings.DEFAULT_INPUT_VIDEO, help="Ruta al video de entrada")
    parser.add_argument("--output", "-o", type=str, default=settings.DEFAULT_OUTPUT_VIDEO, help="Ruta al video de salida")
    parser.add_argument("--csv", "-c", type=str, default=settings.DEFAULT_OUTPUT_CSV, help="Ruta al reporte CSV de eventos")
    parser.add_argument("--env", "-e", type=str, choices=["LOCAL", "KAGGLE"], default=settings.ENVIRONMENT, help="Entorno de ejecución")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Límite de fotogramas a procesar")
    parser.add_argument("--keypoint-url", type=str, default=settings.KEYPOINT_SERVICE_URL, help="URL servicio Keypoints")
    parser.add_argument("--player-url", type=str, default=settings.PLAYER_SERVICE_URL, help="URL servicio Jugadores")
    parser.add_argument("--ball-url", type=str, default=settings.BALL_SERVICE_URL, help="URL servicio Balón")
    parser.add_argument("--event-url", type=str, default=settings.EVENT_SERVICE_URL, help="URL servicio Eventos")

    args = parser.parse_args()

    if args.env:
        os.environ["ENVIRONMENT"] = args.env

    logger.info(f"Starting SOA Pipeline v4.0 (Environment: {settings.ENVIRONMENT})")
    logger.info(f"Input Video: {args.video}")
    logger.info(f"Output Video: {args.output}")

    pipeline = SoccerAnalysisSOA(
        keypoint_url=args.keypoint_url,
        player_url=args.player_url,
        ball_url=args.ball_url,
        event_url=args.event_url
    )

    resolved_video = settings.resolve_video_source(args.video)
    if resolved_video.exists():
        await pipeline.process_video_async(
            video_path=str(resolved_video),
            output_path=args.output,
            csv_output_path=args.csv,
            max_frames=args.max_frames
        )
    else:
        logger.warning(f"Video source file not found: {resolved_video}")
        logger.info("Provide a valid video path with '--video input.mp4' or set VIDEO_INPUT env var.")

if __name__ == "__main__":
    asyncio.run(main())
