"""
ScoutingData v5.0 - Overlays de Visualización (Optimizados)

OPT-4: get_pitch_field_points() — pre-calcula puntos de cancha una sola vez.
OPT-6: Renders in-place (sin .copy() en cada función).
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from homography.mappings import KP_MAP
from homography.estimator import filter_keypoints_by_pitch_side
from config.field_points import FIELD_COORDS
from config.field_dimensions import FieldDimensions

# ── OPT-4: Caché de puntos de campo ──────────────────────────────────────────
_FIELD_POINTS_CACHE: dict = {}


def get_pitch_field_points(dims: FieldDimensions = FieldDimensions()) -> List[np.ndarray]:
    """
    Calcula UNA VEZ los segmentos de líneas de la cancha en coordenadas de campo (metros).
    Retorna lista de arrays float32 para cv2.perspectiveTransform.
    Solo se recalcula si cambian las dimensiones del campo.
    """
    key = (dims.length, dims.width)
    if key in _FIELD_POINTS_CACHE:
        return _FIELD_POINTS_CACHE[key]

    segments: List[np.ndarray] = []

    def seg(*pts):
        segments.append(np.array(pts, dtype=np.float32).reshape(-1, 1, 2))

    L, W = dims.length, dims.width
    PBL  = dims.penalty_box_length
    PBW  = dims.penalty_box_width
    GBL  = dims.goal_box_length
    GBW  = dims.goal_box_width
    PSD  = dims.penalty_spot_distance
    CR   = dims.centre_circle_radius

    # Borde exterior
    seg((0,0),(L,0),(L,W),(0,W),(0,0))
    # Línea de medio campo
    seg((L/2,0),(L/2,W))
    # Círculo central (36 puntos → triángulos suaves, más rápido que 360)
    angles = np.linspace(0, 2*np.pi, 73)
    circle_pts = np.column_stack([
        L/2 + CR*np.cos(angles),
        W/2 + CR*np.sin(angles),
    ]).astype(np.float32).reshape(-1,1,2)
    segments.append(circle_pts)
    # Áreas de penalti
    seg((0, W/2-PBW/2),(PBL,W/2-PBW/2),(PBL,W/2+PBW/2),(0,W/2+PBW/2))
    seg((L, W/2-PBW/2),(L-PBL,W/2-PBW/2),(L-PBL,W/2+PBW/2),(L,W/2+PBW/2))
    # Áreas de meta
    seg((0,W/2-GBW/2),(GBL,W/2-GBW/2),(GBL,W/2+GBW/2),(0,W/2+GBW/2))
    seg((L,W/2-GBW/2),(L-GBL,W/2-GBW/2),(L-GBL,W/2+GBW/2),(L,W/2+GBW/2))
    # Arco de penalti izquierdo
    left_arc_a = np.deg2rad(np.arange(-53, 54))
    left_arc   = np.column_stack([
        PSD + CR*np.cos(left_arc_a),
        W/2 + CR*np.sin(left_arc_a),
    ]).astype(np.float32).reshape(-1,1,2)
    segments.append(left_arc)
    # Arco de penalti derecho
    right_arc_a = np.deg2rad(np.arange(127, 234))
    right_arc   = np.column_stack([
        L-PSD + CR*np.cos(right_arc_a),
        W/2   + CR*np.sin(right_arc_a),
    ]).astype(np.float32).reshape(-1,1,2)
    segments.append(right_arc)

    _FIELD_POINTS_CACHE[key] = segments
    return segments


def _project_segment_clipped(
    pts_segment: np.ndarray,
    H_inv: np.ndarray,
    w_clip: float = 0.01,
    max_coord: float = 50000.0,
) -> List[np.ndarray]:
    """
    Proyecta un segmento de cancha aplicando clipping en el plano w > w_clip.
    Evita que puntos detrás de la cámara (w <= 0) se inviertan y generen líneas cruzadas en pantalla.
    """
    pts = pts_segment.reshape(-1, 2)
    N = len(pts)
    if N < 2:
        return []

    homo = np.column_stack([pts, np.ones(N, dtype=np.float32)])
    cam = homo @ H_inv.T

    result_polylines: List[np.ndarray] = []
    current_chain: List[np.ndarray] = []

    for i in range(N - 1):
        p1 = cam[i]
        p2 = cam[i + 1]
        w1, w2 = p1[2], p2[2]

        if w1 > w_clip and w2 > w_clip:
            xy1 = p1[:2] / w1
            xy2 = p2[:2] / w2
            if abs(xy1[0]) < max_coord and abs(xy1[1]) < max_coord and abs(xy2[0]) < max_coord and abs(xy2[1]) < max_coord:
                if not current_chain:
                    current_chain.append(xy1)
                current_chain.append(xy2)
            else:
                if len(current_chain) >= 2:
                    result_polylines.append(np.array(current_chain, dtype=np.int32).reshape(-1, 1, 2))
                current_chain = []
        elif w1 > w_clip and w2 <= w_clip:
            t = (w_clip - w1) / (w2 - w1) if abs(w2 - w1) > 1e-6 else 0.5
            p_cut = p1 + t * (p2 - p1)
            xy1 = p1[:2] / w1
            xy_cut = p_cut[:2] / w_clip
            if abs(xy1[0]) < max_coord and abs(xy1[1]) < max_coord and abs(xy_cut[0]) < max_coord and abs(xy_cut[1]) < max_coord:
                if not current_chain:
                    current_chain.append(xy1)
                current_chain.append(xy_cut)
            if len(current_chain) >= 2:
                result_polylines.append(np.array(current_chain, dtype=np.int32).reshape(-1, 1, 2))
            current_chain = []
        elif w1 <= w_clip and w2 > w_clip:
            t = (w_clip - w1) / (w2 - w1) if abs(w2 - w1) > 1e-6 else 0.5
            p_cut = p1 + t * (p2 - p1)
            xy_cut = p_cut[:2] / w_clip
            xy2 = p2[:2] / w2
            if len(current_chain) >= 2:
                result_polylines.append(np.array(current_chain, dtype=np.int32).reshape(-1, 1, 2))
            current_chain = []
            if abs(xy_cut[0]) < max_coord and abs(xy_cut[1]) < max_coord and abs(xy2[0]) < max_coord and abs(xy2[1]) < max_coord:
                current_chain = [xy_cut, xy2]
        else:
            if len(current_chain) >= 2:
                result_polylines.append(np.array(current_chain, dtype=np.int32).reshape(-1, 1, 2))
            current_chain = []

    if len(current_chain) >= 2:
        result_polylines.append(np.array(current_chain, dtype=np.int32).reshape(-1, 1, 2))

    return result_polylines


def draw_pitch_lines_on_image(
    img: np.ndarray,
    H: np.ndarray,
    dims: FieldDimensions = FieldDimensions(),
    precomputed_segments: Optional[List[np.ndarray]] = None,
    color: tuple = (0, 255, 255),
    thickness: int = 2,
    w_clip: float = 0.01,
) -> np.ndarray:
    """
    Proyecta líneas de la cancha sobre la imagen de cámara usando H_inv.
    OPT-4: Si se pasan `precomputed_segments`, evita recalcular la geometría.
    OPT-6: Opera in-place sobre `img` (sin crear una copia).
    Incluye clipping en el plano w > w_clip para evitar artefactos que cruzan la pantalla.
    """
    ret, H_inv = cv2.invert(H)
    if not ret or H_inv is None:
        return img
    segs = precomputed_segments if precomputed_segments is not None else get_pitch_field_points(dims)
    for pts in segs:
        polylines = _project_segment_clipped(pts, H_inv, w_clip=w_clip)
        for poly in polylines:
            cv2.polylines(img, [poly], False, color, thickness, cv2.LINE_AA)
    return img


def draw_calibration_overlay(
    img: np.ndarray,
    result,
    mask: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5,
    draw_outliers: bool = False,
    enable_side_filter: bool = True,
    field_coverage_ratio: Optional[float] = None,
    hull_area_m2: Optional[float] = None,
    H: Optional[np.ndarray] = None,
    draw_residuals: bool = False,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Dibuja keypoints detectados e instrumentación de diagnóstico.
    Si draw_residuals=True y se provee H:
      - VERDE = keypoint detectado por YOLO (u, v)
      - ROJO  = keypoint reproyectado desde terreno vía H^-1 (u', v')
      - Línea residual amarilla con vector de error y etiqueta métrica (m y px).
    OPT-6: in-place.
    """
    badge_x = 20
    badge_y = 20
    badge_h = 32
    h_img, w_img = img.shape[:2]

    # ── Badge 1: Estado de Homografía (H: VALID / H: LOST) ─────────────────
    h_is_valid = (H is not None)
    h_status_str = "H: VALID" if h_is_valid else "H: LOST"
    (htw, hth), _ = cv2.getTextSize(h_status_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    hbx, hby = badge_x, badge_y
    hbw, hbh = htw + 32, badge_h

    if hby + hbh < h_img and hbx + hbw < w_img:
        sub = img[hby:hby+hbh, hbx:hbx+hbw]
        bg = np.zeros_like(sub)
        cv2.addWeighted(bg, 0.65, sub, 0.35, 0, sub)
        img[hby:hby+hbh, hbx:hbx+hbw] = sub

        h_dot = (0, 255, 128) if h_is_valid else (0, 0, 255)
        h_border = (0, 180, 80) if h_is_valid else (0, 0, 220)
        cv2.rectangle(img, (hbx, hby), (hbx+hbw, hby+hbh), h_border, 1, cv2.LINE_AA)
        cv2.circle(img, (hbx + 12, hby + hbh//2), 4, h_dot, -1, cv2.LINE_AA)
        cv2.putText(img, h_status_str, (hbx + 22, hby + hbh - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        badge_x = hbx + hbw + 10

    # ── Badge 2: Cobertura de Terreno (Field Coverage Ratio) ─────────────
    if field_coverage_ratio is not None and field_coverage_ratio > 0 and h_is_valid:
        pct_val = field_coverage_ratio * 100.0
        pct_str = f"Field Coverage: {pct_val:.1f}%"
        if hull_area_m2 is not None and hull_area_m2 > 0:
            pct_str += f" ({int(round(hull_area_m2))} m2)"

        (tw, th), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        bx, by = badge_x, badge_y
        bw, bh = tw + 32, th + 16
        badge_h = bh

        if by + bh < h_img and bx + bw < w_img:
            sub = img[by:by+bh, bx:bx+bw]
            bg = np.zeros_like(sub)
            cv2.addWeighted(bg, 0.65, sub, 0.35, 0, sub)
            img[by:by+bh, bx:bx+bw] = sub

            dot_color = (0, 255, 128) if pct_val >= 10.0 else ((0, 220, 255) if pct_val >= 5.0 else (0, 100, 255))
            cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (60, 60, 60), 1, cv2.LINE_AA)
            cv2.circle(img, (bx + 12, by + bh//2), 4, dot_color, -1, cv2.LINE_AA)
            cv2.putText(img, pct_str, (bx + 22, by + bh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            badge_x = bx + bw + 10

    # ── Badge 3: Número de Frame (al costado de badges de estado) ─────────
    if frame_idx is not None:
        frame_str = f"Frame: {frame_idx}"
        (ftw, fth), _ = cv2.getTextSize(frame_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        fbx, fby = badge_x, badge_y
        fbw, fbh = ftw + 32, badge_h

        if fby + fbh < h_img and fbx + fbw < w_img:
            sub = img[fby:fby+fbh, fbx:fbx+fbw]
            bg = np.zeros_like(sub)
            cv2.addWeighted(bg, 0.65, sub, 0.35, 0, sub)
            img[fby:fby+fbh, fbx:fbx+fbw] = sub

            cv2.rectangle(img, (fbx, fby), (fbx+fbw, fby+fbh), (60, 60, 60), 1, cv2.LINE_AA)
            cv2.circle(img, (fbx + 12, fby + fbh//2), 4, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(img, frame_str, (fbx + 22, fby + fbh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


    if (
        result is None
        or result.keypoints is None
        or len(result.keypoints) == 0
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        return img
    raw_xy = result.keypoints.xy[0]
    kps_xy = raw_xy.cpu().numpy() if hasattr(raw_xy, "cpu") else np.asarray(raw_xy)
    raw_conf = result.keypoints.conf[0]
    kps_conf = raw_conf.cpu().numpy() if hasattr(raw_conf, "cpu") else np.asarray(raw_conf)
    if len(kps_xy) == 0:
        return img

    if enable_side_filter:
        valid_candidates, _, _ = filter_keypoints_by_pitch_side(
            kps_xy, kps_conf, conf_threshold=conf_threshold
        )
    else:
        valid_candidates = []
        for idx, (xy, conf) in enumerate(zip(kps_xy, kps_conf)):
            if conf < conf_threshold or idx not in KP_MAP:
                continue
            lbl = KP_MAP[idx]
            if lbl not in FIELD_COORDS:
                continue
            valid_candidates.append((idx, xy, float(conf), lbl))

    H_inv = None
    if draw_residuals and H is not None:
        try:
            H_inv = np.linalg.inv(H)
        except Exception:
            H_inv = None

    for used, (idx, xy, conf, label) in enumerate(valid_candidates):
        is_inlier = True
        if mask is not None and used < len(mask):
            is_inlier = bool(mask[used][0])
        elif mask is not None or H is None:
            is_inlier = False

        if not is_inlier and not draw_outliers and not draw_residuals:
            continue

        x_det, y_det = int(round(xy[0])), int(round(xy[1]))

        if draw_residuals and H is not None and H_inv is not None:
            # 1. Punto detectado por YOLO (VERDE)
            cv2.circle(img, (x_det, y_det), 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (x_det, y_det), 8, (255, 255, 255), 1, cv2.LINE_AA)

            # 2. Punto teórico reproyectado desde terreno: field_point -> H^-1 -> image_point (ROJO)
            fld_pt = np.array(FIELD_COORDS[label], dtype=np.float32).reshape(1, 1, 2)
            proj_img = cv2.perspectiveTransform(fld_pt, H_inv)[0, 0]
            x_rep, y_rep = int(round(proj_img[0])), int(round(proj_img[1]))
            cv2.circle(img, (x_rep, y_rep), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (x_rep, y_rep), 8, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Vector residual que conecta detectado con reproyectado
            line_color = (0, 255, 255) if is_inlier else (0, 100, 255)
            cv2.line(img, (x_det, y_det), (x_rep, y_rep), line_color, 2, cv2.LINE_AA)

            # 4. Error euclídeo en metros de campo y en píxeles
            proj_fld = cv2.perspectiveTransform(np.array(xy, dtype=np.float32).reshape(1, 1, 2), H)[0, 0]
            err_m = float(np.linalg.norm(proj_fld - FIELD_COORDS[label]))
            err_px = float(np.linalg.norm(proj_img - xy))

            status_tag = "IN" if is_inlier else "OUT"
            lbl_text = f"KP{label}: {err_m:.2f}m ({err_px:.0f}px) [{status_tag}]"
            cv2.putText(img, lbl_text, (x_det + 10, y_det - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, lbl_text, (x_det + 10, y_det - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            color = (0, 255, 0) if is_inlier else (0, 0, 255)
            cv2.circle(img, (x_det, y_det), 8, color, -1)
            cv2.circle(img, (x_det, y_det), 9, (255, 255, 255), 1)
            text = f"Y:{idx} R:{label} ({conf:.2f})"
            cv2.putText(img, text, (x_det+12, y_det-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (x_det+12, y_det-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    return img


def draw_reprojection_residuals(
    img: np.ndarray,
    H: np.ndarray,
    result,
    mask: Optional[np.ndarray] = None,
    conf_threshold: float = 0.35,
    enable_side_filter: bool = True,
    field_coverage_ratio: Optional[float] = None,
    hull_area_m2: Optional[float] = None,
) -> np.ndarray:
    """
    Visualización dedicada de depuración:
    VERDE = keypoint detectado por YOLO
    ROJO  = keypoint reproyectado usando: field_point -> H^-1 -> image_point
    Líneas de error residual entre cada par de puntos con métricas cuantitativas.
    """
    return draw_calibration_overlay(
        img=img,
        result=result,
        mask=mask,
        conf_threshold=conf_threshold,
        draw_outliers=True,
        enable_side_filter=enable_side_filter,
        field_coverage_ratio=field_coverage_ratio,
        hull_area_m2=hull_area_m2,
        H=H,
        draw_residuals=True,
    )


def draw_player_detections(
    img: np.ndarray,
    player_result=None,
    player_detections: Optional[List[dict]] = None,
    conf_threshold: float = 0.3,
    ellipse_color: tuple = (255, 0, 255),
    text_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Dibuja jugadores detectados. OPT-6: in-place.
    Si se suministra `player_detections` enriquecido, utiliza los colores de equipo y rol calculados.
    """
    if player_detections is not None:
        for p in player_detections:
            conf = p.get("confidence", 1.0)
            if conf < conf_threshold:
                continue
            bbox = p.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            xc = int((x1 + x2) / 2)
            yb = int(y2)
            bw = int(x2 - x1)
            aw = max(8, int(bw / 2.5))
            ah = max(4, int(aw / 2.0))

            # Color dinámico de equipo (BGR)
            team_col = p.get("team_color", {})
            p_color = tuple(team_col.get("bgr", ellipse_color)) if isinstance(team_col, dict) else ellipse_color

            cv2.ellipse(img, (xc, yb), (aw, ah), 0, 0, 360, p_color, 2, cv2.LINE_AA)
            cv2.circle(img, (xc, yb), 3, (0, 255, 255), -1, cv2.LINE_AA)

            role = p.get("role", "player")
            p_id = p.get("player_id", "?")
            t_id = p.get("team_id", 0)
            if role == "goalkeeper":
                label = f"GK:{p_id} (T{t_id})"
            elif role == "referee":
                label = f"REF:{p_id}"
            else:
                label = f"ID:{p_id} (T{t_id})"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (xc - tw // 2 - 4, yb + 6), (xc + tw // 2 + 4, yb + 6 + th + 6), (0, 0, 0), -1)
            cv2.rectangle(img, (xc - tw // 2 - 4, yb + 6), (xc + tw // 2 + 4, yb + 6 + th + 6), p_color, 1)
            cv2.putText(img, label, (xc - tw // 2, yb + 6 + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)
        return img

    if player_result is None or player_result.boxes is None or len(player_result.boxes) == 0:
        return img
    boxes     = player_result.boxes.xyxy.cpu().numpy()
    confs     = player_result.boxes.conf.cpu().numpy()
    track_ids = (
        player_result.boxes.id.cpu().numpy().astype(int)
        if player_result.boxes.id is not None else None
    )
    for idx, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < conf_threshold:
            continue
        player_id = track_ids[idx] if track_ids is not None else idx
        x1, y1, x2, y2 = box
        xc = int((x1+x2)/2)
        yb = int(y2)
        bw = int(x2-x1)
        aw = max(8, int(bw/2.5))
        ah = max(4, int(aw/2.0))
        cv2.ellipse(img, (xc, yb), (aw, ah), 0, 0, 360, ellipse_color, 2, cv2.LINE_AA)
        cv2.circle(img, (xc, yb), 3, (0, 255, 255), -1, cv2.LINE_AA)
        label = f"ID:{player_id} ({int(conf*100)}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (xc-tw//2-4, yb+6), (xc+tw//2+4, yb+6+th+6), (0,0,0), -1)
        cv2.rectangle(img, (xc-tw//2-4, yb+6), (xc+tw//2+4, yb+6+th+6), ellipse_color, 1)
        cv2.putText(img, label, (xc-tw//2, yb+6+th+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    return img



def draw_ball_detections(
    img: np.ndarray,
    ball_detection: Optional[dict] = None,
    ball_history: Optional[list] = None,
    ball_color: tuple = (0, 255, 255),
    trail_color: tuple = (0, 200, 255),
) -> np.ndarray:
    """Dibuja balón y trayectoria. OPT-6: in-place."""
    if ball_history and len(ball_history) > 1:
        valid = [p for p in ball_history if p is not None]
        for i in range(1, len(valid)):
            pt1 = (int(valid[i-1][0]), int(valid[i-1][1]))
            pt2 = (int(valid[i][0]),   int(valid[i][1]))
            alpha = float(i) / float(len(valid))
            cv2.line(img, pt1, pt2, trail_color, max(1, int(alpha*3)), cv2.LINE_AA)
    if ball_detection is not None:
        px = int(ball_detection["pixel_x"])
        py = int(ball_detection["pixel_y"])
        conf = ball_detection["confidence"]
        is_interp = ball_detection.get("interpolated", False)
        color  = (0, 165, 255) if is_interp else ball_color
        radius = 8
        cv2.circle(img, (px, py), radius+4, (0,0,0), 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), 2, (255,255,255), -1, cv2.LINE_AA)
        tag   = " (est)" if is_interp else ""
        label = f"Balon{tag} {int(conf*100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (px-tw//2-3, py-radius-th-8), (px+tw//2+3, py-radius-2), (0,0,0), -1)
        cv2.rectangle(img, (px-tw//2-3, py-radius-th-8), (px+tw//2+3, py-radius-2), color, 1)
        cv2.putText(img, label, (px-tw//2, py-radius-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    return img
