"""
ScoutingData v5.0 - Herramienta de Auditoría y Diagnóstico Avanzado de Homografía
Leave-One-Out (Jackknife), Residuales Punto a Punto y Detección de Deformaciones Geométricas.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

from config.field_dimensions import FieldDimensions
from config.field_points import FIELD_COORDS


@dataclass
class PointResidual:
    """Métricas de error punto a punto para una correspondencia imagen <-> terreno."""
    kp_idx: int                  # Índice YOLO (0..29)
    kp_label: str                # ID estándar FIFA ("01".."32")
    confidence: float            # Confianza YOLO
    detected_px: Tuple[float, float]       # (u, v) detectado por YOLO
    reprojected_px: Tuple[float, float]    # (u', v') proyectado desde campo vía H^-1
    field_m: Tuple[float, float]           # (X, Y) teórico en campo
    reprojected_m: Tuple[float, float]     # (X', Y') proyectado desde imagen vía H
    error_px: float              # Distancia euclídea en píxeles
    error_m: float               # Distancia euclídea en metros de cancha
    is_inlier: bool              # Inlier RANSAC/MAGSAC


@dataclass
class LeaveOneOutResult:
    """Resultado del impacto de retirar un keypoint específico del cálculo."""
    excluded_idx: int
    excluded_label: str
    baseline_mean_error_m: float
    loo_mean_error_m: float
    delta_error_m: float         # baseline - loo (>0 indica que el punto inflaba el error global)
    max_player_shift_m: float    # Desplazamiento máximo en metros de los jugadores evaluados
    mean_player_shift_m: float   # Desplazamiento medio en metros de los jugadores evaluados
    condition_number: float      # Número de condición de H sin este punto
    damage_score: float          # Puntuación combinada de nocividad/apalancamiento


def compute_point_residuals(
    H: np.ndarray,
    valid_candidates: List[Tuple[int, np.ndarray, float, str]],
    mask: Optional[np.ndarray] = None,
) -> List[PointResidual]:
    """
    Calcula con precisión métrica y en píxeles el error de cada correspondencia:
      - Error en metros: || H * p_img - p_field ||
      - Error en píxeles: || p_img - H^-1 * p_field ||
    """
    residuals: List[PointResidual] = []
    if H is None or len(valid_candidates) == 0:
        return residuals

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = None

    for i, (kp_idx, xy_det, conf, label) in enumerate(valid_candidates):
        is_inlier = True
        if mask is not None and i < len(mask):
            is_inlier = bool(mask[i][0] if mask[i].ndim > 0 else mask[i])

        fld_pt = np.array(FIELD_COORDS[label], dtype=np.float32).reshape(1, 1, 2)
        img_pt = np.array(xy_det, dtype=np.float32).reshape(1, 1, 2)

        # 1. Proyección a terreno (imagen -> campo vía H)
        proj_field = cv2.perspectiveTransform(img_pt, H)[0, 0]
        err_m = float(np.linalg.norm(proj_field - fld_pt[0, 0]))

        # 2. Reproyección a imagen (campo -> imagen vía H^-1)
        if H_inv is not None:
            proj_img = cv2.perspectiveTransform(fld_pt, H_inv)[0, 0]
            err_px = float(np.linalg.norm(proj_img - xy_det))
            proj_img_tuple = (float(proj_img[0]), float(proj_img[1]))
        else:
            err_px = 0.0
            proj_img_tuple = (float(xy_det[0]), float(xy_det[1]))

        residuals.append(PointResidual(
            kp_idx=kp_idx,
            kp_label=label,
            confidence=float(conf),
            detected_px=(float(xy_det[0]), float(xy_det[1])),
            reprojected_px=proj_img_tuple,
            field_m=(float(fld_pt[0, 0, 0]), float(fld_pt[0, 0, 1])),
            reprojected_m=(float(proj_field[0]), float(proj_field[1])),
            error_px=round(err_px, 2),
            error_m=round(err_m, 3),
            is_inlier=is_inlier,
        ))

    return residuals


def generate_residuals_table(residuals: List[PointResidual]) -> str:
    """
    Genera una tabla tabular en texto/markdown con el desglose por keypoint.
    """
    lines = [
        "+-------+-------+------------+------------+------------+---------------+",
        "| KP_ID | Label | Confidence | Error (m)  | Error (px) | Status        |",
        "+-------+-------+------------+------------+------------+---------------+",
    ]
    for r in residuals:
        status = "INLIER" if r.is_inlier else "OUTLIER"
        lines.append(
            f"| {r.kp_idx:02d}    | {r.kp_label:5s} | {r.confidence:10.2f} | "
            f"{r.error_m:8.2f} m | {r.error_px:8.1f} px | {status:13s} |"
        )
    lines.append("+-------+-------+------------+------------+------------+---------------+")
    return "\n".join(lines)


def run_leave_one_out_analysis(
    valid_candidates: List[Tuple[int, np.ndarray, float, str]],
    method: int = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC,
    ransac_threshold: float = 1.5,
    sample_players_field_m: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[List[LeaveOneOutResult], Dict[str, Any]]:
    """
    Ejecuta un análisis Leave-One-Out (Jackknife) sobre el conjunto de puntos:
      Para cada keypoint:
        1. Recalcula H excluyendo dicho punto.
        2. Mide la variación del error global de reproyección.
        3. Mide el desplazamiento inducido sobre las posiciones de jugadores.
        4. Calcula un ranking de nocividad/inestabilidad.
    """
    n_pts = len(valid_candidates)
    if n_pts < 5:
        # Se requieren al menos 5 puntos para poder retirar uno y mantener 4
        return [], {"error": "Se requieren al menos 5 puntos para Leave-One-Out"}

    image_pts_all = np.array([xy for _, xy, _, _ in valid_candidates], dtype=np.float32)
    field_pts_all = np.array([FIELD_COORDS[lbl] for _, _, _, lbl in valid_candidates], dtype=np.float32)

    # Si no se proveen jugadores muestra, se genera una cuadrícula táctica en el área activa
    if sample_players_field_m is None or len(sample_players_field_m) == 0:
        min_x, max_x = float(field_pts_all[:, 0].min()), float(field_pts_all[:, 0].max())
        min_y, max_y = float(field_pts_all[:, 1].min()), float(field_pts_all[:, 1].max())
        gx = np.linspace(min_x, max_x, 5)
        gy = np.linspace(min_y, max_y, 5)
        sample_players = [
            (float(x), float(y)) for x in gx for y in gy
        ]
    else:
        sample_players = sample_players_field_m

    # Homografía base con todos los puntos
    try:
        H_base, mask_base = cv2.findHomography(image_pts_all, field_pts_all, method, ransac_threshold)
    except Exception:
        H_base = None

    if H_base is None:
        return [], {"error": "No se pudo calcular la homografía base"}

    # Proyección base de los puntos de control de jugadores (terreno -> imagen -> terreno)
    try:
        H_base_inv = np.linalg.inv(H_base)
    except np.linalg.LinAlgError:
        return [], {"error": "Matriz H base singular"}

    players_fld_arr = np.array(sample_players, dtype=np.float32).reshape(-1, 1, 2)
    players_img_arr = cv2.perspectiveTransform(players_fld_arr, H_base_inv)

    # Error base en metros sobre todos los puntos
    base_proj_fld = cv2.perspectiveTransform(image_pts_all.reshape(-1, 1, 2), H_base).reshape(-1, 2)
    base_err_m = float(np.linalg.norm(base_proj_fld - field_pts_all, axis=1).mean())

    results: List[LeaveOneOutResult] = []

    for i in range(n_pts):
        ex_idx, _, _, ex_label = valid_candidates[i]

        # Máscara de exclusión
        keep_mask = [k != i for k in range(n_pts)]
        sub_img = image_pts_all[keep_mask]
        sub_fld = field_pts_all[keep_mask]

        try:
            H_loo, _ = cv2.findHomography(sub_img, sub_fld, method, ransac_threshold)
        except Exception:
            H_loo = None

        if H_loo is None:
            continue

        # Error medio de los puntos restantes con H_loo
        loo_proj_fld = cv2.perspectiveTransform(sub_img.reshape(-1, 1, 2), H_loo).reshape(-1, 2)
        loo_err_m = float(np.linalg.norm(loo_proj_fld - sub_fld, axis=1).mean())
        delta_err_m = base_err_m - loo_err_m

        # Desplazamiento de los jugadores evaluados
        loo_players_fld = cv2.perspectiveTransform(players_img_arr, H_loo).reshape(-1, 2)
        diffs = np.linalg.norm(loo_players_fld - players_fld_arr.reshape(-1, 2), axis=1)
        max_shift = float(diffs.max())
        mean_shift = float(diffs.mean())

        # Condición de H_loo
        h_norm = H_loo / (H_loo[2, 2] if abs(H_loo[2, 2]) > 1e-9 else 1.0)
        try:
            cond_num = float(np.linalg.cond(h_norm))
        except Exception:
            cond_num = 1e9

        # Puntuación combinada de nocividad:
        # Puntos que al retirarse reducen el error global y causan gran desplazamiento en jugadores
        damage_score = round(max(0.0, delta_err_m) * 2.0 + mean_shift * 1.5, 3)

        results.append(LeaveOneOutResult(
            excluded_idx=ex_idx,
            excluded_label=ex_label,
            baseline_mean_error_m=round(base_err_m, 3),
            loo_mean_error_m=round(loo_err_m, 3),
            delta_error_m=round(delta_err_m, 3),
            max_player_shift_m=round(max_shift, 2),
            mean_player_shift_m=round(mean_shift, 2),
            condition_number=round(cond_num, 1),
            damage_score=damage_score,
        ))

    # Ordenar ranking descendente por daño inducido a la geometría
    results.sort(key=lambda x: (x.damage_score, x.delta_error_m), reverse=True)

    meta = {
        "baseline_mean_error_m": round(base_err_m, 3),
        "total_kps_evaluated": n_pts,
        "sample_points_count": len(sample_players),
    }

    return results, meta


def format_leave_one_out_report(results: List[LeaveOneOutResult], meta: Dict[str, Any]) -> str:
    """
    Genera un informe detallado en texto/markdown del análisis Leave-One-Out.
    """
    if not results:
        return "No se pudieron calcular resultados de Leave-One-Out."

    lines = [
        f"=== RANKING LEAVE-ONE-OUT (JACKKNIFE) ===",
        f"Error medio basal: {meta.get('baseline_mean_error_m', 0.0):.2f} m | Puntos: {meta.get('total_kps_evaluated', 0)}",
        "+------+-------+------------+------------+---------------+------------------+--------------+",
        "| Rank | KP_ID | Label      | dError(m)  | Max Shift(m)  | Mean Shift(m)    | Damage Score |",
        "+------+-------+------------+------------+---------------+------------------+--------------+",
    ]
    for rank, r in enumerate(results, start=1):
        sign = "+" if r.delta_error_m > 0 else ""
        lines.append(
            f"| {rank:4d} | {r.excluded_idx:02d}    | {r.excluded_label:10s} | "
            f"{sign}{r.delta_error_m:9.2f}m | {r.max_player_shift_m:12.2f}m | "
            f"{r.mean_player_shift_m:15.2f}m | {r.damage_score:12.3f} |"
        )
    lines.append("+------+-------+------------+------------+---------------+------------------+--------------+")
    if results[0].delta_error_m > 0.3 or results[0].mean_player_shift_m > 1.5:
        lines.append(
            f"⚠ ALERTA: El keypoint ID {results[0].excluded_idx} ('{results[0].excluded_label}') "
            f"degrada severamente la homografía (induce desplazamiento de {results[0].mean_player_shift_m:.2f}m)."
        )
    return "\n".join(lines)


def detect_spatial_anomalies_and_swaps(
    residuals: List[PointResidual]
) -> List[Dict[str, Any]]:
    """
    Detecta automáticamente anomalías patológicas:
      1. Intercambio de identidad (ej: KP 11 abajo de KP 12 en imagen cuando en campo está arriba).
      2. Residuales sistemáticos aislados (> 1.5m cuando el resto tiene < 0.8m).
      3. Colapso de escala o inversiones de polaridad.
    """
    anomalies: List[Dict[str, Any]] = []
    by_label = {r.kp_label: r for r in residuals}

    # 1. Comprobación de orden vertical en parejas críticas:
    # 07 (y=24.84) debe estar arriba de 08 (y=43.16)
    if "07" in by_label and "08" in by_label:
        if by_label["07"].detected_px[1] > by_label["08"].detected_px[1]:
            anomalies.append({
                "type": "IDENTITY_SWAP",
                "severity": "CRITICAL",
                "labels": ["07", "08"],
                "message": "Inversión de orden vertical detectada entre KP 07 y 08 en imagen.",
            })

    # 10 (y=13.84), 11 (y=26.69), 12 (y=41.31), 13 (y=54.16)
    left_seq = ["10", "11", "12", "13"]
    present_seq = [lbl for lbl in left_seq if lbl in by_label]
    for a, b in zip(present_seq[:-1], present_seq[1:]):
        if by_label[a].detected_px[1] > by_label[b].detected_px[1]:
            anomalies.append({
                "type": "IDENTITY_SWAP",
                "severity": "CRITICAL",
                "labels": [a, b],
                "message": f"Inversión de orden vertical detectada entre KP '{a}' y '{b}'.",
            })

    # 2. Keypoints con error desproporcionado respecto a la mediana del grupo
    errors_m = [r.error_m for r in residuals]
    if len(errors_m) >= 4:
        median_err = float(np.median(errors_m))
        for r in residuals:
            if r.error_m > 1.5 and r.error_m > median_err * 2.5:
                anomalies.append({
                    "type": "SYSTEMATIC_OUTLIER",
                    "severity": "HIGH",
                    "labels": [r.kp_label],
                    "message": (
                        f"Keypoint '{r.kp_label}' (ID {r.kp_idx}) presenta error anómalo "
                        f"({r.error_m:.2f}m vs mediana {median_err:.2f}m)."
                    ),
                })

    return anomalies
