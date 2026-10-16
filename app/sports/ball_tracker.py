"""
ScoutingData v5.4 - Ball Tracker con Filtro de Kalman 2D y Gating de Mahalanobis

Mejoras Fase 1:
  - Filtro de Kalman 2D con vector de estado cinemático [x, y, vx, vy]^T.
  - Modelo de fricción/amortiguamiento físico (damping=0.98) para trayectorias realistas.
  - Gating estadístico con Distancia de Mahalanobis (chi-cuadrado) para rechazo de outliers.
  - Selección multicandidato óptima basada en probabilidad combinada (YOLO conf + Mahalanobis).
  - Validación cinemática métrica en espacio de campo (pitch space <= 38 m/s = 137 km/h).
  - Interpolación inercial suave durante oclusiones de hasta N frames.
"""

from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import logging

try:
    from homography.transforms import pixel_to_field
except ImportError:
    try:
        from app.homography.transforms import pixel_to_field
    except ImportError:
        pixel_to_field = None

logger = logging.getLogger("BallTracker")


class BallKalmanFilter:
    """
    Filtro de Kalman Lineal 2D para seguimiento balístico y cinemático del balón.
    Estado x: [pos_x, pos_y, vel_x, vel_y]^T
    Medición z: [pos_x, pos_y]^T
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 2.0,
        measurement_noise_std: float = 3.5,
        damping: float = 0.98,
    ):
        self.dt = dt
        self.damping = damping

        # Matriz de transición de estado F (inercia + fricción)
        self.F = np.array([
            [1.0, 0.0, dt,       0.0],
            [0.0, 1.0, 0.0,      dt],
            [0.0, 0.0, damping,  0.0],
            [0.0, 0.0, 0.0,      damping],
        ], dtype=np.float32)

        # Matriz de observación H (observamos posición en píxeles)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # Matriz de covarianza de ruido de proceso Q
        q = float(process_noise_std ** 2)
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        self.Q = np.array([
            [dt4 * q, 0.0,     dt3 * q, 0.0],
            [0.0,     dt4 * q, 0.0,     dt3 * q],
            [dt3 * q, 0.0,     dt2 * q, 0.0],
            [0.0,     dt3 * q, 0.0,     dt2 * q],
        ], dtype=np.float32)

        # Matriz base de covarianza de ruido de medición R
        self.base_R = (float(measurement_noise_std) ** 2) * np.eye(2, dtype=np.float32)

        self.state = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.is_initialized = False

    def initialize(self, x: float, y: float):
        """Inicializa el filtro en una coordenada inicial (velocidad en cero)."""
        self.state = np.array([[float(x)], [float(y)], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 15.0
        self.is_initialized = True

    def predict(self) -> Tuple[float, float]:
        """Predice el estado en el siguiente frame: x_hat = F * x, P = F * P * F^T + Q."""
        if not self.is_initialized:
            return 0.0, 0.0
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.state[0, 0]), float(self.state[1, 0])

    def compute_mahalanobis_distance(self, z_x: float, z_y: float, confidence: float = 1.0) -> float:
        """
        Calcula la distancia de Mahalanobis d_M = sqrt(y^T * S^-1 * y)
        entre la medición candidata z y la predicción del filtro.
        """
        if not self.is_initialized:
            return 0.0
        z = np.array([[float(z_x)], [float(z_y)]], dtype=np.float32)
        y = z - (self.H @ self.state)  # Vector de innovación

        conf_weight = max(0.1, min(1.0, float(confidence)))
        R = self.base_R / conf_weight
        S = self.H @ self.P @ self.H.T + R  # Covarianza de innovación (2x2)

        try:
            S_inv = np.linalg.inv(S)
            d2 = float((y.T @ S_inv @ y)[0, 0])
            return float(np.sqrt(max(0.0, d2)))
        except np.linalg.LinAlgError:
            return float(np.hypot(y[0, 0], y[1, 0]) / 10.0)

    def update(self, z_x: float, z_y: float, confidence: float = 1.0):
        """Corrige el estado con la medición observada z."""
        if not self.is_initialized:
            self.initialize(z_x, z_y)
            return

        z = np.array([[float(z_x)], [float(z_y)]], dtype=np.float32)
        y = z - (self.H @ self.state)

        conf_weight = max(0.1, min(1.0, float(confidence)))
        R = self.base_R / conf_weight
        S = self.H @ self.P @ self.H.T + R

        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ self.H.T @ S_inv  # Ganancia de Kalman (4x2)
            self.state = self.state + K @ y
            I = np.eye(4, dtype=np.float32)
            self.P = (I - K @ self.H) @ self.P
        except np.linalg.LinAlgError:
            self.state[0, 0] = float(z_x)
            self.state[1, 0] = float(z_y)

    def get_position(self) -> Tuple[float, float]:
        return float(self.state[0, 0]), float(self.state[1, 0])

    def get_velocity(self) -> Tuple[float, float]:
        return float(self.state[2, 0]), float(self.state[3, 0])


class BallTracker:
    """
    Rastreador de balón con Filtro de Kalman 2D, Gating de Mahalanobis y
    selección multicandidato probabilística (Fase 1).
    """

    def __init__(
        self,
        max_history: int = 15,
        max_missing_frames: int = 10,
        max_distance_jump: float = 220.0,
        mahalanobis_gate: float = 4.0,       # Umbral Chi-cuadrado (~99.5% confianza)
        max_metric_speed_ms: float = 38.0,   # Máxima velocidad física (137 km/h)
    ):
        self.max_history = max_history
        self.max_missing_frames = max_missing_frames
        self.max_distance_jump = max_distance_jump
        self.mahalanobis_gate = mahalanobis_gate
        self.max_metric_speed_ms = max_metric_speed_ms

        self.kf = BallKalmanFilter(dt=1.0, damping=0.98)
        self.history: List[Optional[Tuple[float, float]]] = []
        self.missing_count = 0
        self.last_known_detection: Optional[Dict] = None
        self.last_pitch_pos: Optional[Tuple[float, float]] = None

    def reset(self):
        """Reinicia el estado del rastreador."""
        self.kf = BallKalmanFilter(dt=1.0, damping=0.98)
        self.history.clear()
        self.missing_count = 0
        self.last_known_detection = None
        self.last_pitch_pos = None

    def update(
        self,
        current_detection: Union[Optional[Dict], List[Dict]],
        H: Optional[np.ndarray] = None,
        fps: float = 30.0,
    ) -> Tuple[Optional[Dict], List[Tuple[float, float]]]:
        """
        Actualiza el rastreador con la(s) detección(es) del frame actual.
        Soporta tanto un único diccionario (compatibilidad) como una lista de candidatos.
        """
        # Normalizar entrada a lista de candidatos válidos
        if current_detection is None:
            candidates: List[Dict] = []
        elif isinstance(current_detection, dict):
            candidates = [current_detection]
        elif isinstance(current_detection, list):
            candidates = [c for c in current_detection if c is not None and isinstance(c, dict)]
        else:
            candidates = []

        # ── 1. Predicción cinemática previa de Kalman ─────────────────────────
        pred_x, pred_y = self.kf.predict() if self.kf.is_initialized else (None, None)

        best_candidate: Optional[Dict] = None
        best_score = -float("inf")
        best_pitch_pt: Optional[Tuple[float, float]] = None

        # ── 2. Evaluación probabilística de candidatos ────────────────────────
        if candidates:
            if not self.kf.is_initialized:
                # Primer frame con detección: tomar el candidato con mayor confianza
                candidates_sorted = sorted(candidates, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
                best_candidate = candidates_sorted[0].copy()
                self.kf.initialize(best_candidate["pixel_x"], best_candidate["pixel_y"])
                if H is not None and pixel_to_field is not None:
                    try:
                        best_pitch_pt = pixel_to_field(best_candidate["pixel_x"], best_candidate["pixel_y"], H)
                    except Exception:
                        best_pitch_pt = None
            else:
                for c in candidates:
                    cx = float(c["pixel_x"])
                    cy = float(c["pixel_y"])
                    conf = float(c.get("confidence", 1.0))

                    # Distancia de Mahalanobis y distancia euclidiana a la predicción
                    d_m = self.kf.compute_mahalanobis_distance(cx, cy, conf)
                    d_euc = float(np.hypot(cx - pred_x, cy - pred_y))

                    # Gating estricto: descartar saltos bruscos imposibles
                    if d_euc > self.max_distance_jump or d_m > self.mahalanobis_gate:
                        continue

                    # Validación métrica física con Homografía (si está disponible)
                    pitch_pt = None
                    metric_penalty = 0.0
                    if H is not None and pixel_to_field is not None:
                        try:
                            fx, fy = pixel_to_field(cx, cy, H)
                            pitch_pt = (fx, fy)

                            # Verificar límites razonables del campo de fútbol (-10m a 115m, -10m a 78m)
                            if not (-10.0 <= fx <= 115.0 and -10.0 <= fy <= 78.0):
                                metric_penalty += 0.40  # Candidato fuera del estadio/campo

                            # Verificar velocidad métrica máxima permitida
                            if self.last_pitch_pos is not None:
                                pfx, pfy = self.last_pitch_pos
                                delta_dist_m = float(np.hypot(fx - pfx, fy - pfy))
                                delta_time_s = max(1.0 / fps, 0.001)
                                metric_speed = delta_dist_m / delta_time_s
                                if metric_speed > self.max_metric_speed_ms:
                                    # Físicamente imposible para un balón de fútbol (> 137 km/h)
                                    continue
                        except Exception:
                            pitch_pt = None

                    # Score compuesto: conf de YOLO penalizada por distancia de Mahalanobis
                    score = conf - (0.12 * d_m) - metric_penalty
                    if score > best_score:
                        best_score = score
                        best_candidate = c.copy()
                        best_pitch_pt = pitch_pt

        # ── 3. Manejo de detección válida ────────────────────────────────────
        if best_candidate is not None:
            self.kf.update(best_candidate["pixel_x"], best_candidate["pixel_y"], best_candidate.get("confidence", 1.0))
            if self.last_known_detection is not None and len(self.history) == 1:
                # Inicialización de velocidad por 2 puntos para convergencia inmediata
                vx = float(best_candidate["pixel_x"] - self.last_known_detection["pixel_x"])
                vy = float(best_candidate["pixel_y"] - self.last_known_detection["pixel_y"])
                self.kf.state[2, 0] = vx
                self.kf.state[3, 0] = vy

            self.missing_count = 0
            self.last_known_detection = best_candidate.copy()
            if best_pitch_pt is not None:
                self.last_pitch_pos = best_pitch_pt

            self.history.append((best_candidate["pixel_x"], best_candidate["pixel_y"]))
            if len(self.history) > self.max_history:
                self.history.pop(0)

            return best_candidate, [p for p in self.history if p is not None]

        # ── 4. Manejo de oclusión / pérdida temporal (Interpolación Kalman) ───
        self.missing_count += 1
        if self.missing_count <= self.max_missing_frames and self.kf.is_initialized and pred_x is not None:
            smooth_x, smooth_y = float(pred_x), float(pred_y)

            interpolated_det = (
                self.last_known_detection.copy()
                if self.last_known_detection is not None
                else {}
            )
            interpolated_det["pixel_x"] = round(smooth_x, 2)
            interpolated_det["pixel_y"] = round(smooth_y, 2)

            # Atenuación gradual de la confianza durante la oclusión
            prev_conf = float(self.last_known_detection.get("confidence", 0.6)) if self.last_known_detection else 0.5
            decay = float(0.92 ** self.missing_count)
            interpolated_det["confidence"] = round(max(0.10, prev_conf * decay), 2)
            interpolated_det["interpolated"] = True

            # Generar bounding box proyectado alrededor del centro predicho
            bw, bh = 22.0, 22.0
            interpolated_det["bbox"] = [
                round(smooth_x - bw / 2, 2),
                round(smooth_y - bh / 2, 2),
                round(smooth_x + bw / 2, 2),
                round(smooth_y + bh / 2, 2),
            ]

            self.history.append((smooth_x, smooth_y))
            if len(self.history) > self.max_history:
                self.history.pop(0)

            return interpolated_det, [p for p in self.history if p is not None]

        # Si se superó el límite de frames perdidos, re-inicializar el filtro
        if self.missing_count > self.max_missing_frames:
            self.kf.is_initialized = False

        return None, [p for p in self.history if p is not None]
