"""
ScoutingData v5.2 - Módulo de Clasificación de Equipos y Color de Camiseta
Fases 1 y 2:
  - Extracción de ROI (Torso superior: [0.15H-0.50H] x [0.15W-0.85W])
  - Filtrado de césped y piel en espacio HSV
  - Descriptores robustos en espacio CIELAB (Trimmed Median)
  - Calibración global K-Means (K=2) a nivel partido
  - Consistencia temporal y locking por track_id (Filtro por mayoría/EMA)
  - Detección contextual de Porteros y Árbitros usando coordenadas de campo
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import deque


def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    """Convierte tupla BGR a string hexadecimal #RRGGBB."""
    b, g, r = bgr
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _lab_to_bgr(lab: np.ndarray) -> Tuple[int, int, int]:
    """Convierte un vector LAB [L, a, b] a BGR uint8."""
    lab_pixel = np.uint8([[lab]])
    bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


class TeamDetector:
    """
    Detector y clasificador de equipos basado en color predominante de camiseta.
    Optimizado para rendimiento en CPU y GPU mediante operaciones vectorizadas NumPy.
    """

    def __init__(
        self,
        min_samples_to_cluster: int = 15,
        history_window: int = 35,
        lock_threshold: int = 12,
        sample_interval: int = 5,
        outlier_distance_lab: float = 38.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.min_samples_to_cluster = min_samples_to_cluster
        self.history_window = history_window
        self.lock_threshold = lock_threshold
        self.sample_interval = max(1, sample_interval)
        self.outlier_distance_lab = outlier_distance_lab
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Centroides globales de equipo en espacio LAB [L, a, b]
        # Se inicializan como None y se calibran con K-Means en los primeros frames
        self.centroids_lab: Optional[np.ndarray] = None  # shape (2, 3)
        self.team_colors_bgr: Dict[int, Tuple[int, int, int]] = {
            0: (0, 0, 255),       # Default Rojo (BGR)
            1: (255, 255, 255),   # Default Blanco (BGR)
            -1: (40, 40, 40),     # Árbitro Negro (BGR)
        }
        self.team_colors_hex: Dict[int, str] = {
            0: "#FF0000",
            1: "#FFFFFF",
            -1: "#282828",
        }

        # Buffer de muestras para calibración inicial K-Means
        self._calibration_samples: List[np.ndarray] = []
        self._is_calibrated: bool = False

        # Historial temporal por track_id: track_id -> deque de (team_id, confidence, role, bgr)
        self._track_history: Dict[int, deque] = {}
        # Estado consensuado de equipo por track_id (dinámicamente actualizable)
        self._locked_teams: Dict[int, int] = {}
        self._locked_roles: Dict[int, str] = {}
        # Caché de clasificaciones para muestreo intermitente
        self._cached_classifications: Dict[int, Dict[str, Any]] = {}

    def extract_jersey_color(
        self,
        frame_bgr: np.ndarray,
        bbox: List[float],
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
        """
        Extrae el color característico de la camiseta en el ROI del torso superior.
        Retorna (lab_vector, bgr_color) o None si el crop es inválido.
        """
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        if bw < 8 or bh < 16:
            return None

        # ROI: Torso Superior
        # Vertical: 15% a 50% de la altura total
        # Horizontal: 15% a 85% del ancho total
        ry1 = max(0, int(y1 + 0.15 * bh))
        ry2 = min(h_img, int(y1 + 0.50 * bh))
        rx1 = max(0, int(x1 + 0.15 * bw))
        rx2 = min(w_img, int(x2 - 0.15 * bw))

        if ry2 <= ry1 or rx2 <= rx1:
            return None

        crop_bgr = frame_bgr[ry1:ry2, rx1:rx2]
        if crop_bgr.size == 0:
            return None

        # 1. Filtro de Césped y Piel en HSV
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = crop_hsv[:, :, 0], crop_hsv[:, :, 1], crop_hsv[:, :, 2]

        # Máscara de Césped Verde: H [32, 88], S [35, 255], V [30, 255]
        grass_mask = (h_ch >= 32) & (h_ch <= 88) & (s_ch >= 35) & (v_ch >= 30)

        # Máscara de Piel: H [0, 25], S [30, 180], V [60, 255]
        skin_mask = (h_ch >= 0) & (h_ch <= 25) & (s_ch >= 30) & (s_ch <= 180) & (v_ch >= 60)

        # Píxeles de camiseta válidos: sin césped
        valid_mask = ~grass_mask

        # Si quedan suficientes píxeles, excluir también piel
        non_skin_count = np.count_nonzero(valid_mask & (~skin_mask))
        if non_skin_count >= 15:
            valid_mask = valid_mask & (~skin_mask)

        valid_pixels_bgr = crop_bgr[valid_mask]
        if len(valid_pixels_bgr) < 8:
            # Fallback a región central del torso sin filtro
            ch, cw = crop_bgr.shape[:2]
            valid_pixels_bgr = crop_bgr[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4].reshape(-1, 3)
            if len(valid_pixels_bgr) == 0:
                return None

        # 2. Representación en espacio CIELAB
        valid_crop_1d = valid_pixels_bgr.reshape(1, -1, 3)
        valid_lab = cv2.cvtColor(valid_crop_1d, cv2.COLOR_BGR2LAB)[0]

        # Mediana robusta en canales L*, a*, b*
        median_lab = np.median(valid_lab, axis=0).astype(np.float32)

        # Color BGR representativo para visualización
        median_bgr = np.median(valid_pixels_bgr, axis=0).astype(np.uint8)
        bgr_tuple = (int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2]))

        return median_lab, bgr_tuple

    def _calibrate_centroids(self):
        """Ajusta centroides globales K-Means (K=2) sobre las muestras recolectadas."""
        if len(self._calibration_samples) < self.min_samples_to_cluster:
            return

        samples = np.array(self._calibration_samples, dtype=np.float32)

        # Clustering K-Means K=2 en espacio LAB
        # Ponderamos más la cromaticidad (a*, b*) que la luminancia L* para robustez a sombras
        features = samples.copy()
        features[:, 0] *= 0.35  # Atenuar L* en el clustering

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        _, labels, centers_feat = cv2.kmeans(
            features, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Restaurar escala completa de L* en los centroides reales
        c0 = np.mean(samples[labels.flatten() == 0], axis=0) if np.any(labels == 0) else centers_feat[0]
        c1 = np.mean(samples[labels.flatten() == 1], axis=0) if np.any(labels == 1) else centers_feat[1]

        # Verificar separación cromática real entre centroides (canales a*, b*)
        chroma_sep = float(np.linalg.norm(c0[1:] - c1[1:]))
        if chroma_sep < 12.0 and len(self._calibration_samples) < 60:
            # Ambos centroides corresponden al mismo equipo (ej. en saque inicial), seguir recolectando
            return

        self.centroids_lab = np.array([c0, c1], dtype=np.float32)

        # Derivar colores RGB/BGR representativos para cada equipo
        bgr0 = _lab_to_bgr(c0)
        bgr1 = _lab_to_bgr(c1)

        self.team_colors_bgr[0] = bgr0
        self.team_colors_bgr[1] = bgr1
        self.team_colors_hex[0] = _bgr_to_hex(bgr0)
        self.team_colors_hex[1] = _bgr_to_hex(bgr1)

        self._is_calibrated = True

    def classify_player(
        self,
        lab_color: np.ndarray,
        bgr_color: Tuple[int, int, int],
        track_id: Optional[int] = None,
        pitch_x_m: Optional[float] = None,
        pitch_y_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Clasifica al jugador en Team 0, Team 1, Portero o Árbitro.
        Aplica consistencia temporal y bloqueo por track_id.
        """
        # Si aún estamos recolectando muestras, registrar para calibración
        if not self._is_calibrated or len(self._calibration_samples) < 40:
            self._calibration_samples.append(lab_color)
            if len(self._calibration_samples) >= self.min_samples_to_cluster:
                self._calibrate_centroids()

        # Si aún no tenemos centroides K-Means calibrados, emitir clasificación provisional con color real
        if self.centroids_lab is None:
            team_id = 0
            role = "player"
            conf = 0.50
            return {
                "team_id": team_id,
                "role": role,
                "team_color": {
                    "hex": _bgr_to_hex(bgr_color),
                    "rgb": [int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])],
                    "bgr": list(bgr_color),
                },
                "confidence": conf,
                "is_locked": False,
            }

        # Distancias ponderadas en LAB: prioriza a* y b* (cromaticidad) sobre L* (brillo)
        c0 = self.centroids_lab[0]
        c1 = self.centroids_lab[1]

        # Delta E con atenuación de luminancia: d^2 = 0.30*(dL)^2 + (da)^2 + (db)^2
        w_l = 0.30
        d0 = np.sqrt(w_l * (lab_color[0] - c0[0]) ** 2 + (lab_color[1] - c0[1]) ** 2 + (lab_color[2] - c0[2]) ** 2)
        d1 = np.sqrt(w_l * (lab_color[0] - c1[0]) ** 2 + (lab_color[1] - c1[1]) ** 2 + (lab_color[2] - c1[2]) ** 2)

        min_d = min(d0, d1)
        raw_team = 0 if d0 < d1 else 1

        # Confianza basada en Softmax inverso de distancias
        tau = 18.0
        e0 = np.exp(-d0 / tau)
        e1 = np.exp(-d1 / tau)
        conf = float(max(e0, e1) / max(1e-6, e0 + e1))

        # Detección de Outliers (Porteros / Árbitros)
        is_outlier = min_d > self.outlier_distance_lab
        role = "player"
        assigned_team = raw_team

        if is_outlier:
            # Contexto espacial de homografía (Fase 2)
            if pitch_x_m is not None and pitch_y_m is not None:
                is_near_goal_left = (pitch_x_m <= 16.5) and (abs(pitch_y_m - self.pitch_width / 2.0) <= 22.0)
                is_near_goal_right = (pitch_x_m >= self.pitch_length - 16.5) and (abs(pitch_y_m - self.pitch_width / 2.0) <= 22.0)

                if is_near_goal_left or is_near_goal_right:
                    role = "goalkeeper"
                    # Asignar al equipo del lado de campo correspondiente
                    assigned_team = 0 if is_near_goal_left else 1
                else:
                    role = "referee"
                    assigned_team = -1
            else:
                # Sin homografía: SOLO clasificar como árbitro si el color es realmente oscuro/negro acromático o flúor
                # Un color cromático (celeste, rojo, azul, verde) NUNCA debe clasificarse como árbitro negro
                chroma_dist = float(np.hypot(lab_color[1] - 128.0, lab_color[2] - 128.0))
                is_dark_referee = (lab_color[0] <= 85.0) and (chroma_dist <= 16.0)
                is_neon_referee = (lab_color[0] >= 170.0) and (lab_color[1] <= 115) and (lab_color[2] >= 150)

                if is_dark_referee or is_neon_referee:
                    role = "referee"
                    assigned_team = -1
                else:
                    role = "player"
                    assigned_team = raw_team

        # Consistencia Temporal Dinámica (Smoothing ponderado por track_id)
        if track_id is not None:
            if track_id not in self._track_history:
                self._track_history[track_id] = deque(maxlen=self.history_window)

            self._track_history[track_id].append((assigned_team, conf, role))

            # Ponderación por confianza en la ventana histórica reciente
            hist = list(self._track_history[track_id])
            if len(hist) >= 5:
                team_scores: Dict[int, float] = {0: 0.0, 1: 0.0, -1: 0.0}
                role_scores: Dict[str, float] = {}
                total_weight = 0.0
                for t_vote, c_vote, r_vote in hist:
                    team_scores[t_vote] = team_scores.get(t_vote, 0.0) + c_vote
                    role_scores[r_vote] = role_scores.get(r_vote, 0.0) + c_vote
                    total_weight += c_vote

                majority_team = max(team_scores, key=team_scores.get)
                majority_role = max(role_scores, key=role_scores.get)
                agreement = team_scores[majority_team] / max(1e-6, total_weight)

                if agreement >= 0.65:
                    assigned_team = majority_team
                    role = majority_role
                    conf = float(min(0.99, agreement))

                # Si hay suficiente consenso (> 80%) y muestras, actualizar o confirmar estado bloqueado
                if len(hist) >= self.lock_threshold and agreement >= 0.80:
                    self._locked_teams[track_id] = majority_team
                    self._locked_roles[track_id] = majority_role
            elif track_id in self._locked_teams:
                # Fallback al consenso previo si aún no hay 5 muestras en la nueva ventana
                assigned_team = self._locked_teams[track_id]
                role = self._locked_roles.get(track_id, "player")
                conf = 0.90

        # Asignar color de salida
        color_bgr = self.team_colors_bgr.get(assigned_team, bgr_color)
        color_hex = self.team_colors_hex.get(assigned_team, _bgr_to_hex(color_bgr))

        return {
            "team_id": assigned_team,
            "role": role,
            "team_color": {
                "hex": color_hex,
                "rgb": [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])],
                "bgr": list(color_bgr),
            },
            "confidence": round(float(conf), 2),
            "is_locked": track_id in self._locked_teams if track_id is not None else False,
        }

    def annotate_player_detections(
        self,
        frame_bgr: np.ndarray,
        player_detections: List[Dict],
        frame_idx: int = 0,
    ) -> List[Dict]:
        """
        Inyecta en cada dict de player_detections los campos de equipo y color.
        Implementa muestreo adaptativo por sample_interval para optimizar rendimiento.
        """
        for p in player_detections:
            bbox = p.get("bbox")
            if bbox is None:
                continue

            track_id = p.get("player_id")

            # Muestreo adaptativo: Si el jugador ya tiene historial o está bloqueado, evitar crop redundante
            is_locked = track_id is not None and track_id in self._locked_teams
            has_history = track_id is not None and len(self._track_history.get(track_id, [])) >= 5
            is_cached = track_id is not None and track_id in self._cached_classifications
            skip_sampling = is_cached and (
                (is_locked and frame_idx % 15 != 0)
                or (has_history and frame_idx % self.sample_interval != 0)
            )

            if skip_sampling:
                classification = self._cached_classifications[track_id]
            else:
                extracted = self.extract_jersey_color(frame_bgr, bbox)
                px_m = p.get("pitch_x_m")
                py_m = p.get("pitch_y_m")

                if extracted is not None:
                    lab_col, bgr_col = extracted
                    classification = self.classify_player(
                        lab_color=lab_col,
                        bgr_color=bgr_col,
                        track_id=track_id,
                        pitch_x_m=px_m,
                        pitch_y_m=py_m,
                    )
                else:
                    # Fallback a caché previa si existe, o valor neutral
                    classification = self._cached_classifications.get(
                        track_id,
                        {
                            "team_id": 0,
                            "role": "player",
                            "team_color": {"hex": "#FFFFFF", "rgb": [255, 255, 255], "bgr": [255, 255, 255]},
                            "confidence": 0.5,
                            "is_locked": False,
                        }
                    )

                if track_id is not None:
                    self._cached_classifications[track_id] = classification

            p["team_id"] = classification["team_id"]
            p["role"] = classification["role"]
            p["team_color"] = classification["team_color"]
            p["team_confidence"] = classification["confidence"]

        return player_detections

    def get_teams_metadata(self) -> Dict[str, Any]:
        """Retorna diccionario estructurado con los equipos detectados para metadata JSON."""
        c0 = self.team_colors_bgr.get(0, (0, 0, 255))
        c1 = self.team_colors_bgr.get(1, (255, 255, 255))
        cr = self.team_colors_bgr.get(-1, (40, 40, 40))
        return {
            "0": {
                "name": "Team A",
                "primary_color_hex": self.team_colors_hex.get(0, "#FF0000"),
                "primary_color_rgb": [int(c0[2]), int(c0[1]), int(c0[0])],
                "primary_color_bgr": list(c0),
                "role_name": "outfield",
                "real_team_id": None,
                "real_team_name": None,
            },
            "1": {
                "name": "Team B",
                "primary_color_hex": self.team_colors_hex.get(1, "#FFFFFF"),
                "primary_color_rgb": [int(c1[2]), int(c1[1]), int(c1[0])],
                "primary_color_bgr": list(c1),
                "role_name": "outfield",
                "real_team_id": None,
                "real_team_name": None,
            },
            "-1": {
                "name": "Referees",
                "primary_color_hex": self.team_colors_hex.get(-1, "#282828"),
                "primary_color_rgb": [int(cr[2]), int(cr[1]), int(cr[0])],
                "primary_color_bgr": list(cr),
                "role_name": "referee",
                "real_team_id": None,
                "real_team_name": "Árbitros",
            },
        }
