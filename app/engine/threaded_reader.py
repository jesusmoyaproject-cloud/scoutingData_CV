"""
ScoutingData v5.0 - ThreadedOpenCVReader (OPT-9)

Lector de video multihilo con prefetching en segundo plano y profiling de lectura y decodificación.
Evita que el loop principal de inferencia (GPU/CPU) se bloquee esperando I/O de disco y 
decodificación FFmpeg/OpenCV en cada frame.

Incluye profiling integrado de tiempo de decodificación (read_and_decode) tanto a nivel
de hilo en background como a nivel de consumidor en el pipeline principal.
"""

import queue
import threading
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np

logger = logging.getLogger("ThreadedOpenCVReader")


class ThreadedOpenCVReader:
    """
    Lector de video con buffer asíncrono y profiling de read_and_decode.
    - Un hilo de fondo lee y decodifica continuamente frames con OpenCV cv2.VideoCapture.
    - Almacena los frames decodificados en una cola acotada (queue.Queue) para desacoplar I/O e inferencia.
    - Mide con alta precisión el tiempo de decodificación por frame (read_and_decode) y métricas globales.
    - Implementa interfaz compatible con cv2.VideoCapture (read, isOpened, get, release).
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        buffer_size: int = 64,
        timeout: float = 10.0,
    ):
        self.video_path = str(video_path)
        self.buffer_size = buffer_size
        self.timeout = timeout

        self._cap = cv2.VideoCapture(self.video_path)
        self._is_opened = self._cap.isOpened()

        if not self._is_opened:
            logger.error(f"[ThreadedOpenCVReader] ❌ No se pudo abrir el archivo de video: {self.video_path}")
            self.width = 0
            self.height = 0
            self.fps = 25.0
            self.total_frames = 0
            self._released = True
            return

        # Metadatos del video
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.fps = raw_fps if raw_fps > 0 else 25.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Cola y sincronización de hilos
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._stopped = False
        self._released = False
        self._reached_eof = False
        self._lock = threading.Lock()

        # Métricas de profiling (read_and_decode)
        self._total_decode_time: float = 0.0
        self._frames_decoded: int = 0
        self._total_consumer_wait_time: float = 0.0
        self._frames_consumed: int = 0

        # Iniciar thread en segundo plano
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="ThreadedOpenCVReader",
        )
        self._thread.start()

        logger.info(
            f"[ThreadedOpenCVReader] 🚀 Iniciado → {self.video_path} "
            f"({self.width}x{self.height} @ {self.fps:.1f}fps, total={self.total_frames}, buffer={buffer_size})"
        )

    def isOpened(self) -> bool:
        """Indica si el reader está abierto y disponible para lectura."""
        if self._released:
            return False
        # El reader está disponible mientras no se haya liberado y
        # no haya terminado el video o aún queden frames en el buffer
        return self._is_opened and (not self._stopped or not self._queue.empty())

    def is_opened(self) -> bool:
        """Alias pythonic de isOpened."""
        return self.isOpened()

    def get(self, prop_id: int) -> float:
        """Delega consulta de propiedades a VideoCapture subyacente si sigue disponible."""
        if self._cap is not None:
            return self._cap.get(prop_id)
        return 0.0

    def _worker_loop(self):
        """Loop continuo del hilo lector: lee, decodifica y encola frames con profiling."""
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            decode_time = time.perf_counter() - t0

            if not ret or frame is None:
                self._reached_eof = True
                self._stopped = True
                # Encolar marcador de fin de archivo
                while not self._stop_event.is_set():
                    try:
                        self._queue.put((False, None), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                break

            with self._lock:
                self._total_decode_time += decode_time
                self._frames_decoded += 1

            # Encolar frame; reintentar con timeout para verificar stop_event periódicamente
            while not self._stop_event.is_set():
                try:
                    self._queue.put((True, frame), timeout=0.1)
                    break
                except queue.Full:
                    continue

    def read(self, timeout: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Obtiene el siguiente frame decodificado desde el buffer interno.
        Mide el tiempo de espera del consumidor (overhead de consumo).
        Retorna:
            (ret, frame_bgr)
        """
        if self._released:
            return False, None

        to = timeout if timeout is not None else self.timeout
        t0 = time.perf_counter()

        try:
            ret, frame = self._queue.get(timeout=to)
            wait_time = time.perf_counter() - t0

            with self._lock:
                self._total_consumer_wait_time += wait_time
                if ret and frame is not None:
                    self._frames_consumed += 1

            if not ret or frame is None:
                self._stopped = True
                return False, None

            return True, frame

        except queue.Empty:
            if self._reached_eof or self._stopped:
                return False, None
            logger.warning(
                f"[ThreadedOpenCVReader] ⚠️ Timeout ({to}s) esperando frame del buffer de decodificación."
            )
            return False, None

    def read_and_decode(self, timeout: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """Alias explícito para read(), para coincidir con la nomenclatura de profiling."""
        return self.read(timeout=timeout)

    def release(self):
        """Detiene el hilo lector, drena la cola y libera el VideoCapture."""
        if self._released:
            return

        self._stop_event.set()
        self._stopped = True

        # Drenar cola para desbloquear el worker si estaba esperando en queue.put
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._released = True

        logger.info(
            f"[ThreadedOpenCVReader] ✅ Cerrado. Decodificados: {self.frames_decoded} frames | "
            f"Consumidos: {self.frames_consumed} frames | "
            f"Tiempo decode worker: {self.total_decode_time:.2f}s ({self.avg_decode_time_ms:.2f} ms/frame, {self.decode_fps:.1f} FPS) | "
            f"Espera consumidor: {self.total_consumer_wait_time:.3f}s"
        )

    # ── Métricas y Propiedades de Profiling ──────────────────────────────────
    @property
    def total_decode_time(self) -> float:
        """Tiempo total acumulado en el thread lector decodificando frames (segundos)."""
        with self._lock:
            return self._total_decode_time

    @property
    def frames_decoded(self) -> int:
        """Número de frames decodificados por el worker thread."""
        with self._lock:
            return self._frames_decoded

    @property
    def frames_consumed(self) -> int:
        """Número de frames consumidos por el pipeline principal."""
        with self._lock:
            return self._frames_consumed

    @property
    def total_consumer_wait_time(self) -> float:
        """Tiempo total que el pipeline consumidor esperó para recibir frames (segundos)."""
        with self._lock:
            return self._total_consumer_wait_time

    @property
    def avg_decode_time_ms(self) -> float:
        """Tiempo promedio de decodificación por frame en el worker (milisegundos)."""
        with self._lock:
            if self._frames_decoded == 0:
                return 0.0
            return (self._total_decode_time / self._frames_decoded) * 1000.0

    @property
    def avg_consumer_wait_ms(self) -> float:
        """Tiempo promedio de espera del consumidor por frame (milisegundos)."""
        with self._lock:
            if self._frames_consumed == 0:
                return 0.0
            return (self._total_consumer_wait_time / self._frames_consumed) * 1000.0

    @property
    def decode_fps(self) -> float:
        """Velocidad efectiva de decodificación en frames por segundo (FPS)."""
        with self._lock:
            if self._total_decode_time <= 0:
                return 0.0
            return self._frames_decoded / self._total_decode_time

    @property
    def pending_frames(self) -> int:
        """Número actual de frames encolados y listos en el buffer."""
        return self._queue.qsize()

    def get_metrics(self) -> Dict[str, Any]:
        """Resumen estructurado de métricas de profiling de read_and_decode."""
        return {
            "total_decode_time_s": round(self.total_decode_time, 4),
            "frames_decoded": self.frames_decoded,
            "avg_decode_time_ms": round(self.avg_decode_time_ms, 2),
            "decode_fps": round(self.decode_fps, 2),
            "frames_consumed": self.frames_consumed,
            "total_consumer_wait_s": round(self.total_consumer_wait_time, 4),
            "avg_consumer_wait_ms": round(self.avg_consumer_wait_ms, 2),
            "buffer_size": self.buffer_size,
            "buffer_pending": self.pending_frames,
        }

    # ── Context Manager & Iterator ──────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.read()
        if not ret or frame is None:
            raise StopIteration
        return frame


def profile_read_and_decode(source: Union[cv2.VideoCapture, ThreadedOpenCVReader]) -> Tuple[bool, Optional[np.ndarray], float]:
    """
    Función utilitaria para leer un frame y medir exactamente el tiempo invertido en read_and_decode.
    Retorna: (ret, frame, elapsed_seconds)
    """
    t0 = time.perf_counter()
    ret, frame = source.read()
    elapsed = time.perf_counter() - t0
    return ret, frame, elapsed
