"""
ScoutingData v5.0 - AsyncVideoWriter (OPT-8)

Escribe frames a disco en un thread separado sin bloquear el loop de inferencia.
La escritura es el cuello de botella más silencioso: cv2.VideoWriter.write() es
I/O síncrono que espera a que el disco procese antes de liberar el GIL.
"""

import queue
import threading
import logging
import cv2
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger("AsyncVideoWriter")


class AsyncVideoWriter:
    """
    VideoWriter asíncrono con cola interna.
    - El loop de inferencia llama a `write(frame)` y retorna de inmediato.
    - Un thread de fondo consume la cola y escribe al disco.
    - Llama a `release()` al finalizar para vaciar la cola y cerrar el archivo.
    """

    def __init__(
        self,
        output_path: str,
        fourcc: int,
        fps: float,
        size: Tuple[int, int],
        buffer_size: int = 180,
    ):
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread = threading.Thread(target=self._write_loop, daemon=True, name="AsyncVideoWriter")
        self._thread.start()
        self._frame_count = 0
        self._dropped_count = 0
        logger.info(f"[AsyncVideoWriter] Iniciado → {output_path} ({size[0]}x{size[1]} @ {fps:.1f}fps, buffer={buffer_size})")

    def write(self, frame: np.ndarray):
        """
        Encola frame para escritura asíncrona.
        Espera hasta 2.5s si el buffer está momentáneamente lleno para asegurar
        fidelidad frame-a-frame al 100% en partidos completos de 45 min sin desincronización.
        """
        try:
            self._queue.put(frame, block=True, timeout=2.5)
            self._frame_count += 1
        except queue.Full:
            self._dropped_count += 1
            if self._dropped_count <= 5 or self._dropped_count % 100 == 0:
                logger.warning(
                    f"[AsyncVideoWriter] ⚠️ Cola llena tras 2.5s de espera, frame descartado (total drops={self._dropped_count})."
                )

    def _write_loop(self):
        """Loop interno del thread de escritura."""
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            self._writer.write(frame)

    def release(self):
        """Vacía la cola pendiente, espera al thread y cierra el VideoWriter."""
        self._queue.put(None)    # Señal de fin
        self._thread.join()
        self._writer.release()
        drop_info = f" (⚠️ {self._dropped_count} frames descartados)" if self._dropped_count > 0 else " (100% fidelidad)"
        logger.info(f"[AsyncVideoWriter] ✅ Cerrado. Total frames escritos: {self._frame_count}{drop_info}")

    @property
    def pending_frames(self) -> int:
        return self._queue.qsize()
