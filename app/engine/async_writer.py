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
        buffer_size: int = 60,
    ):
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread = threading.Thread(target=self._write_loop, daemon=True, name="AsyncVideoWriter")
        self._thread.start()
        self._frame_count = 0
        logger.info(f"[AsyncVideoWriter] Iniciado → {output_path} ({size[0]}x{size[1]} @ {fps:.1f}fps, buffer={buffer_size})")

    def write(self, frame: np.ndarray):
        """Encola frame para escritura asíncrona. No bloquea el caller."""
        try:
            self._queue.put_nowait(frame)
            self._frame_count += 1
        except queue.Full:
            # Si la cola está llena (disco muy lento), descartamos el frame
            # para no bloquear el pipeline de inferencia.
            logger.warning("[AsyncVideoWriter] ⚠️  Cola llena, frame descartado (disco lento).")

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
        logger.info(f"[AsyncVideoWriter] ✅ Cerrado. Total frames escritos: {self._frame_count}")

    @property
    def pending_frames(self) -> int:
        return self._queue.qsize()
