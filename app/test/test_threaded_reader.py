"""
Test unitario y de integración para ThreadedOpenCVReader y profiling de read_and_decode.
"""

import os
import sys
import tempfile
import unittest
import time
from pathlib import Path
import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from engine.threaded_reader import ThreadedOpenCVReader, profile_read_and_decode
from main import SoccerAnalysisV5


def create_test_video(path: str, num_frames: int = 15, width: int = 320, height: int = 240, fps: int = 30):
    """Genera un archivo de video sintético ligero para testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 10) % 255, dtype=np.uint8)
        cv2.putText(frame, f"F:{i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


class TestThreadedOpenCVReader(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.tmpdir.name, "sample_test.mp4")
        self.num_frames = 12
        self.width = 320
        self.height = 240
        self.fps = 30
        create_test_video(self.video_path, num_frames=self.num_frames, width=self.width, height=self.height, fps=self.fps)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_initialization_and_metadata(self):
        """Verifica que el lector extraiga metadatos correctamente del video."""
        reader = ThreadedOpenCVReader(self.video_path, buffer_size=8)
        try:
            self.assertTrue(reader.isOpened())
            self.assertTrue(reader.is_opened())
            self.assertEqual(reader.width, self.width)
            self.assertEqual(reader.height, self.height)
            self.assertAlmostEqual(reader.fps, self.fps, delta=1.0)
            self.assertEqual(reader.total_frames, self.num_frames)
            self.assertEqual(int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), self.width)
        finally:
            reader.release()

    def test_full_read_and_metrics(self):
        """Lee todos los frames y valida métricas de profiling de read_and_decode."""
        reader = ThreadedOpenCVReader(self.video_path, buffer_size=8)
        read_frames = []
        try:
            while reader.isOpened():
                ret, frame = reader.read()
                if not ret:
                    break
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape, (self.height, self.width, 3))
                read_frames.append(frame)

            self.assertEqual(len(read_frames), self.num_frames)

            # Verificar profiling
            self.assertEqual(reader.frames_consumed, self.num_frames)
            self.assertEqual(reader.frames_decoded, self.num_frames)
            self.assertGreater(reader.total_decode_time, 0.0)
            self.assertGreater(reader.avg_decode_time_ms, 0.0)
            self.assertGreater(reader.decode_fps, 0.0)

            metrics = reader.get_metrics()
            self.assertIn("total_decode_time_s", metrics)
            self.assertIn("frames_decoded", metrics)
            self.assertIn("avg_decode_time_ms", metrics)
            self.assertIn("decode_fps", metrics)
            self.assertIn("total_consumer_wait_s", metrics)
            self.assertEqual(metrics["frames_decoded"], self.num_frames)
        finally:
            reader.release()

    def test_read_and_decode_alias_and_profiler(self):
        """Prueba read_and_decode() como alias y la función utilitaria de profiling."""
        reader = ThreadedOpenCVReader(self.video_path, buffer_size=4)
        try:
            ret, frame = reader.read_and_decode()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)

            ret2, frame2, elapsed = profile_read_and_decode(reader)
            self.assertTrue(ret2)
            self.assertIsNotNone(frame2)
            self.assertGreaterEqual(elapsed, 0.0)
        finally:
            reader.release()

    def test_early_release_no_deadlock(self):
        """Verifica que el reader termine limpiamente y sin deadlock si el consumidor sale antes."""
        reader = ThreadedOpenCVReader(self.video_path, buffer_size=4)
        ret, frame = reader.read()
        self.assertTrue(ret)
        # Salir tempranamente sin leer el resto de frames
        t0 = time.time()
        reader.release()
        release_time = time.time() - t0
        self.assertLess(release_time, 1.5, "La liberación del reader tardó demasiado (posible hang).")
        self.assertFalse(reader.isOpened())
        ret_after, frame_after = reader.read()
        self.assertFalse(ret_after)
        self.assertIsNone(frame_after)

    def test_context_manager_and_iteration(self):
        """Prueba el uso de ThreadedOpenCVReader como context manager e iterable."""
        with ThreadedOpenCVReader(self.video_path, buffer_size=6) as reader:
            frames = list(reader)
            self.assertEqual(len(frames), self.num_frames)
            self.assertGreater(reader.frames_decoded, 0)

    def test_nonexistent_file(self):
        """Verifica comportamiento robusto ante archivos inexistentes."""
        non_existent = os.path.join(self.tmpdir.name, "missing_video.mp4")
        reader = ThreadedOpenCVReader(non_existent)
        self.assertFalse(reader.isOpened())
        ret, frame = reader.read()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        reader.release()  # No debe lanzar excepción

    def test_integration_pipeline_with_threaded_reader(self):
        """Prueba de integración en SoccerAnalysisV5 con use_threaded_reader=True."""
        output_json = os.path.join(self.tmpdir.name, "out_tracking.json")
        pipeline = SoccerAnalysisV5(headless=True, use_threaded_reader=True, reader_buffer_size=16)

        pipeline.process_video(
            video_path=self.video_path,
            output_path=os.path.join(self.tmpdir.name, "dummy.mp4"),
            json_output_path=output_json,
            headless=True,
            use_threaded_reader=True,
            max_frames=5,
        )

        self.assertTrue(os.path.exists(output_json))
        import json
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("performance", data["metadata"])
        perf = data["metadata"]["performance"]
        self.assertIn("read_and_decode_sec", perf)
        self.assertIn("reader_metrics", perf)
        self.assertIsNotNone(perf["reader_metrics"])
        self.assertGreaterEqual(perf["reader_metrics"]["frames_decoded"], 5)


if __name__ == "__main__":
    unittest.main()
