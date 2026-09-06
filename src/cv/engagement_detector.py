"""
CV engagement detection using OpenCV Haar cascade (switched from MediaPipe
due to API version incompatibility -- mp.solutions.face_detection doesn't
exist in newer MediaPipe installs).
"""

import time
import threading
import logging

log = logging.getLogger("engagement")


class EngagementDetector:
    def __init__(self, camera_index: int = 0, check_interval_s: float = 1.0,
                 absence_grace_s: float = 5.0, cascade_path: str = None):
        self.camera_index = camera_index
        self.check_interval_s = check_interval_s
        self.absence_grace_s = absence_grace_s
        self.cascade_path = cascade_path

        self._present = False
        self._last_seen_ts = 0
        self._running = False
        self._thread = None
        self._cap = None
        self._face_cascade = None
        self._last_frame = None

    def _init_detector(self):
        import cv2
        cascade_path = self.cascade_path or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError(
                f"Could not load Haar cascade from {cascade_path}. "
                "Your opencv-python install may not bundle this file -- "
                "download it manually and pass cascade_path explicitly."
            )

    def start(self):
        import cv2

        self._init_detector()
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Check that a camera is connected and not in use by another process."
            )

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("Engagement detector started (camera index %d)", self.camera_index)

    def stop(self):
        self._running = False
        if self._cap is not None:
            self._cap.release()

    def is_present(self) -> bool:
        if not self._running:
            return True
        return (time.time() - self._last_seen_ts) < self.absence_grace_s

    def get_last_frame(self):
        """Returns the most recently captured frame (BGR), or None."""
        return self._last_frame

    def _run_loop(self):
        import cv2
        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(self.check_interval_s)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                self._last_frame = frame

                if len(faces) > 0:
                    self._last_seen_ts = time.time()
                    if not self._present:
                        self._present = True
                        log.info("Patient detected -- engagement started")
                else:
                    if self._present and (time.time() - self._last_seen_ts) >= self.absence_grace_s:
                        self._present = False
                        log.info("No patient detected for %.0fs -- engagement ended", self.absence_grace_s)

                time.sleep(self.check_interval_s)
        except Exception:
            import traceback
            log.error("EngagementDetector worker thread crashed:")
            traceback.print_exc()
