import cv2
import threading
import time
from logic_detect import detect_violence

class CameraStream:
    def __init__(self, source):
        self.source = source
        self.capture = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.capture = cv2.VideoCapture(self.source)
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            frame = detect_violence(frame)
            with self.lock:
                self.frame = frame
            time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            _, buffer = cv2.imencode('.jpg', self.frame)
            return buffer.tobytes()