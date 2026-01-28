import time
import random


class MockQualityModel:
    def __init__(self, mode="INT8"):
        self.mode = mode
        self.latency = 0.022 if mode == "INT8" else 0.082

        # PERSISTENCE STATE
        self.active_detections = []
        self.frames_left = 0  # How many frames the current anomaly "lives"

    def predict(self, frame):
        start_time = time.time()
        time.sleep(self.latency)

        # 1. If we have a persistent anomaly, keep showing it
        if self.frames_left > 0:
            self.frames_left -= 1
            # Add a tiny "jitter" (1-2 pixels) to make it look like a real live AI
            for det in self.active_detections:
                det["bbox"][0] += random.randint(-1, 1)
                det["bbox"][1] += random.randint(-1, 1)
        else:
            # 2. Decide if we should spawn a new anomaly (30% chance)
            if random.random() > 0.7:
                self.active_detections = []
                # Spawn 1 to 2 persistent anomalies
                for _ in range(random.randint(1, 2)):
                    w, h = random.randint(80, 160), random.randint(80, 160)
                    x, y = random.randint(100, 400), random.randint(100, 250)
                    self.active_detections.append({
                        "label": "ANOMALY",
                        "confidence": round(random.uniform(0.92, 0.99), 2),
                        "bbox": [x, y, w, h]
                    })
                # Keep these boxes for the next 45-90 frames (~1.5 to 3 seconds)
                self.frames_left = random.randint(45, 90)
            else:
                # No anomalies
                self.active_detections = []

        inference_ms = (time.time() - start_time) * 1000
        # Return a copy to avoid reference issues
        return [dict(d) for d in self.active_detections], inference_ms