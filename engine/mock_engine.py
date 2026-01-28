import time
import random


class MockQualityModel:
    def __init__(self, mode="INT8"):
        self.mode = mode
        self.latency = 0.022 if mode == "INT8" else 0.078

    def predict(self, frame):
        start_time = time.time()
        time.sleep(self.latency)

        detections = []
        # Decide if this image has anomalies (40% chance for demo)
        if random.random() > 0.6:
            # Generate 1 to 3 random anomalies
            num_defects = random.randint(1, 3)
            for _ in range(num_defects):
                # Create random positions that don't always overlap
                w, h = random.randint(50, 150), random.randint(50, 150)
                x = random.randint(50, 400)
                y = random.randint(50, 300)

                detections.append({
                    "label": "ANOMALY",
                    "confidence": round(random.uniform(0.85, 0.99), 2),
                    "bbox": [x, y, w, h]
                })

        inference_ms = (time.time() - start_time) * 1000
        return detections, inference_ms