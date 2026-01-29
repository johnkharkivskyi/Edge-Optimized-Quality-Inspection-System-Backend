import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
import time
import os


# 1. THE UPDATED ARCHITECTURE (Distillation / STFPM style)
# We now have a Student and Teacher instead of an Encoder and Decoder
class DistillationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Both are EfficientNet-B0
        self.student = models.efficientnet_b0(weights=None).features
        self.teacher = models.efficientnet_b0(weights=None).features
        # We don't train the teacher, so it stays frozen
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        s_feat = self.student(x)
        t_feat = self.teacher(x)
        return s_feat, t_feat


# 2. THE WRAPPER
class RealQualityModel:
    def __init__(self, mode="FP32"):
        # CRITICAL FIX: Define attributes FIRST so they exist even if loading fails
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 256
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.mode = mode

        # Initialize Model
        self.model = DistillationModel().to(self.device)

        model_path = os.path.join("engine", "best_model.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                # Extract the correct state dict
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                else:
                    state_dict = checkpoint

                # --- KEY MAPPING HACK ---
                # Your checkpoint has keys like 'student_base.features...' or 'student.features...'
                # Our class expects 'student...' and 'teacher...'
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('student_base.features', 'student')
                    new_key = new_key.replace('teacher_base.features', 'teacher')
                    new_key = new_key.replace('student.features', 'student')  # handle both
                    new_key = new_key.replace('teacher.features', 'teacher')
                    new_state_dict[new_key] = v

                # Load with strict=False to ignore any extra keys (like Adapters or Optimizers)
                self.model.load_state_dict(new_state_dict, strict=False)
                print(f"✅ SUCCESS: Distillation model loaded on {self.device}")

            except Exception as e:
                print(f"⚠️ MODEL LOAD WARNING: {e}")

        self.model.float().eval()

    def predict(self, frame):
        start_time = time.time()
        h, w = frame.shape[:2]

        # --- PREPROCESSING ---
        img_res = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            input_norm = (img_tensor - self.mean) / self.std

            # --- INFERENCE (Distillation Style) ---
            # Instead of reconstruction, we compare the features
            s_feat, t_feat = self.model(input_norm)

            # Anomaly map = Cosine distance or Euclidean distance between features
            # We resize the feature map back to 256x256
            diff = F.mse_loss(s_feat, t_feat, reduction='none')
            anomaly_map = torch.mean(diff, dim=1).squeeze()
            anomaly_map = F.interpolate(anomaly_map.unsqueeze(0).unsqueeze(0), size=(256, 256),
                                        mode='bilinear').squeeze()

            # Self-Calibration
            map_min, map_max = anomaly_map.min(), anomaly_map.max()
            anomaly_map = (anomaly_map - map_min) / (map_max - map_min + 1e-6)

            final_map = anomaly_map.cpu().numpy()

        # --- POST-PROCESSING ---
        # With distillation, the 'suspicion' is usually in the top 2% of pixels
        threshold = max(np.percentile(final_map, 98), 0.35)

        mask = (final_map > threshold).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                scale_x, scale_y = w / self.img_size, h / self.img_size

                intensity = np.mean(final_map[y:y + h_box, x:x + w_box])
                conf = min(0.99, 0.85 + intensity)

                detections.append({
                    "label": "ANOMALY",
                    "confidence": round(float(conf), 2),
                    "bbox": [int(x * scale_x), int(y * scale_y), int(w_box * scale_x), int(h_box * scale_y)]
                })

        inference_ms = (time.time() - start_time) * 1000
        return detections, inference_ms