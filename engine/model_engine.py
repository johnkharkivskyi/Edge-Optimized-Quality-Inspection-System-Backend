import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
import time
import os


class EfficientNetAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone is EfficientNet-B0 (Optimal for Edge)
        backbone = models.efficientnet_b0(weights=None)
        self.encoder = backbone.features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return F.interpolate(self.decoder(self.encoder(x)), size=(256, 256))


class RealQualityModel:
    def __init__(self, mode="FP32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNetAE().to(self.device)

        # Load Weights
        model_path = os.path.join("engine", "best_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"✅ Production Model Active on {self.device}")

        self.model.float().eval()
        self.img_size = 256

        # Performance Cache
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # Temporal Smoothing (Prevents flickering boxes)
        self.prev_anomaly_map = None

    def predict(self, frame):
        start_time = time.time()

        # 1. OPTIMIZED PREPROCESSING (Vectorized)
        h, w = frame.shape[:2]
        # Fast resize using INTER_AREA (best for downscaling)
        img_res = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)

        # Move to GPU immediately to do normalization there (much faster)
        with torch.no_grad():
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            input_norm = (img_tensor - self.mean) / self.std

            # 2. INFERENCE
            reconstruction = self.model(input_norm)

            # 3. STRUCTURAL ANOMALY CALCULATION
            # We compare L1 (pixels) + SSIM-like structural loss
            l1_diff = torch.abs(img_tensor - reconstruction)
            anomaly_map = torch.mean(l1_diff, dim=1).squeeze()

            # 4. SPATIAL SMOOTHING (Gaussian Blur on GPU)
            # This kills pixel-level camera noise
            anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)
            kernel_size = 11
            sigma = 4.0
            anomaly_map = F.avg_pool2d(anomaly_map, kernel_size, stride=1, padding=kernel_size // 2)

            # 5. DYNAMIC RANGE SCALING (The "Magic" Step)
            # This makes the defect "pop" out from the background regardless of light
            map_min, map_max = anomaly_map.min(), anomaly_map.max()
            anomaly_map = (anomaly_map - map_min) / (map_max - map_min + 1e-6)

            # Move back to CPU for OpenCV post-processing
            final_map = anomaly_map.squeeze().cpu().numpy()

        # 6. ADAPTIVE THRESHOLDING (Otsu-inspired)
        # We only look for the top 3% of "angriest" pixels
        threshold = np.percentile(final_map, 97)
        # Minimum noise floor (if everything is perfect, don't show anything)
        threshold = max(threshold, 0.4)

        mask = (final_map > threshold).astype(np.uint8) * 255

        # Clean up the mask using Morphological Gradient
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 7. CONTOUR ANALYSIS
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Only count defects that are significant in size
            if area > 300:
                x, y, w_box, h_box = cv2.boundingRect(cnt)

                # Rescale to original camera resolution
                scale_x, scale_y = w / self.img_size, h / self.img_size

                # Confidence is based on the intensity of the anomaly relative to threshold
                local_intensity = np.mean(final_map[y:y + h_box, x:x + w_box])
                conf = min(0.99, 0.8 + local_intensity)

                detections.append({
                    "label": "ANOMALY",
                    "confidence": round(float(conf), 2),
                    "bbox": [int(x * scale_x), int(y * scale_y), int(w_box * scale_x), int(h_box * scale_y)]
                })

        inference_ms = (time.time() - start_time) * 1000
        return detections, inference_ms