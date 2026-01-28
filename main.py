import cv2
import base64
import asyncio
import time
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from engine.model_provider import provider
from utils.hardware import get_hardware_stats

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class GlobalState:
    def __init__(self):
        self.frame = None
        self.encoded_image = None
        self.detections = []
        self.latency = 0
        self.fps = 0
        self.hw = {"cpu": 0, "ram": 0, "gpu": 0, "platform": "Init..."}


state = GlobalState()


def camera_worker():
    # Back to standard capture (no DSHOW to avoid crashes)
    cap = cv2.VideoCapture(0)

    # We set these but don't touch Exposure
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_start = time.time()
    frames = 0

    while True:
        success, frame = cap.read()
        if not success: continue

        state.frame = frame

        # JPEG Quality 60 is the best balance
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        state.encoded_image = base64.b64encode(buffer).decode('utf-8')

        frames += 1
        if time.time() - fps_start > 1.0:
            state.fps = round(frames / (time.time() - fps_start), 1)
            frames = 0
            fps_start = time.time()


def inference_worker():
    while True:
        if state.frame is not None:
            model = provider.get_active_model()
            dets, lat = model.predict(state.frame)
            state.detections = dets
            state.latency = lat
        time.sleep(0.01)


def hardware_worker():
    while True:
        state.hw = get_hardware_stats()
        time.sleep(0.5)


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    threading.Thread(target=hardware_worker, daemon=True).start()


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Check for toggle
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if msg in ["INT8", "FP32"]:
                    provider.set_mode(msg)
            except asyncio.TimeoutError:
                pass

            if state.encoded_image:
                # Use the standard long names again to match your working App.js
                await websocket.send_json({
                    "image": state.encoded_image,
                    "detections": state.detections,
                    "metrics": {
                        "latency": state.latency,
                        "fps": state.fps,
                        "mode": provider.current_mode,
                        **state.hw
                    }
                })

            # 0.001 sleep is better for pushing the FPS limit
            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")