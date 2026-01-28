import cv2
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from engine.model_provider import provider
from utils.hardware import get_hardware_stats

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SHARED CACHE: This stays updated in the background
hw_stats_cache = {"cpu": 0, "ram": 0, "gpu": 0, "platform": "Loading..."}


async def update_hardware_background():
    """Background task to poll hardware at 10Hz (every 100ms)."""
    global hw_stats_cache
    while True:
        try:
            # Polling hardware outside the video loop
            hw_stats_cache = get_hardware_stats()
        except Exception as e:
            print(f"HW Poll Error: {e}")
        await asyncio.sleep(0.1)  # 10 updates per second


@app.on_event("startup")
async def startup_event():
    # Start the hardware monitor as soon as the server starts
    asyncio.create_task(update_hardware_background())


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    camera = cv2.VideoCapture(0)

    try:
        while True:
            # 1. Listen for Frontend Commands (No delay)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if msg in ["INT8", "FP32"]:
                    provider.set_mode(msg)
            except asyncio.TimeoutError:
                pass

            # 2. Capture Frame
            success, frame = camera.read()
            if not success: break

            # 3. AI Inference
            model = provider.get_active_model()
            detections, latency = model.predict(frame)

            # 4. READ FROM CACHE (Zero Delay)
            # We no longer "call" the hardware function here.
            # We just take the latest data from the background task.
            current_hw = hw_stats_cache

            # 5. Data Encoding
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')

            # 6. Push Everything
            await websocket.send_json({
                "image": img_str,
                "detections": detections,
                "metrics": {
                    "latency": latency,
                    "mode": provider.current_mode,
                    **current_hw
                }
            })

            # Sync to ~30-40 FPS
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    finally:
        camera.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)