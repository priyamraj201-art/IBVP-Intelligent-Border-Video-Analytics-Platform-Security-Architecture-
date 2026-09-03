import cv2
import asyncio
import json
import threading
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_websockets = []
latest_frame = None
lock = threading.Lock()

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            # keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

def broadcast_alert(alert_data: dict):
    # This must run in the asyncio event loop
    async def _send():
        disconnected = []
        for ws in active_websockets:
            try:
                await ws.send_text(json.dumps(alert_data))
            except:
                disconnected.append(ws)
        for ws in disconnected:
            active_websockets.remove(ws)
    
    # We create a new event loop if called from a background thread
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_send())
        else:
            loop.run_until_complete(_send())
    except RuntimeError:
        asyncio.run(_send())


import torch
import os.path as osp
import sys

# Ensure YOLO modules can be found
sys.path.insert(0, osp.abspath(osp.dirname(__file__)))

from yolox.exp import get_exp
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from tools.demo_track import Predictor, get_model_info

from yolox.frs import FRSPipeline, FRSVisualizer

class MockArgs:
    track_thresh = 0.25
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False
    device = "gpu" if torch.cuda.is_available() else "cpu"
    fp16 = False

def tracking_loop():
    global latest_frame
    print("[INFO] Initializing YOLO Tracker...")
    args = MockArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp = get_exp("exps/example/mot/yolox_x_mix_det.py", None)
    exp.test_conf = 0.25
    model = exp.get_model().to(device)
    model.eval()
    
    ckpt_file = "pretrained/bytetrack_x_mot17.pth.tar"
    print(f"[INFO] Loading checkpoint {ckpt_file}...")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    ckpt_state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(ckpt_state_dict, strict=False)
    
    predictor = Predictor(model, exp, None, None, device, args.fp16)
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    
    print("[INFO] Initializing FRS Pipeline...")
    frs_pipeline = FRSPipeline(
        db_path="frs_faces.db",
        min_box_area=1500.0,
        num_workers=1,
        match_threshold=0.45,
    )
    last_alert_time = {}
    
    print("[INFO] Starting Webcam capture...")
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        outputs, img_info = predictor.inference(frame, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            
            # Run Facial Recognition
            frs_results = frs_pipeline.process_frame(
                img_info['raw_img'], online_tlwhs, online_ids, online_scores, current_time=time.time()
            )
            
            # Draw AI bounding boxes on the frame
            frame = FRSVisualizer.draw_frs_overlay(
                img_info['raw_img'], online_tlwhs, online_ids, frs_results, frame_id=frame_count, fps=1./max(1e-5, timer.average_time)
            )
            
            # Send alerts for matched VIPs/Suspects
            now = time.time()
            for tid, res in (frs_results or {}).items():
                person_id = res.get('person_id')
                if person_id and person_id != 'UNKNOWN' and res.get('confidence', 0) > 0.45:
                    if now - last_alert_time.get(person_id, 0) > 3:
                        last_alert_time[person_id] = now
                        print(f"[ALERT] Broadcasting FRS_HIT for {res.get('name')}")
                        broadcast_alert({
                            "type": "FRS_HIT",
                            "name": res.get("name", "Unknown"),
                            "category": res.get("category", "UNKNOWN"),
                            "timestamp": now,
                            "plate": person_id  # Mocking as plate for generic alert UI if needed
                        })
        
        # Removed BSF text overlay
        
        frame_count += 1
            
        with lock:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                latest_frame = buffer.tobytes()

# Start tracking in background
threading.Thread(target=tracking_loop, daemon=True).start()

def frame_generator():
    while True:
        with lock:
            frame = latest_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.get("/api/stream")
def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

# REST API for Watchlist (Dummy for now)
class WatchlistEntry(BaseModel):
    plate: str
    category: str

@app.post("/api/watchlist/anpr")
def add_anpr_watchlist(entry: WatchlistEntry):
    return {"status": "success", "message": f"Added {entry.plate} to watchlist"}

@app.get("/api/persons")
def get_persons():
    from yolox.frs.face_database import FaceDatabase
    db = FaceDatabase(db_path="frs_faces.db")
    identities = db.get_all_identities()
    return {"status": "success", "data": identities}
