import cv2
import asyncio
import json
import threading
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
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
latest_raw_frame = None
frs_pipeline_instance = None
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
    global latest_raw_frame
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
    global frs_pipeline_instance
    frs_pipeline = FRSPipeline(
        db_path="frs_faces.db",
        min_box_area=1500.0,
        num_workers=1,
        match_threshold=0.45,
    )
    frs_pipeline_instance = frs_pipeline
    last_alert_time = {}
    
    print("[INFO] Starting Webcam capture...")
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # Encode raw frame before any AI bounding boxes are drawn
        with lock:
            ret_raw, buffer_raw = cv2.imencode('.jpg', frame)
            if ret_raw:
                latest_raw_frame = buffer_raw.tobytes()
                
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
            active_tids = set(online_ids)
            for tid, res in (frs_results or {}).items():
                if tid not in active_tids:
                    continue
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

def raw_frame_generator():
    while True:
        with lock:
            frame = latest_raw_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.get("/api/stream")
def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stream/raw")
def raw_video_feed():
    return StreamingResponse(raw_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

from fastapi import Response

@app.get("/api/snapshot")
def get_snapshot():
    with lock:
        frame = latest_frame
    if frame is not None:
        return Response(content=frame, media_type="image/jpeg")
    return {"error": "no frame available"}

@app.get("/api/snapshot/raw")
def get_raw_snapshot():
    with lock:
        frame = latest_raw_frame
    if frame is not None:
        return Response(content=frame, media_type="image/jpeg")
    return {"error": "no raw frame available"}

# REST API for Watchlist (Dummy for now)
class WatchlistEntry(BaseModel):
    plate: str
    category: str

@app.post("/api/watchlist/anpr")
def add_anpr_watchlist(entry: WatchlistEntry):
    return {"status": "success", "message": f"Added {entry.plate} to watchlist"}

@app.post("/api/register_face")
async def register_face(
    name: str = Form(...),
    category: str = Form(...),
    image: UploadFile = File(...)
):
    import numpy as np
    import uuid
    import cv2
    from yolox.frs.face_detector import FaceDetector
    from yolox.frs.face_embedder import FaceEmbedder
    from yolox.frs.face_database import FaceDatabase

    import io
    from PIL import Image, ImageOps

    contents = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "Failed to decode image"}

    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = FaceDatabase(db_path="frs_faces.db")

    face_info = detector.get_best_face(img)
    if not face_info or face_info[1] < 0.6:
        return {"status": "error", "message": "No clear face detected in the image. Please use a closer/clearer photo."}

    face_crop, det_conf = face_info
    embedding, quality = embedder.get_embedding(face_crop)

    person_id = str(uuid.uuid4())[:8].upper()
    person_id = f"SUBJ_{person_id}"

    success = db.enroll_face(
        person_id=person_id,
        name=name,
        embedding=embedding,
        category=category,
        enrolled_by="web_dashboard"
    )

    if success:
        if frs_pipeline_instance:
            with frs_pipeline_instance._lock:
                for cand in frs_pipeline_instance.candidates.values():
                    cand.status = "UNPROCESSED"
                    cand.last_attempt_time = 0.0
                frs_pipeline_instance.results_cache.clear()
        return {"status": "success", "message": f"Successfully enrolled {name}"}
    else:
        return {"status": "error", "message": "Database enrollment failed"}

@app.get("/api/persons")
def get_persons():
    from yolox.frs.face_database import FaceDatabase
    db = FaceDatabase(db_path="frs_faces.db")
    identities = db.get_all_identities()
    return {"status": "success", "data": identities}

@app.delete("/api/persons/{person_id}")
def delete_person(person_id: str):
    from yolox.frs.face_database import FaceDatabase
    db = FaceDatabase(db_path="frs_faces.db")
    success = db.remove_identity(person_id)
    if success:
        if frs_pipeline_instance:
            with frs_pipeline_instance._lock:
                for cand in frs_pipeline_instance.candidates.values():
                    cand.status = "UNPROCESSED"
                    cand.last_attempt_time = 0.0
                frs_pipeline_instance.results_cache.clear()
        return {"status": "success", "message": "Identity removed"}
    return {"status": "error", "message": "Identity not found"}

class UpdatePersonRequest(BaseModel):
    category: str

@app.put("/api/persons/{person_id}")
def update_person(person_id: str, req: UpdatePersonRequest):
    from yolox.frs.face_database import FaceDatabase
    db = FaceDatabase(db_path="frs_faces.db")
    with db._lock:
        conn = db._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE face_identities SET category = ? WHERE person_id = ?", (req.category, person_id))
            if cursor.rowcount > 0:
                conn.commit()
                if frs_pipeline_instance:
                    with frs_pipeline_instance._lock:
                        for cand in frs_pipeline_instance.candidates.values():
                            cand.status = "UNPROCESSED"
                            cand.last_attempt_time = 0.0
                        frs_pipeline_instance.results_cache.clear()
                return {"status": "success"}
            return {"status": "error", "message": "Not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            conn.close()
