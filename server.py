import cv2
import asyncio
import json
import threading
import time
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
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
frs_pipeline_instance = None
lock = threading.Lock()

capture_lock = threading.Lock()
node_pending_frames = {} # node_id -> {"frame": np_array, "id": int}
node_latest_frames = {}  # node_id -> {"bytes": bytes, "version": int}
node_latest_raw_frames = {} # node_id -> {"bytes": bytes, "version": int}
global_capture_event = threading.Event()
active_camera_nodes = set()
JPEG_ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 85]

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
        pass

@app.websocket("/ws/camera/{node_id}")
async def camera_websocket(websocket: WebSocket, node_id: str):
    await websocket.accept()
    active_camera_nodes.add(node_id)
    frame_id_counter = 0
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("data:image"):
                base64_data = data.split(",")[1]
            else:
                base64_data = data
            
            img_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame_id_counter += 1
                with capture_lock:
                    if node_id not in node_pending_frames:
                        node_pending_frames[node_id] = {}
                    node_pending_frames[node_id]["frame"] = frame
                    node_pending_frames[node_id]["id"] = frame_id_counter
                global_capture_event.set()
    except WebSocketDisconnect:
        if node_id in active_camera_nodes:
            active_camera_nodes.remove(node_id)

@app.get("/api/nodes")
def get_active_nodes():
    return list(active_camera_nodes)


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
    print("[INFO] Initializing YOLO Tracker...")
    args = MockArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp = get_exp("exps/example/mot/yolox_s_mix_det.py", None)
    exp.test_conf = 0.25
    model = exp.get_model().to(device)
    model.eval()

    ckpt_file = "pretrained/bytetrack_s_mot17.pth.tar"
    print(f"[INFO] Loading checkpoint {ckpt_file}...")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    ckpt_state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(ckpt_state_dict, strict=False)
    
    predictor = Predictor(model, exp, None, None, device, args.fp16)
    
    print("[INFO] Initializing FRS Pipeline...")
    global frs_pipeline_instance
    frs_pipeline = FRSPipeline(
        db_path="frs_faces.db",
        min_box_area=1500.0,
        num_workers=1,
        match_threshold=0.45,
    )
    frs_pipeline_instance = frs_pipeline
    
    import os
    camera_source = os.environ.get("CAMERA_SOURCE", "0")
    if camera_source.isdigit():
        camera_source = int(camera_source)
    threading.Thread(target=capture_loop, args=(camera_source,), daemon=True).start()

    trackers = {}
    last_seen_ids = {}
    last_alert_times = {}
    frame_counts = {}

    while True:
        global_capture_event.wait(timeout=0.1)
        global_capture_event.clear()

        with capture_lock:
            nodes_to_process = list(node_pending_frames.keys())
            
        for node_id in nodes_to_process:
            with capture_lock:
                node_data = node_pending_frames.get(node_id, {})
                frame = node_data.get("frame")
                fid = node_data.get("id")
            
            if frame is None or fid == last_seen_ids.get(node_id, -1):
                continue
                
            last_seen_ids[node_id] = fid
            
            if node_id not in trackers:
                trackers[node_id] = BYTETracker(args, frame_rate=30)
                last_alert_times[node_id] = {}
                frame_counts[node_id] = 0

            tracker = trackers[node_id]
            last_alert_time = last_alert_times[node_id]
            frame_counts[node_id] += 1
            timer = Timer()

            with lock:
                ret_raw, buffer_raw = cv2.imencode('.jpg', frame, JPEG_ENCODE_PARAMS)
                if ret_raw:
                    if node_id not in node_latest_raw_frames:
                        node_latest_raw_frames[node_id] = {"version": 0}
                    node_latest_raw_frames[node_id]["bytes"] = buffer_raw.tobytes()
                    node_latest_raw_frames[node_id]["version"] += 1

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
            
            frame = FRSVisualizer.draw_frs_overlay(
                img_info['raw_img'], online_tlwhs, online_ids, frs_results, frame_id=frame_counts[node_id], fps=1./max(1e-5, timer.average_time)
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
                        print(f"[ALERT] Broadcasting FRS_HIT for {res.get('name')} from node {node_id}")
                        broadcast_alert({
                            "type": "FRS_HIT",
                            "name": res.get("name", "Unknown"),
                            "category": res.get("category", "UNKNOWN"),
                            "timestamp": now,
                            "plate": person_id,
                            "node_id": node_id
                        })
            
            with lock:
                ret, buffer = cv2.imencode('.jpg', frame, JPEG_ENCODE_PARAMS)
                if ret:
                    if node_id not in node_latest_frames:
                        node_latest_frames[node_id] = {"version": 0}
                    node_latest_frames[node_id]["bytes"] = buffer.tobytes()
                    node_latest_frames[node_id]["version"] += 1

def capture_loop(camera_source):
    # Default capture loop adds a 'local_camera' node
    import requests
    node_id = "local_camera"
    active_camera_nodes.add(node_id)

    def publish(frame):
        nonlocal node_id
        with capture_lock:
            if node_id not in node_pending_frames:
                node_pending_frames[node_id] = {"id": 0}
            node_pending_frames[node_id]["frame"] = frame
            node_pending_frames[node_id]["id"] += 1
        global_capture_event.set()

    if is_ip_camera:
        print(f"[INFO] Connecting to IP Camera MJPEG stream: {camera_source}")
        bytes_buffer = b''

        def connect_stream():
            try:
                return requests.get(camera_source, stream=True, timeout=5).iter_content(chunk_size=8192)
            except Exception:
                return None

        stream = connect_stream()
        while True:
            if stream is None:
                time.sleep(1.0)
                stream = connect_stream()
                continue
            try:
                chunk = next(stream)
                bytes_buffer += chunk
                # Guard against unbounded growth if JPEG markers are never found
                # (e.g. CAMERA_SOURCE pointing at an HTML page instead of the
                # MJPEG endpoint, which is /video for the IP Webcam app).
                if len(bytes_buffer) > 2_000_000:
                    bytes_buffer = bytes_buffer[-200_000:]
                a = bytes_buffer.find(b'\xff\xd8')
                b = bytes_buffer.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_buffer[a:b + 2]
                    bytes_buffer = bytes_buffer[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.resize(frame, (640, 480))
                        publish(frame)
            except StopIteration:
                print("[WARN] Stream ended, reconnecting...")
                stream = connect_stream()
            except Exception as e:
                print(f"[WARN] Stream read error: {e}")
                stream = connect_stream()
    else:
        print(f"[INFO] Starting Webcam capture from source: {camera_source}")
        cap = cv2.VideoCapture(camera_source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        while True:
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            publish(frame)

# Start tracking in background
threading.Thread(target=tracking_loop, daemon=True).start()

def frame_generator(node_id: str):
    last_version = -1
    last_sent = 0.0
    min_interval = 1.0 / 60.0  # cap at 60fps
    while True:
        with lock:
            node_data = node_latest_frames.get(node_id, {})
            frame = node_data.get("bytes")
            version = node_data.get("version", -1)
        now = time.time()
        if frame is not None and version != last_version and (now - last_sent) >= min_interval:
            last_version = version
            last_sent = now
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.004)

def raw_frame_generator(node_id: str):
    last_version = -1
    last_sent = 0.0
    min_interval = 1.0 / 60.0  # cap at 60fps
    while True:
        with lock:
            node_data = node_latest_raw_frames.get(node_id, {})
            frame = node_data.get("bytes")
            version = node_data.get("version", -1)
        now = time.time()
        if frame is not None and version != last_version and (now - last_sent) >= min_interval:
            last_version = version
            last_sent = now
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.004)

@app.get("/api/stream/{node_id}")
def video_feed(node_id: str):
    return StreamingResponse(frame_generator(node_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stream/raw/{node_id}")
def raw_video_feed(node_id: str):
    return StreamingResponse(raw_frame_generator(node_id), media_type="multipart/x-mixed-replace; boundary=frame")

from fastapi import Response

@app.get("/api/snapshot/{node_id}")
def get_snapshot(node_id: str):
    with lock:
        frame = node_latest_frames.get(node_id, {}).get("bytes")
    if frame is not None:
        return Response(content=frame, media_type="image/jpeg")
    return {"error": "no frame available"}

@app.get("/api/snapshot/raw/{node_id}")
def get_raw_snapshot(node_id: str):
    with lock:
        frame = node_latest_raw_frames.get(node_id, {}).get("bytes")
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
