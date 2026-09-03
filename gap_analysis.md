# IBVAP — Tech Stack Gap Analysis
### Current Codebase vs. Problem Statement Requirements

---

## ✅ Already Implemented (What You Have)

| Requirement | Status | Module |
|---|---|---|
| Human detection & tracking | ✅ Full | `BYTETracker` + YOLOX |
| Vehicle detection & classification | ✅ Full | Multi-class routing + COCO |
| ANPR | ✅ Full | `yolox/anpr/` pipeline |
| Real-time alert generation | ✅ Partial | `MotionAlertSystem` (speed-based only) |
| Event logging (SQLite) | ✅ Full | `WatchlistDB` audit logs |
| Multi-class track routing | ✅ Full | `TrackRouter` |

---

## ❌ Missing Features — Full Breakdown

---

### 1. 🧑 Face Detection + Facial Recognition (FRS)
**PS Requirement:** *"Facial Recognition Systems (FRS)"*, *"Support facial recognition through software"*

**Currently:** ❌ Zero face detection code exists.

#### Recommended Stack

| Component | Library | Why |
|---|---|---|
| Face Detection | `InsightFace` (RetinaFace backbone) | Best accuracy, ONNX-ready, handles small faces at distance |
| Face Recognition | `InsightFace` (`buffalo_l` model) | 512-d ArcFace embeddings, top IJBC accuracy |
| Embedding DB | `faiss` (Facebook AI) | GPU-accelerated nearest-neighbor search at scale |
| Fallback (CPU) | `face_recognition` (dlib HOG) | Lightweight, no GPU needed |

#### How to Add — Module Plan
```
yolox/frs/
├── __init__.py
├── face_detector.py        ← InsightFace RetinaFace wrapper
├── face_embedder.py        ← ArcFace 512-d embedding extractor
├── face_database.py        ← FAISS index + SQLite identity store
├── frs_pipeline.py         ← Async face matching pipeline (mirrors ANPRPipeline)
└── frs_visualizer.py       ← Overlay rendering
```

**Integration point in `demo_track.py`:**
```python
# After routing → human_tracks → crop face ROI → FRS pipeline
frs_results = frs_pipeline.process_frame(
    raw_img, hum_tlwhs, hum_ids, current_time=time.time()
)
```

#### Efficiency Rating: ⭐⭐⭐⭐⭐
- InsightFace runs at **30+ FPS on GPU**, 5–8 FPS on CPU
- FAISS IndexFlatIP (cosine sim on 512-d) scales to **millions of faces in milliseconds**
- Can be pipelined with ByteTrack: only run FRS every N frames per track ID (same quality-gating logic as ANPR)

---

### 2. 🔴 Virtual Fence / Intrusion Detection
**PS Requirement:** *"Virtual fence intrusion detection"*, *"real-time alerts for border intrusions"*

**Currently:** ❌ No zone/ROI/polygon logic exists.

#### Recommended Stack

| Component | Library | Why |
|---|---|---|
| Polygon ROI definition | `cv2.fillPoly`, `shapely` | Define multi-point virtual fence zones |
| Point-in-polygon test | `shapely.geometry.Point.within()` | O(1) per check |
| Line-crossing detection | Custom vector cross product | Detect direction of crossing (in vs out) |
| Zone config storage | JSON / YAML | Operator-configurable fence lines per camera |

#### How to Add — Module Plan
```
yolox/intrusion/
├── __init__.py
├── zone_manager.py         ← Load/save fence polygons from JSON config
├── intrusion_detector.py   ← Point-in-polygon + line crossing logic
└── intrusion_visualizer.py ← Draw fence zones + breach highlights
```

**Core logic:**
```python
# Track foot position (bottom-center of bbox)
foot_x = x + w / 2
foot_y = y + h

from shapely.geometry import Point, Polygon
zone = Polygon(fence_coords)
if Point(foot_x, foot_y).within(zone):
    trigger_intrusion_alert(track_id, zone_name)
```

#### Efficiency Rating: ⭐⭐⭐⭐⭐
- Shapely point-in-polygon: **< 1 µs per check**, scales to 1000+ tracks/frame
- Zero GPU cost — pure CPU geometry
- Direction-of-crossing via cross product of fence vector × motion vector

---

### 3. 🌙 Night-Time / Low-Light Movement Detection
**PS Requirement:** *"Night-time movement detection"*

**Currently:** ❌ No low-light handling. YOLOX degrades heavily in darkness.

#### Recommended Stack

| Component | Library | Why |
|---|---|---|
| Frame enhancement | `cv2.createCLAHE` (already in PlateDetector!) | CLAHE on L-channel of LAB image |
| IR / low-light normalization | `cv2.normalize` + gamma correction | Boost dark frames before YOLOX |
| Background subtraction | `cv2.createBackgroundSubtractorMOG2` | High sensitivity to motion in dark scenes |
| Thermal image support | Custom loader | If thermal cameras exist at border posts |

#### Recommended Stack (Advanced)

| Approach | Model | Why |
|---|---|---|
| Retrain on ExDark dataset | YOLOX fine-tuned | ExDark has 7,363 night/low-light images |
| Zero-DCE enhancement | `Zero-DCE` or `LLFlow` | Real-time deep enhancement, 50+ FPS |

**Integration point in `demo_track.py`:**
```python
if is_dark_frame(frame):       # mean pixel < threshold
    frame = enhance_low_light(frame)
```

#### How to Add — Module Plan
```
yolox/night/
├── __init__.py
├── illumination_checker.py  ← Detect dark frames (histogram mean)
├── image_enhancer.py        ← CLAHE + gamma + optional Zero-DCE
└── motion_bg_subtractor.py  ← MOG2 fallback for pure dark scenes
```

#### Efficiency Rating: ⭐⭐⭐⭐
- CLAHE: **< 2ms per frame** (pure CPU), reuses existing PlateDetector code
- MOG2: ~5ms per frame, handles pitch-black scenes with motion detection
- Zero-DCE: ~15ms on GPU

---

### 4. 🕵️ Suspicious Activity Detection
**PS Requirement:** *"Suspicious activity detection"*, *"behavioral analytics"*

**Currently:** ❌ Only speed-based motion. No behavioral pattern detection.

#### Recommended Stack

| Behavior | Approach | Library |
|---|---|---|
| Loitering | Track dwell time per zone | `time.time()` delta per track_id |
| Crowd formation | Track density heatmap | `scipy.spatial.cKDTree` |
| Direction anomaly | Speed vector + heading deviation | Custom geometry |
| Object abandonment | Stationary object without nearby human | Track stillness + dissociation |
| Perimeter pacing | Repeated back-and-forth in zone | Track position history |
| Pose estimation | `MediaPipe Pose` / `mmpose` | Detect crawling/fence-climbing postures |

#### How to Add — Module Plan
```
yolox/behavior/
├── __init__.py
├── loitering_detector.py      ← Per-zone dwell time tracking
├── crowd_analyzer.py          ← Density heatmap + anomaly threshold
├── direction_analyzer.py      ← Heading vector, flow anomaly
├── behavior_alert.py          ← Unified suspicious activity alert generator
└── behavior_visualizer.py     ← Heatmaps, dwell timers, zone highlights
```

#### Efficiency Rating: ⭐⭐⭐⭐
- Loitering/dwell: trivial O(n) per frame
- Density map: `scipy.cKDTree` fast enough for 200+ simultaneous tracks
- Pose estimation adds 20–40ms latency on GPU (optional)

---

### 5. 📡 Multi-Camera RTSP Stream Ingestion
**PS Requirement:** *"Ingest live video streams from standard IP-based CCTV cameras"*

**Currently:** ❌ Single video/webcam only. No RTSP, no multi-camera support.

#### Recommended Stack

| Component | Library | Why |
|---|---|---|
| RTSP reader | `cv2.VideoCapture("rtsp://...")` | Native OpenCV, works with most IP cameras |
| Multi-stream threading | `threading.Thread` per camera | Parallel frame grab without blocking |
| Queue-based pipeline | `queue.Queue` | Decouple capture from inference |
| GPU stream batching | `torch` batched inference | Multiple camera frames in one forward pass |

#### How to Add
```python
# yolox/stream/rtsp_manager.py
class RTSPStreamManager:
    def __init__(self, stream_urls: List[str]):
        # Start one reader thread per camera
        
# cameras.json config:
[
  {"id": 0, "name": "Gate-1",       "url": "rtsp://192.168.1.10:554/stream"},
  {"id": 1, "name": "Border-Road-N","url": "rtsp://192.168.1.11:554/stream"},
  {"id": 2, "name": "BOP-Alpha",    "url": "rtsp://10.0.0.5:554/ch1"}
]
```

#### Efficiency Rating: ⭐⭐⭐⭐
- RTSP + OpenCV: handles 8–16 cameras on a single server with GPU batching
- Separate capture thread per camera eliminates frame-drop latency

---

### 6. 🖥️ Web Dashboard / Operator UI
**PS Requirement:** *"Integration with command and control systems"*, *"Situational awareness"*

**Currently:** ❌ Only `cv2.imshow()`. No web UI, no remote access, no multi-camera grid.

#### Recommended Stack

| Component | Library | Why |
|---|---|---|
| Backend API | `FastAPI` + `uvicorn` | Async, WebSocket support, auto docs |
| Video streaming | `WebSocket` + MJPEG | Real-time annotated frames to browser |
| Frontend | `React` + `Tailwind CSS` | Camera grid, alert feed, map |
| State management | `Redis` | Shared live alert state across workers |
| Database | `PostgreSQL` (upgrade from SQLite) | Production-grade, concurrent writes |
| Real-time push | `WebSocket` / Server-Sent Events | Instant alert delivery to operators |

#### Architecture
```
FastAPI Backend ← WebSocket → React Dashboard
     ↓                              ↓
PostgreSQL (events)          Camera Grid + Alert Feed
Redis (live state)           Plate Search + Map View
```

#### Efficiency Rating: ⭐⭐⭐⭐⭐
- FastAPI WebSocket pushes **annotated frames at 15–25 FPS** to browser
- React shows multi-camera grid, alert timeline, plate/face search

---

### 7. 🔔 Real-Time Notification System
**PS Requirement:** *"Real-time alert generation"*, *"improve response time"*

**Currently:** ✅ Console/log alerts + `winsound` beep only. ❌ No push/SMS/email.

#### Recommended Stack

| Channel | Library | Use Case |
|---|---|---|
| Telegram bot | `python-telegram-bot` | Low-cost, works on 2G/4G in remote areas |
| SMS alerts | `Twilio` API | Immediate HIGH alerts to field units |
| Email | `smtplib` / `SendGrid` | Detailed incident report with snapshot |
| Mobile push | `Firebase FCM` | Mobile app for officers |

```python
# yolox/alerts/notification_dispatcher.py
class NotificationDispatcher:
    def dispatch(self, alert_type, track_id, snapshot, location):
        if alert_type == "INTRUSION":
            self.send_telegram(f"INTRUSION at Gate-1 | Track #{track_id}")
            self.send_sms("+91XXXXXXXXXX", "Border breach detected")
```

#### Efficiency Rating: ⭐⭐⭐⭐
- Telegram: free, < 500ms delivery, works on 2G/4G
- Twilio SMS: async API call < 100ms

---

## 📊 Complete Priority + Efficiency Matrix

| Feature | Priority | Dev Effort | GPU Cost | CPU Cost | Best Library |
|---|---|---|---|---|---|
| Virtual Fence / Intrusion | 🔴 Critical | **Low** (1 day) | None | Negligible | `shapely` |
| Multi-Camera RTSP | 🔴 Critical | Medium (2 days) | None | Low | `OpenCV` + threading |
| Face Detection + FRS | 🔴 Critical | Medium (3 days) | High | Medium | `InsightFace` + `FAISS` |
| Night-Time Enhancement | 🟠 High | **Low** (hours) | Low | Very Low | `CLAHE` + `MOG2` |
| Suspicious Activity | 🟠 High | Medium (2 days) | None | Low | `scipy` + custom |
| Web Dashboard + API | 🟠 High | High (1 week) | None | Low | `FastAPI` + `React` |
| Telegram Notifications | 🟡 Medium | **Low** (hours) | None | Negligible | `python-telegram-bot` |
| GIS Map Integration | 🟡 Medium | Medium (2 days) | None | Negligible | `Leaflet.js` |

---

## 🏗️ Final Recommended Architecture

```
CCTV Cameras (RTSP/IP)
       ↓
┌──────────────────────────────────────────────────────────┐
│              IBVAP Core Engine (Python)                   │
│                                                          │
│  RTSPStreamManager → Frame Buffer Queue                  │
│       ↓                                                  │
│  YOLOX Detector (GPU Batched, multi-camera)              │
│       ↓                                                  │
│  BYTETracker (HUMAN + VEHICLE + FACE classes)            │
│       ↓                                                  │
│  TrackRouter ────┬─────────────────────────────┐        │
│                  ↓              ↓               ↓        │
│           [FRS Pipeline]  [ANPR Pipeline]  [Intrusion]  │
│           InsightFace     EasyOCR          Shapely zones │
│                  ↓              ↓               ↓        │
│           [Alert System] + [Behavior Detector]           │
│           Speed + Dwell     Loitering + Crowd            │
│                       ↓                                  │
│       NotificationDispatcher → Telegram / SMS / Email    │
│       EventLogger → PostgreSQL                           │
└──────────────────────────────────────────────────────────┘
       ↓
FastAPI Backend (WebSocket + REST API)
       ↓
React Dashboard (Camera Grid + Alert Feed + Map + Search)
```

---

## 🚀 SIH Demo Sprint Plan

### Sprint 1 — Quick Wins (Day 1–2)
1. `shapely` virtual fence intrusion — **2 hours, massive visual impact**
2. CLAHE night enhancement hook — **1 hour**
3. Telegram notification dispatcher — **2 hours**

### Sprint 2 — AI Features (Day 3–7)
4. InsightFace FRS pipeline (mirror ANPR architecture)
5. Loitering + behavior detection
6. Multi-RTSP camera config

### Sprint 3 — Platform (Week 2)
7. FastAPI backend + WebSocket video stream
8. React dashboard (camera grid + alert panel)

### Sprint 4 — Polish (Week 3)
9. GIS map integration (Leaflet.js)
10. Docker compose for easy deployment
