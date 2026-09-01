import os
import time
import queue
import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from .face_database import FaceDatabase
from .face_detector import FaceDetector
from .face_embedder import FaceEmbedder


class FaceTrackCandidate:
    """
    Maintains biometric observation quality and recognition state for a tracked human candidate.
    """

    def __init__(self, track_id: int):
        self.track_id: int = track_id
        self.best_score: float = 0.0
        self.best_area: float = 0.0
        self.best_crop: Optional[np.ndarray] = None
        self.status: str = "UNPROCESSED"  # UNPROCESSED, QUEUED, COMPLETED, NO_FACE
        self.last_update_time: float = time.time()
        self.last_attempt_time: float = 0.0
        self.face_result: Optional[Dict[str, Any]] = None


class FRSPipeline:
    """
    High-Performance Asynchronous Facial Recognition System (FRS) Controller.

    1. Selects the highest-resolution human crop per track (area x confidence).
    2. Executes face detection, alignment, embedding extraction, and watchlist matching asynchronously.
    3. Employs quality-gated re-evaluations when targets move closer to the camera.
    4. Securely logs forensic identification events and watchlist hits to SQLite.
    """

    def __init__(
        self,
        db_path: str = "frs_faces.db",
        min_box_area: float = 1500.0,
        quality_boost_ratio: float = 1.3,
        recheck_interval_sec: float = 1.0,
        num_workers: int = 1,
        match_threshold: float = 0.45,
        save_crops_dir: Optional[str] = None,
    ):
        self.db_path = db_path
        self.min_box_area = min_box_area
        self.quality_boost_ratio = quality_boost_ratio
        self.recheck_interval_sec = recheck_interval_sec
        self.num_workers = num_workers
        self.match_threshold = match_threshold
        self.save_crops_dir = save_crops_dir

        if self.save_crops_dir:
            os.makedirs(self.save_crops_dir, exist_ok=True)

        # Core Components
        self.face_db = FaceDatabase(db_path=self.db_path)
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

        # Thread-safe track candidates and results cache
        self.candidates: Dict[int, FaceTrackCandidate] = {}
        self.results_cache: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Asynchronous task queue and worker pool
        self.task_queue = queue.Queue(maxsize=128)
        self.is_running = True
        self.workers: List[threading.Thread] = []
        self._start_workers()

        logger.info(
            f"FRS Pipeline initialized (Workers: {self.num_workers}, Min Area: {self.min_box_area}, Threshold: {self.match_threshold})"
        )

    def _start_workers(self):
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"FRS-Worker-{i}", daemon=True
            )
            t.start()
            self.workers.append(t)

    def _worker_loop(self):
        """Background worker loop for face detection and biometric matching."""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None:
                break

            track_id, human_crop, bbox_area, conf, timestamp = task
            try:
                # 1. Detect best aligned face crop
                face_info = self.detector.get_best_face(human_crop)
                if face_info is None:
                    with self._lock:
                        if track_id in self.candidates:
                            self.candidates[track_id].status = "NO_FACE"
                    continue

                face_crop, det_conf = face_info

                # 2. Extract 512-d biometric embedding and quality metric
                embedding, quality = self.embedder.get_embedding(face_crop)
                if quality < 0.05:  # Extremely blurry or degraded face region
                    with self._lock:
                        if track_id in self.candidates:
                            self.candidates[track_id].status = "NO_FACE"
                    continue

                # 3. Query biometric watchlist database
                matches = self.face_db.search_face(embedding, top_k=1)
                best_match = matches[0] if matches else None
                best_conf = best_match["confidence"] if best_match else 0.0

                if best_match is not None and best_conf >= self.match_threshold:
                    # POSITIVE IDENTIFICATION
                    person_id = best_match["person_id"]
                    name = best_match["name"]
                    category = best_match["category"].upper()
                    notes = best_match.get("notes", "")
                    is_flagged = category in ["WANTED", "SUSPECT", "UNKNOWN_REPEAT"]

                    # 4. Optional Forensic Crop Storage
                    crop_file = None
                    if self.save_crops_dir and face_crop is not None and face_crop.size > 0:
                        crop_file = os.path.join(
                            self.save_crops_dir,
                            f"{person_id}_{track_id}_{int(timestamp)}.jpg",
                        )
                        cv2.imwrite(crop_file, face_crop)

                    # 5. Log to SQLite Audit Table
                    self.face_db.log_recognition(
                        track_id=track_id,
                        person_id=person_id,
                        name=name,
                        confidence=best_conf,
                        is_flagged=is_flagged,
                        category=category,
                        bbox_area=bbox_area,
                    )

                    # 6. Update Cache
                    face_data = {
                        "person_id": person_id,
                        "name": name,
                        "category": category,
                        "confidence": best_conf,
                        "is_flagged": is_flagged,
                        "notes": notes,
                        "timestamp": timestamp,
                        "quality": quality,
                    }

                    with self._lock:
                        self.results_cache[track_id] = face_data
                        if track_id in self.candidates:
                            self.candidates[track_id].status = "COMPLETED"
                            self.candidates[track_id].face_result = face_data

                    if is_flagged:
                        logger.warning(
                            f"🚨 [FRS WATCHLIST HIT] Track #{track_id} | [{category}] {name} ({person_id}) - Conf: {best_conf:.2f}"
                        )
                    else:
                        logger.info(
                            f"👤 [FRS IDENTIFIED] Track #{track_id} | [{category}] {name} - Conf: {best_conf:.2f}"
                        )
                else:
                    # UNKNOWN IDENTITY (Below match threshold)
                    face_data = {
                        "person_id": "UNKNOWN",
                        "name": "Unknown Person",
                        "category": "UNKNOWN",
                        "confidence": best_conf,
                        "is_flagged": False,
                        "notes": "",
                        "timestamp": timestamp,
                        "quality": quality,
                    }

                    with self._lock:
                        self.results_cache[track_id] = face_data
                        if track_id in self.candidates:
                            self.candidates[track_id].status = "COMPLETED"
                            self.candidates[track_id].face_result = face_data

            except Exception as e:
                logger.error(f"Error processing FRS for track #{track_id}: {e}")
            finally:
                self.task_queue.task_done()

    def process_frame(
        self,
        raw_img: np.ndarray,
        tlwhs: List[List[float]],
        track_ids: List[int],
        scores: List[float],
        current_time: Optional[float] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate active human tracks and schedule biometric inference without blocking the tracking loop.

        :param raw_img: Full video frame BGR array.
        :param tlwhs: List of [x, y, w, h] human bounding boxes.
        :param track_ids: Track IDs.
        :param scores: Detection confidences.
        :param current_time: Current timestamp.
        :return: Copy of current results cache.
        """
        if current_time is None:
            current_time = time.time()

        h_img, w_img = raw_img.shape[:2]
        active_ids = set(track_ids)

        # 1. Clean stale candidates older than 60 seconds
        with self._lock:
            stale_ids = [
                tid
                for tid, cand in self.candidates.items()
                if tid not in active_ids
                and (current_time - cand.last_update_time > 60.0)
            ]
            for tid in stale_ids:
                del self.candidates[tid]
                self.results_cache.pop(tid, None)

        # 2. Evaluate candidate crops for each active human track
        for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
            x, y, w, h = map(int, tlwh)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

            box_area = float((x2 - x1) * (y2 - y1))
            if box_area < self.min_box_area:
                continue

            quality_score = box_area * float(score)

            with self._lock:
                cand = self.candidates.get(track_id)
                if cand is None:
                    cand = FaceTrackCandidate(track_id)
                    self.candidates[track_id] = cand

                cand.last_update_time = current_time

                # Decide if we should queue this crop
                should_queue = False
                if cand.status == "UNPROCESSED":
                    cand.best_score = quality_score
                    cand.best_area = box_area
                    cand.best_crop = raw_img[y1:y2, x1:x2].copy()
                    cand.last_attempt_time = current_time
                    should_queue = True
                elif (
                    cand.status == "NO_FACE"
                    and (current_time - cand.last_attempt_time > self.recheck_interval_sec)
                ):
                    cand.best_score = quality_score
                    cand.best_area = box_area
                    cand.best_crop = raw_img[y1:y2, x1:x2].copy()
                    cand.last_attempt_time = current_time
                    should_queue = True
                elif (
                    cand.status == "COMPLETED"
                    and cand.face_result is not None
                    and cand.face_result.get("category") == "UNKNOWN"
                    and (current_time - cand.last_attempt_time > self.recheck_interval_sec)
                ):
                    cand.best_score = quality_score
                    cand.best_area = box_area
                    cand.best_crop = raw_img[y1:y2, x1:x2].copy()
                    cand.last_attempt_time = current_time
                    should_queue = True
                elif (
                    cand.status == "COMPLETED"
                    and quality_score > cand.best_score * self.quality_boost_ratio
                ):
                    cand.best_score = quality_score
                    cand.best_area = box_area
                    cand.best_crop = raw_img[y1:y2, x1:x2].copy()
                    cand.last_attempt_time = current_time
                    should_queue = True

                if should_queue and cand.best_crop is not None:
                    cand.status = "QUEUED"
                    try:
                        self.task_queue.put_nowait(
                            (
                                track_id,
                                cand.best_crop,
                                box_area,
                                float(score),
                                current_time,
                            )
                        )
                    except queue.Full:
                        pass

        with self._lock:
            return dict(self.results_cache)

    def get_track_face(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached face recognition result for a specific track ID."""
        with self._lock:
            return self.results_cache.get(track_id)

    def stop(self):
        """Cleanly terminate background FRS workers."""
        self.is_running = False
        for _ in range(self.num_workers):
            try:
                self.task_queue.put_nowait(None)
            except Exception:
                pass
        for t in self.workers:
            t.join(timeout=1.0)
        logger.info("FRS Pipeline background workers stopped.")
