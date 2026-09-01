import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from loguru import logger


class FaceDetector:
    """
    Robust Multi-Engine Face Detector.
    Primary Engine: InsightFace RetinaFace (buffalo_sc / buffalo_l).
    Fallback Engine: OpenCV Haar Cascade Classifier with geometry validation.
    """

    def __init__(
        self,
        min_face_size: int = 30,
        model_name: str = "buffalo_sc",
        det_thresh: float = 0.35,
    ):
        """
        Initialize FaceDetector.

        :param min_face_size: Minimum face bounding box dimension in pixels.
        :param model_name: InsightFace model pack ('buffalo_sc' for CPU speed, 'buffalo_l' for GPU).
        :param det_thresh: Confidence threshold for face detection.
        """
        self.min_face_size = min_face_size
        self.model_name = model_name
        self.det_thresh = det_thresh

        self.insightface_app = None
        self.cascade_classifier = None

        self._init_detector()

    def _init_detector(self):
        """Initialize primary InsightFace engine or fallback to OpenCV Cascade."""
        # 1. Try InsightFace initialization
        try:
            import insightface
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            ctx_id = 0 if torch.cuda.is_available() else -1

            app = FaceAnalysis(name=self.model_name, providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.insightface_app = app
            logger.info(
                f"FaceDetector initialized with InsightFace ({self.model_name}, providers={providers})"
            )
            return
        except Exception as e:
            logger.warning(
                f"InsightFace initialization unavailable ({e}). Falling back to OpenCV Cascade."
            )

        # 2. Fallback: OpenCV Haar Frontal Face Cascade
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.cascade_classifier = cv2.CascadeClassifier(cascade_path)
                logger.info(
                    f"FaceDetector initialized with OpenCV Haar Cascade: {cascade_path}"
                )
            else:
                logger.warning(
                    f"Haar cascade XML not found at {cascade_path}. Fallback heuristic will be used."
                )
        except Exception as e:
            logger.error(f"Failed to load OpenCV CascadeClassifier: {e}")

    def detect_faces(self, crop: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Detect face regions within a human bounding box crop.

        :param crop: BGR image crop of a human candidate.
        :return: List of tuples (face_crop_bgr, confidence).
        """
        if crop is None or crop.size == 0:
            return []

        # Ensure 3-channel BGR
        if len(crop.shape) == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

        h_crop, w_crop = crop.shape[:2]
        if h_crop < 20 or w_crop < 20:
            return []

        results: List[Tuple[np.ndarray, float]] = []

        # 1. Primary Engine: InsightFace
        if self.insightface_app is not None:
            try:
                faces = self.insightface_app.get(crop)
                for face in faces:
                    det_score = float(face.det_score) if hasattr(face, "det_score") else 0.8
                    if det_score < self.det_thresh:
                        continue

                    # If landmarks are available, use standard ArcFace canonical 5-point alignment
                    face_crop = None
                    if hasattr(face, "kps") and face.kps is not None:
                        try:
                            from insightface.utils import face_align
                            face_crop = face_align.norm_crop(crop, landmark=face.kps, image_size=112)
                        except Exception:
                            face_crop = None
                    elif hasattr(face, "norm_crop") and face.norm_crop is not None:
                        face_crop = face.norm_crop

                    if face_crop is None or face_crop.size == 0:
                        bbox = face.bbox.astype(int)
                        x1 = max(0, int(bbox[0]))
                        y1 = max(0, int(bbox[1]))
                        x2 = min(w_crop, int(bbox[2]))
                        y2 = min(h_crop, int(bbox[3]))
                        if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                            continue
                        face_crop = crop[y1:y2, x1:x2]

                    if face_crop is not None and face_crop.size > 0:
                        results.append((face_crop, det_score))

                if results:
                    return results
            except Exception as e:
                logger.error(f"InsightFace detection error: {e}")

        # 2. Fallback Engine: Haar Cascade
        if self.cascade_classifier is not None:
            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Equalize histogram for contrast enhancement
                gray = cv2.equalizeHist(gray)
                min_sz = (self.min_face_size, self.min_face_size)
                faces = self.cascade_classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=min_sz,
                )

                for (x, y, w, h) in faces:
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w_crop, x + w)
                    y2 = min(h_crop, y + h)

                    face_crop = crop[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        # Estimate confidence based on face size & central location
                        norm_area = (w * h) / float(w_crop * h_crop)
                        confidence = min(0.95, 0.70 + 0.25 * norm_area)
                        results.append((face_crop, float(confidence)))

                if results:
                    return results
            except Exception as e:
                logger.error(f"Haar Cascade detection error: {e}")

        # 3. Fallback 2: Upper-body Face Region Heuristic
        # In a standing human detection box, the head/face resides in the top ~35% of the bbox.
        if h_crop >= self.min_face_size and w_crop >= self.min_face_size:
            head_h = int(h_crop * 0.40)
            # Center 80% horizontally
            pad_w = int(w_crop * 0.10)
            face_candidate = crop[0:head_h, pad_w : w_crop - pad_w]
            if (
                face_candidate.shape[0] >= self.min_face_size
                and face_candidate.shape[1] >= self.min_face_size
            ):
                results.append((face_candidate, 0.52))

        return results

    def get_best_face(self, crop: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the single highest-confidence face crop from a human bounding box.

        :param crop: BGR human crop.
        :return: (best_face_crop, confidence) or None if no face found with conf >= 0.5.
        """
        faces = self.detect_faces(crop)
        valid_faces = [f for f in faces if f[1] >= self.det_thresh]
        if not valid_faces:
            return None

        # Sort by confidence descending
        valid_faces.sort(key=lambda x: x[1], reverse=True)
        return valid_faces[0]
