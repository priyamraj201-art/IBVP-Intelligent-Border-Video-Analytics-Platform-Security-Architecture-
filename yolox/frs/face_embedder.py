import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
from typing import Tuple, Optional
from loguru import logger


class FaceEmbedder:
    """
    High-Dimensional Face Feature Extractor & Quality Analyzer.
    Extracts 512-dimensional L2-normalized embeddings via ArcFace.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        model_name: str = "w600k_r50",
    ):
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.rec_model = None

        self._init_model()

    def _init_model(self):
        """Initialize ArcFace recognition model or prepare fallback."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(name="buffalo_sc", providers=providers)
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            if hasattr(app, "models") and "recognition" in app.models:
                self.rec_model = app.models["recognition"]
                logger.info(
                    f"FaceEmbedder initialized with InsightFace ArcFace ({self.model_name})"
                )
            else:
                logger.info("FaceEmbedder initialized with InsightFace pipeline")
                self.rec_model = app
        except Exception as e:
            logger.warning(
                f"InsightFace ArcFace unavailable ({e}). Fallback embedder active."
            )

    @staticmethod
    def calculate_sharpness(face_bgr: np.ndarray) -> float:
        """
        Calculate face sharpness/clarity via normalized Laplacian variance (0.0 to 1.0).
        """
        if face_bgr is None or face_bgr.size == 0:
            return 0.0
        if len(face_bgr.shape) == 2:
            gray = face_bgr
        else:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalization scale: lap_var of 300+ is considered sharp and clear
        norm_quality = float(np.clip(lap_var / 300.0, 0.0, 1.0))
        return norm_quality

    def get_embedding(self, face_crop: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Extract a 512-dimensional L2-normalized embedding and quality score.

        :param face_crop: Face BGR image crop.
        :return: (embedding_512d, quality_score)
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros((self.embedding_dim,), dtype=np.float32), 0.0

        # Handle grayscale or BGRA
        if len(face_crop.shape) == 2:
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
        elif face_crop.shape[2] == 4:
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2BGR)
        else:
            face_bgr = face_crop

        quality_score = self.calculate_sharpness(face_bgr)

        # Standard face alignment size: 112x112
        face_resized = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_AREA)

        # 1. Primary Engine: InsightFace Model
        if self.rec_model is not None:
            try:
                # If rec_model is ArcFace model with get_feat
                if hasattr(self.rec_model, "get_feat"):
                    # Note: get_feat internally performs swapRB=True (expects BGR input)
                    emb = self.rec_model.get_feat(face_resized).flatten()
                elif hasattr(self.rec_model, "get"):
                    faces = self.rec_model.get(face_resized)
                    if faces and hasattr(faces[0], "normed_embedding"):
                        emb = faces[0].normed_embedding.flatten()
                    elif faces and hasattr(faces[0], "embedding"):
                        emb = faces[0].embedding.flatten()
                    else:
                        emb = None
                else:
                    emb = None

                if emb is not None and len(emb) == self.embedding_dim:
                    emb = np.asarray(emb, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm
                    return emb, quality_score
            except Exception as e:
                logger.error(f"InsightFace embedding extraction error: {e}")

        # 2. Deterministic Fallback Feature Vector
        # Generates a pseudo-embedding based on multi-scale spatial color/gradient histograms
        gray_112 = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray_112, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_112, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)

        # Build feature blocks
        blocks = []
        for r in range(4):
            for c in range(4):
                block_mag = mag[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28]
                block_ang = ang[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28]
                hist, _ = np.histogram(block_ang, bins=16, range=(0, 2 * np.pi), weights=block_mag)
                blocks.append(hist)

        # Spatial color distribution (16 blocks * 16 bins = 256 + 256 color = 512)
        color_hist, _ = np.histogram(face_resized, bins=256, range=(0, 256))
        raw_feat = np.concatenate([np.concatenate(blocks), color_hist]).astype(np.float32)
        if len(raw_feat) > self.embedding_dim:
            raw_feat = raw_feat[: self.embedding_dim]
        elif len(raw_feat) < self.embedding_dim:
            raw_feat = np.pad(raw_feat, (0, self.embedding_dim - len(raw_feat)))

        norm = np.linalg.norm(raw_feat)
        if norm > 1e-6:
            raw_feat = raw_feat / norm
        else:
            raw_feat = np.zeros((self.embedding_dim,), dtype=np.float32)

        return raw_feat, quality_score

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors (-1.0 to 1.0).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        v1 = emb1.flatten().astype(np.float32)
        v2 = emb2.flatten().astype(np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two embedding vectors.
        """
        if emb1 is None or emb2 is None:
            return float("inf")
        v1 = emb1.flatten().astype(np.float32)
        v2 = emb2.flatten().astype(np.float32)
        return float(np.linalg.norm(v1 - v2))
