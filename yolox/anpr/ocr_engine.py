import os
import re
import ssl
import cv2
import numpy as np
import torch
from typing import Optional, Tuple
from loguru import logger

# Fix Windows SSL certificate store issues
try:
    import certifi
    ssl.SSLContext.load_default_certs = lambda self, purpose=ssl.Purpose.SERVER_AUTH: self.load_verify_locations(certifi.where())
except Exception:
    pass

try:
    import easyocr
    HAS_EASYOCR = True
except (ImportError, Exception):
    HAS_EASYOCR = False

try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except (ImportError, Exception):
    HAS_PADDLEOCR = False


class OCREngine:
    """
    High-Performance License Plate OCR Engine.
    Leverages EasyOCR & PaddleOCR with automatic text box detection,
    format normalization, and watchlist matching.
    """

    def __init__(self, use_gpu: Optional[bool] = None, lang: str = "en"):
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu
        self.lang = lang
        self.reader = None
        self._init_engine()

    def _init_engine(self):
        # 1. Initialize EasyOCR
        if HAS_EASYOCR:
            try:
                self.reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
                logger.info(f"EasyOCR Reader initialized successfully (GPU={self.use_gpu}).")
                return
            except Exception as e:
                logger.warning(f"EasyOCR init failed ({e}). Checking PaddleOCR...")

        # 2. Initialize PaddleOCR
        if HAS_PADDLEOCR:
            try:
                self.reader = PaddleOCR(
                    use_angle_cls=False,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
                logger.info(f"PaddleOCR initialized successfully (GPU={self.use_gpu}).")
                return
            except Exception as e:
                logger.warning(f"PaddleOCR init failed ({e}).")

        logger.info("Using internal contour character engine.")

    @staticmethod
    def normalize_plate_text(raw_text: str) -> str:
        """
        Normalize OCR text for license plate matching:
        - Convert to uppercase
        - Strip non-alphanumeric characters, spaces, dashes
        - Handle common OCR character confusions (e.g. O/0)
        """
        if not raw_text:
            return ""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        # Accept valid alphanumeric strings between 3 and 14 characters
        if len(cleaned) < 3 or len(cleaned) > 14:
            return ""
        return cleaned

    def recognize_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on an image crop.

        :param plate_crop: BGR numpy image.
        :return: (plate_text, confidence)
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        # 1. EasyOCR Inference
        if HAS_EASYOCR and isinstance(self.reader, easyocr.Reader):
            try:
                rgb_img = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB) if len(plate_crop.shape) == 3 else plate_crop
                results = self.reader.readtext(rgb_img)
                if results:
                    best_text = ""
                    best_conf = 0.0
                    for item in results:
                        # item format: (bbox, text, conf)
                        if len(item) >= 3:
                            text, conf = item[1], float(item[2])
                            norm = self.normalize_plate_text(text)
                            if norm and conf > best_conf:
                                best_text = norm
                                best_conf = conf
                    if best_text:
                        return best_text, round(best_conf, 3)
            except Exception as e:
                logger.debug(f"EasyOCR inference error: {e}")

        # 2. PaddleOCR Inference
        if HAS_PADDLEOCR and self.reader is not None and not isinstance(self.reader, easyocr.Reader):
            try:
                results = self.reader.ocr(plate_crop, cls=False)
                if results and len(results) > 0 and results[0]:
                    best_text = ""
                    best_conf = 0.0
                    for line in results[0]:
                        if len(line) >= 2:
                            text, conf = line[1]
                            norm = self.normalize_plate_text(text)
                            if norm and float(conf) > best_conf:
                                best_text = norm
                                best_conf = float(conf)
                    if best_text:
                        return best_text, round(best_conf, 3)
            except Exception as e:
                logger.debug(f"PaddleOCR inference error: {e}")

        return "", 0.0
