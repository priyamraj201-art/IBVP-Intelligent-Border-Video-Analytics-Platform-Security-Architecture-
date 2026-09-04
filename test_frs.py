import cv2
import numpy as np
from yolox.frs.face_detector import FaceDetector
from yolox.frs.face_embedder import FaceEmbedder

detector = FaceDetector()
embedder = FaceEmbedder()
print("Models loaded successfully")
