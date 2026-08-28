import cv2
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class PlateDetector:
    """
    Two-Stage License Plate Localizer.
    Isolates license plate / paper sheet candidate regions from a detected bounding box.
    """

    def __init__(
        self,
        min_plate_aspect: float = 1.2,
        max_plate_aspect: float = 7.0,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.95,
        target_height: int = 64,
    ):
        self.min_plate_aspect = min_plate_aspect
        self.max_plate_aspect = max_plate_aspect
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.target_height = target_height

    def preprocess_plate_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Enhance plate contrast and normalize resolution for OCR input.
        """
        if crop is None or crop.size == 0:
            return crop

        # Standardize height
        h, w = crop.shape[:2]
        if h != self.target_height and h > 0:
            new_w = max(32, int(w * (self.target_height / float(h))))
            crop = cv2.resize(crop, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)

        # Contrast enhancement using CLAHE in LAB color space
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            crop = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return crop

    def extract_plate(self, vehicle_img: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Locate and extract the tight license plate / text region from a bounding box.

        :param vehicle_img: BGR image crop.
        :return: (plate_crop, (px, py, pw, ph) relative to vehicle_img)
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return vehicle_img, None

        vh, vw = vehicle_img.shape[:2]
        if vh < 20 or vw < 20:
            return vehicle_img, (0, 0, vw, vh)

        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)

        # Morphological gradient to highlight character and plate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Sobel vertical edge detection
        grad_x = cv2.Sobel(morph, cv2.CV_32F, 1, 0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        if max_val - min_val > 0:
            grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
        else:
            grad_x = grad_x.astype("uint8")

        # Gaussian blur + Otsu thresholding
        blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Close gaps between characters
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_plate_box = None
        best_score = -1.0
        vehicle_area = float(vh * vw)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0 or w <= 0:
                continue
            aspect = float(w) / float(h)
            area = float(w * h)
            area_ratio = area / vehicle_area

            # Filter candidates by aspect ratio & proportional area
            if self.min_plate_aspect <= aspect <= self.max_plate_aspect and self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                # Prefer central placement and reasonable width
                center_dist_x = abs((x + w / 2.0) - (vw / 2.0)) / (vw / 2.0)
                score = (w * h) / (1.0 + center_dist_x)
                if score > best_score:
                    best_score = score
                    best_plate_box = (x, y, w, h)

        if best_plate_box is not None:
            bx, by, bw, bh = best_plate_box
            # Add small padding margin
            pad_x = int(bw * 0.08)
            pad_y = int(bh * 0.12)
            px1 = max(0, bx - pad_x)
            py1 = max(0, by - pad_y)
            px2 = min(vw, bx + bw + pad_x)
            py2 = min(vh, by + bh + pad_y)
            plate_crop = vehicle_img[py1:py2, px1:px2]
            return self.preprocess_plate_crop(plate_crop), (px1, py1, px2 - px1, py2 - py1)

        # Return full crop so OCR DBNet can find text anywhere
        return self.preprocess_plate_crop(vehicle_img), (0, 0, vw, vh)
