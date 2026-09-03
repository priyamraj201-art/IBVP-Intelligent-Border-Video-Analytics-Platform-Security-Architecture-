import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class FRSVisualizer:
    """
    Renders high-visibility Facial Recognition HUD overlays, category badges, and watchlist alerts.
    """

    COLOR_MAP = {
        "WANTED": (0, 0, 230),        # Bright Red
        "SUSPECT": (0, 100, 255),     # Orange-Red
        "VIP": (0, 200, 0),           # Bright Green
        "STAFF": (200, 200, 0),       # Cyan / Teal
        "UNKNOWN": (180, 180, 180),   # Muted Grey
        "UNKNOWN_REPEAT": (0, 140, 255), # Vivid Orange
    }

    @staticmethod
    def draw_frs_overlay(
        image: np.ndarray,
        tlwhs: List[List[float]],
        track_ids: List[int],
        frs_results: Optional[Dict[int, Dict[str, Any]]],
        frame_id: int = 0,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw bounding boxes, facial recognition badges, HUD status bar, and forensic ticker.

        :param image: Input BGR image.
        :param tlwhs: List of [x, y, w, h] bounding boxes.
        :param track_ids: List of integer track IDs.
        :param frs_results: Mapping of track_id -> face recognition result dict.
        :param frame_id: Current frame number.
        :param fps: Processing frames per second.
        :return: Annotated image array.
        """
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        results = frs_results or {}
        total_flagged = 0
        total_identified = 0
        recent_matches: List[Tuple[str, str, Tuple[int, int, int]]] = []

        # 1. Draw Bounding Boxes and Identity Badges
        for tlwh, track_id in zip(tlwhs, track_ids):
            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h

            face_info = results.get(track_id)

            if face_info is not None:
                person_id = face_info.get("person_id", "UNKNOWN")
                name = face_info.get("name", "Unknown Person")
                conf = face_info.get("confidence", 0.0)
                category = face_info.get("category", "UNKNOWN").upper()
                is_flagged = face_info.get("is_flagged", False)

                if is_flagged:
                    total_flagged += 1

                if person_id != "UNKNOWN":
                    total_identified += 1
                    badge_color = FRSVisualizer.COLOR_MAP.get(category, (0, 200, 0))
                    recent_matches.append((name, category, badge_color))

                    # Flagged / Known person badge
                    if is_flagged:
                        badge_text = f"[{category}] {name} ({conf:.2f})"
                        box_thickness = 3
                    else:
                        badge_text = f"[{category}] {name} ({conf:.2f})"
                        box_thickness = 2
                else:
                    # Unknown individual
                    badge_color = FRSVisualizer.COLOR_MAP["UNKNOWN"]
                    badge_text = f"UNKNOWN ({conf:.2f})"
                    box_thickness = 1

                # Draw bounding box
                cv2.rectangle(im, (x1, y1), (x2, y2), badge_color, box_thickness)

                # Draw identity badge
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.52
                (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, 1)

                by1 = max(0, y1 - th - 10)
                by2 = y1
                cv2.rectangle(im, (x1, by1), (x1 + tw + 10, by2), badge_color, -1)
                text_color = (255, 255, 255) if is_flagged or category in ("WANTED", "SUSPECT") else (0, 0, 0)
                cv2.putText(im, badge_text, (x1 + 5, by2 - 4), font, font_scale, text_color, 1, cv2.LINE_AA)

            else:
                # Target currently scanning / unprocessed
                badge_color = (200, 200, 200)
                cv2.rectangle(im, (x1, y1), (x2, y2), badge_color, 1)
                scan_text = f"HUMAN #{track_id} [Scanning...]"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(scan_text, font, 0.45, 1)
                by1 = max(0, y1 - th - 8)
                cv2.rectangle(im, (x1, by1), (x1 + tw + 8, y1), (40, 40, 40), -1)
                cv2.putText(im, scan_text, (x1 + 4, y1 - 4), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # HUD and Bottom Ticker removed per user request

        return im
