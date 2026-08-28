import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ANPRVisualizer:
    """
    Renders high-visibility license plate badges and security watchlist HUD overlays.
    """

    COLOR_MAP = {
        "STOLEN": (0, 0, 230),       # Bright Red
        "WANTED": (0, 0, 230),       # Bright Red
        "SUSPICIOUS": (0, 140, 255),  # Orange
        "VIP": (0, 200, 0),          # Bright Green
        "NORMAL": (230, 180, 0),     # Cyan / Sky Blue
    }

    @staticmethod
    def draw_anpr_overlay(
        image: np.ndarray,
        tlwhs: List[List[float]],
        track_ids: List[int],
        anpr_results: Dict[int, Dict[str, Any]],
        frame_id: int = 0,
        fps: float = 0.0,
    ) -> np.ndarray:
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        total_flagged = 0
        recent_plates = []

        # 1. Draw Bounding Boxes and License Plate Badges
        for tlwh, track_id in zip(tlwhs, track_ids):
            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h

            plate_info = anpr_results.get(track_id)
            if plate_info and plate_info.get("plate_number"):
                plate = plate_info["plate_number"]
                conf = plate_info.get("confidence", 0.0)
                is_flagged = plate_info.get("is_flagged", False)
                category = plate_info.get("alert_category", "NORMAL").upper()

                if is_flagged:
                    total_flagged += 1

                badge_color = ANPRVisualizer.COLOR_MAP.get(category, (230, 180, 0))
                recent_plates.append((plate, category, badge_color))

                # Highlight bounding box with watchlist color
                box_thickness = 3 if is_flagged else 2
                cv2.rectangle(im, (x1, y1), (x2, y2), badge_color, box_thickness)

                # Plate Badge text
                if is_flagged:
                    badge_text = f"🚨 [{category}] {plate} ({conf:.2f})"
                else:
                    badge_text = f"🚗 {plate} ({conf:.2f})"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, 2)

                # Badge background
                by1 = max(0, y1 - th - 10)
                by2 = y1
                cv2.rectangle(im, (x1, by1), (x1 + tw + 10, by2), badge_color, -1)
                text_color = (255, 255, 255)
                cv2.putText(im, badge_text, (x1 + 5, by2 - 5), font, font_scale, text_color, 2, cv2.LINE_AA)

        # 2. Render ANPR Security Dashboard HUD
        hud_h = 50
        overlay = im.copy()
        cv2.rectangle(overlay, (0, 0), (im_w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, im, 0.15, 0, im)

        # Status Title
        if total_flagged > 0:
            status_text = f"ANPR STATUS: 🚨 {total_flagged} WATCHLIST ALERT(S)"
            status_color = (0, 0, 255)
        else:
            status_text = "ANPR WATCHLIST: ALL CLEAR"
            status_color = (0, 220, 0)

        cv2.putText(im, status_text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        # Stats info (Right aligned)
        stats_text = f"Frame: {frame_id} | FPS: {fps:.1f} | Vehicles: {len(tlwhs)} | Recognized: {len(anpr_results)}"
        (stw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(im, stats_text, (im_w - stw - 15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # 3. Bottom Feed for Recently Recognized Plates
        if recent_plates:
            bottom_h = 32
            bot_overlay = im.copy()
            cv2.rectangle(bot_overlay, (0, im_h - bottom_h), (im_w, im_h), (15, 15, 15), -1)
            cv2.addWeighted(bot_overlay, 0.85, im, 0.15, 0, im)

            feed_x = 15
            for plate, cat, col in recent_plates[-4:]:
                p_text = f"[{cat}] {plate}"
                cv2.putText(im, p_text, (feed_x, im_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                (ptw, _), _ = cv2.getTextSize(p_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                feed_x += ptw + 20

        return im
