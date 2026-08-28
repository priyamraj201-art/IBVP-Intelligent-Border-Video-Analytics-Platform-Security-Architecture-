import time
import math
import collections
import threading
import cv2
import numpy as np
from loguru import logger

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlertLevel:
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MotionAlertSystem:
    def __init__(
        self,
        low_thresh=20.0,
        med_thresh=60.0,
        high_thresh=120.0,
        history_len=20,
        enable_sound=False,
        cooldown_sec=1.5,
    ):
        """
        Movement Alert System for ByteTrack.

        :param low_thresh: Speed threshold in px/sec for LOW alert (slow movement / walking).
        :param med_thresh: Speed threshold in px/sec for MEDIUM alert (moderate / active motion).
        :param high_thresh: Speed threshold in px/sec for HIGH alert (rapid movement / running / sudden intrusion).
        :param history_len: Number of historical frames to smooth speed calculation.
        :param enable_sound: Whether to emit an audio beep on HIGH alerts.
        :param cooldown_sec: Minimum seconds between consecutive terminal / audio alerts.
        """
        self.low_thresh = low_thresh
        self.med_thresh = med_thresh
        self.high_thresh = high_thresh
        self.history_len = history_len
        self.enable_sound = enable_sound
        self.cooldown_sec = cooldown_sec

        # track_id -> deque of (timestamp, (cx, cy))
        self.track_history = collections.defaultdict(lambda: collections.deque(maxlen=self.history_len))
        # track_id -> last alert level
        self.track_alerts = {}
        # track_id -> smoothed speed
        self.track_speeds = {}
        # Timestamp of last audio alert
        self.last_sound_time = 0

    def update(self, tlwhs, track_ids, current_time=None):
        """
        Update motion history and compute alert levels for all active tracks.

        :param tlwhs: list of [x, y, w, h] boxes
        :param track_ids: list of int track IDs
        :param current_time: current timestamp (float)
        :return: dict mapping track_id -> {'level': str, 'speed': float, 'color': (B, G, R), 'trail': [(x, y), ...]}
        """
        if current_time is None:
            current_time = time.time()

        active_ids = set(track_ids)
        # Clean up stale track histories
        stale_ids = [tid for tid in self.track_history if tid not in active_ids]
        for tid in stale_ids:
            del self.track_history[tid]
            self.track_alerts.pop(tid, None)
            self.track_speeds.pop(tid, None)

        results = {}
        system_highest_level = AlertLevel.NORMAL

        for i, (tlwh, track_id) in enumerate(zip(tlwhs, track_ids)):
            x, y, w, h = tlwh
            cx = x + w / 2.0
            cy = y + h / 2.0

            history = self.track_history[track_id]
            history.append((current_time, (cx, cy)))

            speed = 0.0
            if len(history) >= 3:
                # Compute displacement over available history window
                t_first, (x_first, y_first) = history[0]
                t_last, (x_last, y_last) = history[-1]
                dt = max(1e-3, t_last - t_first)
                dist = math.hypot(x_last - x_first, y_last - y_first)
                speed = dist / dt

            # Smooth speed
            prev_speed = self.track_speeds.get(track_id, speed)
            smoothed_speed = 0.7 * speed + 0.3 * prev_speed
            self.track_speeds[track_id] = smoothed_speed

            # Determine alert level
            if smoothed_speed >= self.high_thresh:
                level = AlertLevel.HIGH
                color = (0, 0, 255)       # Red (BGR)
            elif smoothed_speed >= self.med_thresh:
                level = AlertLevel.MEDIUM
                color = (0, 165, 255)     # Orange (BGR)
            elif smoothed_speed >= self.low_thresh:
                level = AlertLevel.LOW
                color = (0, 255, 255)     # Yellow (BGR)
            else:
                level = AlertLevel.NORMAL
                color = (0, 220, 0)       # Green (BGR)

            # Check if alert level escalated
            old_level = self.track_alerts.get(track_id, AlertLevel.NORMAL)
            if level != old_level:
                self.track_alerts[track_id] = level
                if level != AlertLevel.NORMAL:
                    logger.warning(
                        f"⚠️ [ALERT: {level}] Target #{track_id} motion detected! Speed: {smoothed_speed:.1f} px/s"
                    )

            if level in (AlertLevel.MEDIUM, AlertLevel.HIGH) and self.enable_sound and HAS_WINSOUND:
                if current_time - self.last_sound_time > self.cooldown_sec:
                    self.last_sound_time = current_time
                    freq = 1400 if level == AlertLevel.HIGH else 900
                    dur = 180 if level == AlertLevel.HIGH else 100
                    try:
                        threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()
                    except Exception:
                        pass

            trail_pts = [pt for _, pt in history]
            results[track_id] = {
                "level": level,
                "speed": smoothed_speed,
                "color": color,
                "trail": trail_pts,
            }

        return results

    def draw_alerts(self, image, tlwhs, track_ids, alert_data, frame_id=0, fps=0.0):
        """
        Render motion trajectories, alert badges, and a top HUD banner on the image.
        """
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        highest_level = AlertLevel.NORMAL
        highest_color = (0, 200, 0)

        for i, (tlwh, track_id) in enumerate(zip(tlwhs, track_ids)):
            info = alert_data.get(track_id, {
                "level": AlertLevel.NORMAL,
                "speed": 0.0,
                "color": (0, 220, 0),
                "trail": [],
            })
            level = info["level"]
            speed = info["speed"]
            color = info["color"]
            trail = info["trail"]

            if level == AlertLevel.HIGH:
                highest_level = AlertLevel.HIGH
                highest_color = (0, 0, 255)
            elif level == AlertLevel.MEDIUM and highest_level != AlertLevel.HIGH:
                highest_level = AlertLevel.MEDIUM
                highest_color = (0, 165, 255)
            elif level == AlertLevel.LOW and highest_level == AlertLevel.NORMAL:
                highest_level = AlertLevel.LOW
                highest_color = (0, 255, 255)

            # 1. Draw motion trajectory trail
            if len(trail) >= 2:
                for idx in range(1, len(trail)):
                    alpha = idx / len(trail)
                    thickness = max(1, int(3 * alpha))
                    pt1 = (int(trail[idx - 1][0]), int(trail[idx - 1][1]))
                    pt2 = (int(trail[idx][0]), int(trail[idx][1]))
                    cv2.line(im, pt1, pt2, color, thickness)

            # 2. Draw bounding box with alert level color
            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h
            box_thickness = 3 if level == AlertLevel.HIGH else 2
            cv2.rectangle(im, (x1, y1), (x2, y2), color, box_thickness)

            # 3. Draw alert badge above box
            badge_text = f"ID:{track_id} | {level} ({speed:.0f} px/s)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, 1)

            # Background for badge
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            cv2.rectangle(im, (x1, badge_y1), (x1 + tw + 8, badge_y2), color, -1)
            text_color = (0, 0, 0) if level in (AlertLevel.LOW, AlertLevel.MEDIUM) else (255, 255, 255)
            cv2.putText(im, badge_text, (x1 + 4, badge_y2 - 4), font, font_scale, text_color, 1, cv2.LINE_AA)

        # 4. Top HUD Banner Overlay
        hud_h = 45
        overlay = im.copy()
        cv2.rectangle(overlay, (0, 0), (im_w, hud_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, im, 0.25, 0, im)

        # Draw status pill / badge
        status_text = f"SYSTEM STATUS: {highest_level} ALERT" if highest_level != AlertLevel.NORMAL else "SYSTEM STATUS: NORMAL"
        cv2.putText(im, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, highest_color, 2, cv2.LINE_AA)

        stats_text = f"Frame: {frame_id} | FPS: {fps:.1f} | Objects: {len(tlwhs)}"
        (stw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(im, stats_text, (im_w - stw - 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        return im
