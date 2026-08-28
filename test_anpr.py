import os
import sys
import time
import numpy as np
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add ByteTrack root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolox.anpr.watchlist_db import WatchlistDB
from yolox.anpr.plate_detector import PlateDetector
from yolox.anpr.ocr_engine import OCREngine
from yolox.anpr.anpr_pipeline import ANPRPipeline
from yolox.anpr.anpr_visualizer import ANPRVisualizer


def test_watchlist_db():
    print("\n--- Testing WatchlistDB ---")
    test_db_path = "test_watchlist.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = WatchlistDB(db_path=test_db_path)
    db.seed_sample_watchlist()

    # Test exact lookup
    hit = db.lookup_plate("MH12AB1234")
    assert hit is not None, "Failed to find seeded plate MH12AB1234"
    assert hit["alert_category"] == "STOLEN"
    print(f"[OK] Found seeded plate: {hit['plate_number']} ({hit['alert_category']})")

    # Test formatting normalization in lookup
    hit2 = db.lookup_plate("mh-12 ab 1234")
    assert hit2 is not None, "Failed to find plate with spaces and dashes"
    print("[OK] Normalization lookup passed")

    # Test add custom plate
    db.add_watchlist_entry("TEST9999", alert_category="WANTED", owner_name="Agent Smith", notes="High priority test")
    hit3 = db.lookup_plate("TEST9999")
    assert hit3 is not None and hit3["alert_category"] == "WANTED"
    print("[OK] Custom plate addition passed")

    # Test log detection
    log_id = db.log_detection(
        track_id=1,
        plate_number="MH12AB1234",
        confidence=0.95,
        is_flagged=True,
        alert_category="STOLEN",
        bbox_area=5000.0
    )
    assert log_id > 0, "Failed to log detection"
    logs = db.get_recent_logs(limit=5)
    assert len(logs) >= 1
    print(f"[OK] Detection logging passed (Log ID: {log_id})")

    # Clean up test DB
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass
    print("[OK] WatchlistDB all tests passed!")


def test_plate_detector_and_ocr():
    print("\n--- Testing PlateDetector & OCREngine ---")
    detector = PlateDetector()
    ocr = OCREngine()

    # Create synthetic vehicle crop (e.g. 200x300 image with a white plate rectangle)
    vehicle_img = np.full((200, 300, 3), 50, dtype=np.uint8)
    # Draw license plate in bottom area
    cv2.rectangle(vehicle_img, (75, 130), (225, 175), (240, 240, 240), -1)
    cv2.putText(vehicle_img, "KA01MJ5678", (85, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    plate_crop, plate_box = detector.extract_plate(vehicle_img)
    assert plate_crop is not None and plate_crop.size > 0, "Plate crop extraction failed"
    assert plate_box is not None, "Plate box was None"
    print(f"[OK] Extracted plate crop shape: {plate_crop.shape}, box: {plate_box}")

    # Test text normalization
    assert OCREngine.normalize_plate_text("ka-01 mj-5678") == "KA01MJ5678"
    assert OCREngine.normalize_plate_text("  dl#3.ca 9999 ") == "DL3CA9999"
    assert OCREngine.normalize_plate_text("ab") == ""  # too short
    print("[OK] Text normalization passed")


def test_anpr_pipeline_caching():
    print("\n--- Testing ANPR Pipeline & Best-Crop Caching ---")
    test_db_path = "test_pipeline.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    pipeline = ANPRPipeline(db_path=test_db_path, min_box_area=500.0, num_workers=1)
    pipeline.watchlist_db.seed_sample_watchlist()

    # Create dummy frame
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (200, 200, 200), -1)

    # Frame 1: Track ID 10 with area = 200*200 = 40000, conf = 0.8
    tlwhs_1 = [[100.0, 100.0, 200.0, 200.0]]
    ids_1 = [10]
    scores_1 = [0.8]

    pipeline.process_frame(frame, tlwhs_1, ids_1, scores_1)
    
    # Wait briefly for worker queue
    time.sleep(0.5)

    assert 10 in pipeline.candidates, "Track candidate 10 not registered"
    initial_score = pipeline.candidates[10].best_score
    print(f"[OK] Initial best score for track #10: {initial_score}")

    # Frame 2: Same Track ID 10 with slightly smaller area (should NOT trigger re-OCR)
    tlwhs_2 = [[100.0, 100.0, 190.0, 190.0]]
    pipeline.process_frame(frame, tlwhs_2, ids_1, [0.8])
    time.sleep(0.2)
    assert pipeline.candidates[10].best_score == initial_score, "Score should not have downgraded"
    print("[OK] Best-crop caching preserved original best crop")

    # Frame 3: Significantly larger area (+60% area, e.g. vehicle got much closer)
    tlwhs_3 = [[100.0, 100.0, 300.0, 300.0]]
    pipeline.process_frame(frame, tlwhs_3, ids_1, [0.9])
    time.sleep(0.3)
    assert pipeline.candidates[10].best_score > initial_score, "Score should have updated for high-res view"
    print(f"[OK] Quality boost successfully updated track candidate (new score: {pipeline.candidates[10].best_score})")

    # Test Visualizer rendering
    annotated = ANPRVisualizer.draw_anpr_overlay(
        frame, tlwhs_3, ids_1, pipeline.results_cache, frame_id=1, fps=30.0
    )
    assert annotated is not None and annotated.shape == frame.shape
    print("[OK] ANPR Visualizer rendered overlay successfully")

    pipeline.stop()
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass
    print("[OK] ANPR Pipeline all tests passed!")


if __name__ == "__main__":
    test_watchlist_db()
    test_plate_detector_and_ocr()
    test_anpr_pipeline_caching()
    print("\n==========================================")
    print("ALL ANPR VERIFICATION TESTS COMPLETED!")
    print("==========================================")
