import os
import sys

# Prevent OpenMP runtime collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add ByteTrack root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolox.frs.face_database import FaceDatabase
from yolox.frs.face_detector import FaceDetector
from yolox.frs.face_embedder import FaceEmbedder
from yolox.frs.frs_pipeline import FRSPipeline
from yolox.frs.frs_visualizer import FRSVisualizer


def test_face_database():
    print("\n--- Testing FaceDatabase ---")
    test_db_path = "test_frs.db"
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

    db = FaceDatabase(db_path=test_db_path)
    db.seed_sample_identities()

    # Test get all identities
    all_identities = db.get_all_identities()
    assert len(all_identities) == 5, f"Expected 5 seeded identities, got {len(all_identities)}"
    print(f"[OK] Seeded {len(all_identities)} facial identities successfully")

    # Test enrolling a specific test identity
    np.random.seed(12345)
    test_emb = np.random.randn(512).astype(np.float32)
    test_emb = test_emb / np.linalg.norm(test_emb)

    ok = db.enroll_face(
        person_id="TEST_SUSPECT_99",
        name="Alexander Pierce",
        embedding=test_emb,
        category="WANTED",
        notes="High security test subject",
    )
    assert ok, "Failed to enroll TEST_SUSPECT_99"

    # Search with identical vector (cosine similarity should be ~1.0)
    matches = db.search_face(test_emb, top_k=1)
    assert len(matches) > 0, "No matches returned for exact vector"
    assert matches[0]["person_id"] == "TEST_SUSPECT_99", f"Expected TEST_SUSPECT_99, got {matches[0]['person_id']}"
    assert matches[0]["confidence"] > 0.98, f"Expected high cosine similarity, got {matches[0]['confidence']}"
    print(f"[OK] Biometric search matched {matches[0]['person_id']} (Conf: {matches[0]['confidence']:.4f})")

    # Test incremental multi-observation refinement
    slightly_noisy_emb = test_emb + np.random.normal(0, 0.05, 512).astype(np.float32)
    slightly_noisy_emb = slightly_noisy_emb / np.linalg.norm(slightly_noisy_emb)
    ok_refine = db.enroll_face(
        person_id="TEST_SUSPECT_99",
        name="Alexander Pierce",
        embedding=slightly_noisy_emb,
        category="WANTED",
        notes="Updated observation",
    )
    assert ok_refine, "Failed to refine TEST_SUSPECT_99"
    all_after_refine = db.get_all_identities()
    target_row = [r for r in all_after_refine if r["person_id"] == "TEST_SUSPECT_99"][0]
    assert target_row["face_count"] == 2, f"Expected face_count 2, got {target_row['face_count']}"
    print(f"[OK] Incremental refinement updated face_count to {target_row['face_count']}")

    # Test recognition audit logging
    log_id = db.log_recognition(
        track_id=42,
        person_id="TEST_SUSPECT_99",
        name="Alexander Pierce",
        confidence=0.97,
        is_flagged=True,
        category="WANTED",
        camera_id=1,
        bbox_area=6400.0,
    )
    assert log_id > 0, "Failed to write recognition log"
    recent_logs = db.get_recent_logs(limit=5)
    assert len(recent_logs) >= 1
    assert recent_logs[0]["person_id"] == "TEST_SUSPECT_99"
    print(f"[OK] Recognition audit logged successfully (Log ID: {log_id})")

    # Test identity removal
    removed = db.remove_identity("TEST_SUSPECT_99")
    assert removed, "Failed to remove TEST_SUSPECT_99"
    assert len(db.get_all_identities()) == 5
    print("[OK] Identity removal passed")

    # Clean up test database
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass
    print("[OK] FaceDatabase all tests passed!")


def test_face_detector_and_embedder():
    print("\n--- Testing FaceDetector & FaceEmbedder ---")
    detector = FaceDetector()
    embedder = FaceEmbedder()

    # Create synthetic human crop (200x300 image with face-like geometry in upper portion)
    human_crop = np.full((300, 200, 3), 40, dtype=np.uint8)
    # Draw head / face oval in upper region
    cv2.ellipse(human_crop, (100, 70), (45, 60), 0, 0, 360, (190, 160, 140), -1)
    # Draw eyes and mouth for visual texture / sharpness
    cv2.circle(human_crop, (80, 60), 6, (40, 40, 40), -1)
    cv2.circle(human_crop, (120, 60), 6, (40, 40, 40), -1)
    cv2.ellipse(human_crop, (100, 95), (20, 10), 0, 0, 180, (50, 50, 180), -1)

    # 1. Test Face Detection
    faces = detector.detect_faces(human_crop)
    assert len(faces) > 0, "Failed to detect face in synthetic human crop"
    best_face = detector.get_best_face(human_crop)
    assert best_face is not None, "get_best_face returned None"
    face_crop, det_conf = best_face
    assert face_crop.size > 0
    print(f"[OK] Face detected: crop shape {face_crop.shape}, confidence: {det_conf:.2f}")

    # 2. Test Face Embedding
    emb, quality = embedder.get_embedding(face_crop)
    assert emb.shape == (512,), f"Expected shape (512,), got {emb.shape}"
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-3, f"Embedding is not unit-normalized: norm={norm}"
    assert quality > 0.0, "Quality score should be positive"
    print(f"[OK] Extracted 512-d unit embedding (L2 Norm: {norm:.4f}, Quality: {quality:.4f})")

    # 3. Test Static Similarity Math
    emb_a = emb.copy()
    emb_b = emb.copy()
    cos_sim_identical = FaceEmbedder.cosine_similarity(emb_a, emb_b)
    assert abs(cos_sim_identical - 1.0) < 1e-4, f"Expected 1.0 for identical embeddings, got {cos_sim_identical}"

    emb_c = -emb_a
    cos_sim_opposite = FaceEmbedder.cosine_similarity(emb_a, emb_c)
    assert abs(cos_sim_opposite - (-1.0)) < 1e-4, f"Expected -1.0 for opposite embeddings, got {cos_sim_opposite}"

    dist_identical = FaceEmbedder.euclidean_distance(emb_a, emb_b)
    assert dist_identical < 1e-4, f"Expected 0.0 euclidean distance, got {dist_identical}"
    print("[OK] Cosine similarity and Euclidean distance math passed!")


def test_frs_pipeline_caching():
    print("\n--- Testing FRS Pipeline & Best-Crop Caching ---")
    test_db_path = "test_frs_pipeline.db"
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass

    pipeline = FRSPipeline(
        db_path=test_db_path,
        min_box_area=1000.0,
        quality_boost_ratio=1.3,
        num_workers=1,
    )
    pipeline.face_db.seed_sample_identities()

    # Create dummy video frame with a human box containing face geometry
    frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 550), (80, 80, 80), -1)  # Human body
    cv2.ellipse(frame, (300, 230), (40, 50), 0, 0, 360, (200, 170, 150), -1)  # Face
    cv2.circle(frame, (285, 220), 5, (20, 20, 20), -1)
    cv2.circle(frame, (315, 220), 5, (20, 20, 20), -1)

    # Frame 1: Track ID 5 with area = 200 * 400 = 80000, conf = 0.85
    tlwhs_1 = [[200.0, 150.0, 200.0, 400.0]]
    ids_1 = [5]
    scores_1 = [0.85]

    pipeline.process_frame(frame, tlwhs_1, ids_1, scores_1)
    time.sleep(0.5)

    assert 5 in pipeline.candidates, "Track candidate 5 not registered in FRS pipeline"
    initial_score = pipeline.candidates[5].best_score
    print(f"[OK] Initial best score for track #5: {initial_score}")

    # Frame 2: Same Track ID 5 with slightly smaller area (should NOT trigger re-evaluation)
    tlwhs_2 = [[200.0, 150.0, 190.0, 380.0]]
    pipeline.process_frame(frame, tlwhs_2, ids_1, [0.85])
    time.sleep(0.2)
    assert pipeline.candidates[5].best_score == initial_score, "Candidate score should not have downgraded"
    print("[OK] Best-crop caching preserved optimal initial crop")

    # Frame 3: Significantly larger area (+60% area increase, target moved closer)
    tlwhs_3 = [[200.0, 150.0, 260.0, 500.0]]
    pipeline.process_frame(frame, tlwhs_3, ids_1, [0.90])
    time.sleep(0.3)
    assert pipeline.candidates[5].best_score > initial_score, "Candidate score should update for high-resolution view"
    print(f"[OK] Quality boost successfully updated track candidate (new score: {pipeline.candidates[5].best_score})")

    # Test Visualizer rendering
    annotated = FRSVisualizer.draw_frs_overlay(
        frame, tlwhs_3, ids_1, pipeline.results_cache, frame_id=1, fps=30.0
    )
    assert annotated is not None and annotated.shape == frame.shape
    print("[OK] FRS Visualizer rendered overlay successfully")

    pipeline.stop()
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass
    print("[OK] FRS Pipeline all tests passed!")


def test_frs_visualizer():
    print("\n--- Testing FRS Visualizer Overlays & Badges ---")
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    tlwhs = [
        [100.0, 100.0, 150.0, 300.0],
        [350.0, 100.0, 150.0, 300.0],
        [600.0, 100.0, 150.0, 300.0],
        [850.0, 100.0, 150.0, 300.0],
    ]
    track_ids = [1, 2, 3, 4]

    mock_frs_results = {
        1: {
            "person_id": "SUSPECT_001",
            "name": "Johnathan Reynolds",
            "category": "WANTED",
            "confidence": 0.88,
            "is_flagged": True,
        },
        2: {
            "person_id": "VIP_OFFICIAL_1",
            "name": "Dr. Alok Sharma",
            "category": "VIP",
            "confidence": 0.92,
            "is_flagged": False,
        },
        3: {
            "person_id": "UNKNOWN",
            "name": "Unknown Person",
            "category": "UNKNOWN",
            "confidence": 0.28,
            "is_flagged": False,
        },
        # Track 4 has no cached result yet (Scanning state)
    }

    annotated = FRSVisualizer.draw_frs_overlay(
        image=frame,
        tlwhs=tlwhs,
        track_ids=track_ids,
        frs_results=mock_frs_results,
        frame_id=10,
        fps=29.97,
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    print("[OK] FRS Visualizer rendered multi-category HUD and badges successfully!")


if __name__ == "__main__":
    print("==========================================================")
    print("RUNNING FRS (FACIAL RECOGNITION SYSTEM) VERIFICATION SUITE")
    print("==========================================================")
    test_face_database()
    test_face_detector_and_embedder()
    test_frs_pipeline_caching()
    test_frs_visualizer()
    print("\n==========================================================")
    print("ALL FRS TESTS PASSED WITH 100% VERIFICATION INTEGRITY!")
    print("==========================================================")
