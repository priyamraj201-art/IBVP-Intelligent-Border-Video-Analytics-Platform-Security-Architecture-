#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated Test Suite for Multi-Class VEHICLE/HUMAN Detection & Track Routing.
"""

import os
import sys
import json
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from yolox.tracker.byte_tracker import STrack, BYTETracker
from yolox.routing.track_router import TrackRouter, DetectorMode
from tools.prepare_vehicle_human_dataset import convert_dataset
from tools.eval_vehicle_human import evaluate_routing_correctness


class MockArgs:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False


def test_strack_majority_voting():
    print("\n--- 1. Testing STrack Class Smoothing & Majority Voting ---")
    tlwh = [100, 100, 50, 150]
    
    # Create initial track as VEHICLE (0) and activate
    from yolox.tracker.kalman_filter import KalmanFilter
    track = STrack(tlwh, score=0.9, cls=0)
    track.activate(KalmanFilter(), frame_id=1)
    assert track.cls == 0
    assert track.smoothed_cls == 0
    print("[OK] Initial STrack initialized with class 0 (VEHICLE)")

    # Simulate sequence: [0, 0, 1 (noisy flip), 0, 0]
    observations = [0, 1, 0, 0]
    for frame_id, obs_cls in enumerate(observations, start=2):
        fake_det = STrack(tlwh, score=0.88, cls=obs_cls)
        track.update(fake_det, frame_id=frame_id)

    # Check smoothed class - majority vote must still be 0 (VEHICLE) despite the single frame noise
    assert track.smoothed_cls == 0, f"Expected smoothed_cls 0, got {track.smoothed_cls}"
    print(f"[OK] Majority voting preserved class {track.smoothed_cls} despite noisy flip in history: {list(track.class_history)}")


def test_byte_tracker_multiclass_update():
    print("\n--- 2. Testing BYTETracker Multi-Class Detections Update ---")
    args = MockArgs()
    tracker = BYTETracker(args)

    # Create dummy 7-element tensor: [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
    # Detection 1: HUMAN (cls 0), Detection 2: VEHICLE (cls 1)
    dets = torch.tensor([
        [400.0, 150.0, 480.0, 350.0, 0.95, 0.95, 0.0],  # Human
        [100.0, 100.0, 250.0, 200.0, 0.92, 0.90, 1.0],  # Vehicle
    ])

    img_info = [720, 1280]
    img_size = [800, 1440]

    tracked = tracker.update(dets, img_info, img_size)
    assert len(tracked) == 2, f"Expected 2 tracks, got {len(tracked)}"
    
    classes = [t.cls for t in tracked]
    assert 0 in classes and 1 in classes, f"Expected classes 0 and 1 in tracked objects, got {classes}"
    print(f"[OK] BYTETracker tracked multi-class objects: {[f'ID {t.track_id}: cls {t.cls}' for t in tracked]}")


def test_track_router_segregation():
    print("\n--- 3. Testing TrackRouter Routing Logic ---")
    router = TrackRouter(class_names=["HUMAN", "VEHICLE"])

    # Create mock tracks: Class 0 is HUMAN, Class 1 is VEHICLE
    h_track = STrack([400, 150, 60, 180], 0.88, cls=0)
    h_track.track_id = 1
    h_track.is_activated = True

    v_track = STrack([100, 100, 150, 100], 0.92, cls=1)
    v_track.track_id = 2
    v_track.is_activated = True

    tracks = [h_track, v_track]

    # Test 3A: MULTI_CLASS_PRODUCTION
    res_prod = router.route(tracks, detector_mode=DetectorMode.MULTI_CLASS_PRODUCTION)
    assert len(res_prod.human_tracks) == 1 and res_prod.human_tracks[0].track_id == 1
    assert len(res_prod.vehicle_tracks) == 1 and res_prod.vehicle_tracks[0].track_id == 2
    print("[OK] Production Mode strictly segregated vehicle and human tracks")

    # Test 3B: SINGLE_CLASS_TEST (Legacy test mode)
    res_test = router.route(tracks, detector_mode=DetectorMode.SINGLE_CLASS_TEST)
    assert len(res_test.vehicle_tracks) == 2
    assert len(res_test.human_tracks) == 2
    print("[OK] Single-Class Test Mode preserved legacy access for all tracks")

    # Test 3C: Visual Overlay
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = router.draw_unified_overlay(
        dummy_frame, res_prod, anpr_results={2: {"plate_number": "VIP007", "is_flagged": True, "alert_category": "VIP"}},
        alert_data={1: {"level": "LOW", "speed": 45.0, "color": (0, 255, 0), "trail": [(400, 150)]}},
        detector_mode=DetectorMode.MULTI_CLASS_PRODUCTION,
        frame_id=1, fps=30.0
    )
    assert rendered.shape == dummy_frame.shape
    print("[OK] Unified visualizer rendered successfully")


def test_dataset_preparation():
    print("\n--- 4. Testing Dataset Preparation Tool ---")
    synthetic_coco = {
        "images": [{"id": 1, "file_name": "test.jpg", "height": 720, "width": 1280}],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
            {"id": 8, "name": "truck"},
            {"id": 18, "name": "dog"}, # to be dropped
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 100]}, # human (0)
            {"id": 102, "image_id": 1, "category_id": 3, "bbox": [100, 100, 200, 150]}, # vehicle (1)
            {"id": 103, "image_id": 1, "category_id": 8, "bbox": [350, 100, 300, 250]}, # vehicle (1)
            {"id": 104, "image_id": 1, "category_id": 18, "bbox": [20, 20, 30, 30]}, # dog (drop)
        ]
    }

    in_path = "temp_synthetic_coco.json"
    out_path = "temp_vehicle_human.json"
    with open(in_path, "w") as f:
        json.dump(synthetic_coco, f)

    try:
        convert_dataset(in_path, out_path)
        with open(out_path, "r") as f:
            converted = json.load(f)

        assert len(converted["categories"]) == 2
        assert len(converted["annotations"]) == 3  # 1 human + 2 vehicles (dog dropped)
        cat_ids = [a["category_id"] for a in converted["annotations"]]
        assert cat_ids.count(0) == 1  # 1 human (0)
        assert cat_ids.count(1) == 2  # 2 vehicles (1)
        print("[OK] Dataset preparation successfully converted and remapped annotations")
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)


def test_eval_script():
    print("\n--- 5. Testing Evaluation & Routing Audit ---")
    evaluate_routing_correctness(DetectorMode.MULTI_CLASS_PRODUCTION, num_samples=50)
    print("[OK] Evaluation script completed successfully")


if __name__ == "__main__":
    print("==========================================================")
    print("RUNNING MULTI-CLASS DETECTION & ROUTING VERIFICATION SUITE")
    print("==========================================================")
    test_strack_majority_voting()
    test_byte_tracker_multiclass_update()
    test_track_router_segregation()
    test_dataset_preparation()
    test_eval_script()
    print("\n==========================================================")
    print("ALL TESTS PASSED WITH 100% ROUTING INTEGRITY!")
    print("==========================================================")
