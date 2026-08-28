#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Preparation Tool for 2-Class YOLOX (VEHICLE vs HUMAN).

This tool filters and merges standard COCO-format datasets into a 2-class schema:
  - Class 0: VEHICLE (car, truck, bus, motorcycle, bicycle)
  - Class 1: HUMAN (person)
All other 74 COCO categories are dropped.

=============================================================================
DEPLOYMENT WORKFLOW & DOMAIN FINE-TUNING NOTES (CVAT / Label Studio):
=============================================================================
1. Starting Point:
   COCO-pretrained models provide strong general feature representations, but
   CCTV, pole-mounted, and border cameras feature steep perspective angles,
   different aspect ratios, and varying night/IR lighting.

2. Annotation Guidelines for Real Camera Data:
   - Tool: Use CVAT (Computer Vision Annotation Tool) or Label Studio.
   - Bounding Box Scheme:
       - 'vehicle': Any motorized or pedal vehicle (car, van, truck, bus, bike).
         Bounding box must encompass the entire chassis (including tow mirrors
         and roof cargo).
       - 'human': Any pedestrian, cyclist mounted on a bike (annotate both the
         human and the vehicle), security guard, or operator.
   - Export: Export annotations from CVAT/Label Studio in 'COCO 1.0' format.

3. Fine-tuning Command:
   python tools/train.py -f exps/example/mot/yolox_vehicle_human.py \\
       -d 1 -b 16 --fp16 -c pretrained/yolox_x.pth
=============================================================================
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from tabulate import tabulate


# Standard 80-Class COCO Category IDs to Target 2-Class Mapping
COCO_MAPPING = {
    1: {"target_id": 0, "name": "HUMAN", "orig": "person"},
    2: {"target_id": 1, "name": "VEHICLE", "orig": "bicycle"},
    3: {"target_id": 1, "name": "VEHICLE", "orig": "car"},
    4: {"target_id": 1, "name": "VEHICLE", "orig": "motorcycle"},
    6: {"target_id": 1, "name": "VEHICLE", "orig": "bus"},
    8: {"target_id": 1, "name": "VEHICLE", "orig": "truck"},
}

TARGET_CATEGORIES = [
    {"id": 0, "name": "human", "supercategory": "human"},
    {"id": 1, "name": "vehicle", "supercategory": "vehicle"},
]


def make_parser():
    parser = argparse.ArgumentParser("COCO to 2-Class Vehicle/Human Converter")
    parser.add_argument(
        "--input_json",
        required=True,
        type=str,
        help="Path to input COCO annotation JSON (e.g. instances_train2017.json)",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        type=str,
        help="Path to output 2-class annotation JSON (e.g. train_vehicle_human.json)",
    )
    parser.add_argument(
        "--keep_empty_images",
        action="store_true",
        default=False,
        help="Whether to retain images with zero vehicle or human annotations as background samples",
    )
    return parser


def convert_dataset(input_json: str, output_json: str, keep_empty: bool = False):
    print(f"\n[INFO] Loading input annotations from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Check category schema
    cat_id_to_name = {c["id"]: c["name"].lower() for c in coco_data.get("categories", [])}

    # Map by name if category IDs differ from standard COCO
    name_to_target = {
        "person": 0,
        "pedestrian": 0,
        "human": 0,
        "bicycle": 1,
        "car": 1,
        "motorcycle": 1,
        "bus": 1,
        "truck": 1,
        "vehicle": 1,
        "van": 1,
    }

    filtered_annotations = []
    annotated_image_ids = set()
    category_counts = defaultdict(int)
    subcat_counts = defaultdict(int)

    for ann in coco_data.get("annotations", []):
        cat_id = ann.get("category_id")
        orig_name = cat_id_to_name.get(cat_id, "")
        
        target_id = None
        if cat_id in COCO_MAPPING:
            target_id = COCO_MAPPING[cat_id]["target_id"]
            subcat = COCO_MAPPING[cat_id]["orig"]
        elif orig_name in name_to_target:
            target_id = name_to_target[orig_name]
            subcat = orig_name
        else:
            continue  # Drop irrelevant category

        new_ann = dict(ann)
        new_ann["category_id"] = target_id
        filtered_annotations.append(new_ann)
        annotated_image_ids.add(ann["image_id"])

        cat_label = "HUMAN" if target_id == 1 else "VEHICLE"
        category_counts[cat_label] += 1
        subcat_counts[subcat] += 1

    # Filter images
    if keep_empty:
        filtered_images = coco_data.get("images", [])
    else:
        filtered_images = [img for img in coco_data.get("images", []) if img["id"] in annotated_image_ids]

    output_data = {
        "info": coco_data.get("info", {"description": "Vehicle & Human 2-Class Dataset"}),
        "licenses": coco_data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": TARGET_CATEGORIES,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"[OK] Successfully converted and saved to: {output_json}\n")

    # Display Summary Table
    summary_table = [
        ["Total Input Images", len(coco_data.get("images", []))],
        ["Retained Images", len(filtered_images)],
        ["Total Converted Annotations", len(filtered_annotations)],
        ["- Class 0: VEHICLE Instances", category_counts["VEHICLE"]],
        ["- Class 1: HUMAN Instances", category_counts["HUMAN"]],
    ]
    print(tabulate(summary_table, headers=["Metric", "Count"], tablefmt="grid"))

    print("\n--- Subcategory Breakdown ---")
    subcat_table = [[k.capitalize(), v] for k, v in sorted(subcat_counts.items(), key=lambda x: -x[1])]
    print(tabulate(subcat_table, headers=["Original Class", "Extracted Instances"], tablefmt="grid"))


def main():
    parser = make_parser()
    args = parser.parse_args()
    convert_dataset(args.input_json, args.output_json, args.keep_empty_images)


if __name__ == "__main__":
    main()
