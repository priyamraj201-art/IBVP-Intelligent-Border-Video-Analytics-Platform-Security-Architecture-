#!/usr/bin/env python
import argparse
import os
import sys

# Prevent OpenMP runtime collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from tabulate import tabulate

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from yolox.frs.face_database import FaceDatabase
from yolox.frs.face_detector import FaceDetector
from yolox.frs.face_embedder import FaceEmbedder


def make_parser():
    parser = argparse.ArgumentParser(description="FRS Facial Identity Database and Audit Log Manager")
    parser.add_argument("--db", default="frs_faces.db", help="Path to SQLite FRS database")

    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # List watchlist
    subparsers.add_parser("list", help="List all enrolled facial identities")

    # Seed sample identities
    subparsers.add_parser("seed", help="Seed database with sample test identities (WANTED, VIP, STAFF, SUSPECT)")

    # Enroll face
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new face identity from an image or vector")
    enroll_parser.add_argument("--person-id", required=True, type=str, help="Unique identifier (e.g. SUSPECT_007)")
    enroll_parser.add_argument("--name", required=True, type=str, help="Full name of person")
    enroll_parser.add_argument(
        "--category",
        default="WANTED",
        choices=["WANTED", "STAFF", "VIP", "SUSPECT", "UNKNOWN_REPEAT"],
        help="Security alert category",
    )
    enroll_parser.add_argument("--notes", default="", type=str, help="Incident notes / remarks")
    enroll_parser.add_argument("--image", type=str, default=None, help="Path to face portrait image file")
    enroll_parser.add_argument("--operator", default="admin", type=str, help="Operator username")

    # Remove identity
    del_parser = subparsers.add_parser("remove", help="Remove an identity from the watchlist")
    del_parser.add_argument("--person-id", required=True, type=str, help="Person ID to remove")

    # Lookup identity
    lookup_parser = subparsers.add_parser("lookup", help="Query a specific person ID or match an image")
    lookup_parser.add_argument("--person-id", type=str, default=None, help="Person ID to look up")
    lookup_parser.add_argument("--image", type=str, default=None, help="Image to search against database")

    # View detection logs
    logs_parser = subparsers.add_parser("logs", help="View recent FRS recognition audit logs")
    logs_parser.add_argument("--limit", type=int, default=25, help="Number of records to show")
    logs_parser.add_argument("--flagged", action="store_true", help="Filter only watchlist flagged hits")

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db = FaceDatabase(db_path=args.db)

    if args.command == "list":
        rows = db.get_all_identities()
        if not rows:
            print(f"No enrolled identities found in database ({args.db}).")
            return
        table = [
            [
                r["person_id"],
                r["name"],
                r["category"],
                r["face_count"],
                r["enrolled_by"],
                r["notes"],
            ]
            for r in rows
        ]
        print(f"\n=== ENROLLED FACIAL IDENTITIES ({args.db}) ===")
        print(
            tabulate(
                table,
                headers=["Person ID", "Full Name", "Category", "Observations", "Enrolled By", "Notes"],
                tablefmt="grid",
            )
        )

    elif args.command == "seed":
        db.seed_sample_identities()
        print(f"Sample FRS identities successfully seeded in: {args.db}")

    elif args.command == "enroll":
        if args.image and os.path.exists(args.image):
            img = cv2.imread(args.image)
            if img is None:
                print(f"Error: Could not read image at '{args.image}'.")
                return
            detector = FaceDetector()
            embedder = FaceEmbedder()
            face_info = detector.get_best_face(img)
            if face_info is None:
                print(f"Error: No valid face detected in '{args.image}'.")
                return
            face_crop, det_conf = face_info
            embedding, quality = embedder.get_embedding(face_crop)
            print(f"Detected face (det conf: {det_conf:.2f}, quality: {quality:.2f})")
        else:
            # Generate random unit vector for testing if no image provided
            np.random.seed(abs(hash(args.person_id)) % (2**31))
            raw_vec = np.random.randn(512).astype(np.float32)
            embedding = raw_vec / np.linalg.norm(raw_vec)
            print(f"Generated synthetic 512-d unit embedding for '{args.person_id}'.")

        success = db.enroll_face(
            person_id=args.person_id,
            name=args.name,
            embedding=embedding,
            category=args.category,
            notes=args.notes,
            enrolled_by=args.operator,
        )
        if success:
            print(f"Successfully enrolled identity '{args.person_id}' ({args.name}) as [{args.category}].")
        else:
            print(f"Failed to enroll identity '{args.person_id}'.")

    elif args.command == "remove":
        removed = db.remove_identity(args.person_id)
        if removed:
            print(f"Identity '{args.person_id}' removed from FRS database.")
        else:
            print(f"Identity '{args.person_id}' not found in database.")

    elif args.command == "lookup":
        if args.image and os.path.exists(args.image):
            img = cv2.imread(args.image)
            detector = FaceDetector()
            embedder = FaceEmbedder()
            face_info = detector.get_best_face(img)
            if face_info is None:
                print(f"No face detected in '{args.image}'.")
                return
            face_crop, det_conf = face_info
            embedding, quality = embedder.get_embedding(face_crop)
            matches = db.search_face(embedding, top_k=3)
            if matches:
                print(f"\n=== Top Face Matches for '{args.image}' ===")
                table = [
                    [m["person_id"], m["name"], m["category"], f"{m['confidence']:.2f}", m["notes"]]
                    for m in matches
                ]
                print(tabulate(table, headers=["Person ID", "Name", "Category", "Cosine Conf", "Notes"], tablefmt="grid"))
            else:
                print("No matching identities found in database.")
        elif args.person_id:
            rows = db.get_all_identities()
            matched = [r for r in rows if r["person_id"] == args.person_id.strip().upper()]
            if matched:
                r = matched[0]
                print(f"\n=== Identity Details for '{args.person_id}' ===")
                print(f"Person ID:    {r['person_id']}")
                print(f"Name:         {r['name']}")
                print(f"Category:     {r['category']}")
                print(f"Observations: {r['face_count']}")
                print(f"Enrolled By:  {r['enrolled_by']}")
                print(f"Notes:        {r['notes']}")
            else:
                print(f"Identity '{args.person_id}' not found in database.")
        else:
            print("Please specify --person-id or --image for lookup.")

    elif args.command == "logs":
        logs = db.get_recent_logs(limit=args.limit, flagged_only=args.flagged)
        if not logs:
            print("No recognition audit logs found.")
            return
        table = [
            [
                l["id"],
                l["datetime_str"],
                l["track_id"],
                l["person_id"],
                l["name"],
                f"{l['confidence']:.2f}",
                "YES (ALERT)" if l["is_flagged"] else "NO",
                l["category"] or "UNKNOWN",
                int(l["bbox_area"]),
            ]
            for l in logs
        ]
        print(f"\n=== RECENT FRS FORENSIC AUDIT LOGS (Limit: {args.limit}) ===")
        print(
            tabulate(
                table,
                headers=["ID", "Timestamp", "Track ID", "Person ID", "Name", "Conf", "Flagged", "Category", "BBox Area"],
                tablefmt="grid",
            )
        )


if __name__ == "__main__":
    main()
