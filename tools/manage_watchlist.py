#!/usr/bin/env python
import argparse
import os
import sys
from tabulate import tabulate

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from yolox.anpr.watchlist_db import WatchlistDB


def make_parser():
    parser = argparse.ArgumentParser(description="ANPR Watchlist and Audit Log Manager")
    parser.add_argument("--db", default="anpr_watchlist.db", help="Path to SQLite watchlist database")

    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # List watchlist
    subparsers.add_parser("list", help="List all plates in the watchlist")

    # Seed sample plates
    subparsers.add_parser("seed", help="Seed database with sample test plates (STOLEN, WANTED, VIP)")

    # Add plate
    add_parser = subparsers.add_parser("add", help="Add or update a plate in the watchlist")
    add_parser.add_argument("--plate", required=True, type=str, help="License plate number (e.g. MH12AB1234)")
    add_parser.add_argument("--category", default="WANTED", choices=["STOLEN", "WANTED", "VIP", "SUSPICIOUS", "UNREGISTERED"], help="Alert category")
    add_parser.add_argument("--owner", default="", type=str, help="Owner name or agency")
    add_parser.add_argument("--notes", default="", type=str, help="Incident notes / remarks")

    # Remove plate
    del_parser = subparsers.add_parser("remove", help="Remove a plate from the watchlist")
    del_parser.add_argument("--plate", required=True, type=str, help="License plate number to remove")

    # Lookup plate
    lookup_parser = subparsers.add_parser("lookup", help="Query a specific plate")
    lookup_parser.add_argument("--plate", required=True, type=str, help="License plate number to check")

    # View detection logs
    logs_parser = subparsers.add_parser("logs", help="View recent ANPR detection audit logs")
    logs_parser.add_argument("--limit", type=int, default=25, help="Number of records to show")
    logs_parser.add_argument("--flagged", action="store_true", help="Filter only watchlist flagged hits")

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db = WatchlistDB(db_path=args.db)

    if args.command == "list":
        rows = db.get_all_watchlist()
        if not rows:
            print(f"No records found in watchlist ({args.db}).")
            return
        table = [[r["plate_number"], r["alert_category"], r["owner_name"], r["notes"]] for r in rows]
        print("\n=== ANPR SECURITY WATCHLIST ===")
        print(tabulate(table, headers=["Plate Number", "Category", "Owner / Org", "Notes"], tablefmt="grid"))

    elif args.command == "seed":
        db.seed_sample_watchlist()
        print(f"Sample watchlist successfully populated in: {args.db}")

    elif args.command == "add":
        success = db.add_watchlist_entry(
            plate_number=args.plate,
            alert_category=args.category,
            owner_name=args.owner,
            notes=args.notes
        )
        if success:
            print(f"Successfully added/updated plate '{args.plate}' as [{args.category}].")
        else:
            print(f"Failed to add plate '{args.plate}'.")

    elif args.command == "remove":
        removed = db.remove_watchlist_entry(args.plate)
        if removed:
            print(f"Plate '{args.plate}' removed from watchlist.")
        else:
            print(f"Plate '{args.plate}' not found in watchlist.")

    elif args.command == "lookup":
        result = db.lookup_plate(args.plate)
        if result:
            print(f"\n[ALERT] Plate '{args.plate}' is in Watchlist!")
            print(f"Category: {result['alert_category']}")
            print(f"Owner:    {result['owner_name']}")
            print(f"Notes:    {result['notes']}")
        else:
            print(f"Plate '{args.plate}' is NOT in the watchlist (Status: CLEAN).")

    elif args.command == "logs":
        logs = db.get_recent_logs(limit=args.limit, flagged_only=args.flagged)
        if not logs:
            print("No detection logs found.")
            return
        table = [
            [
                l["id"],
                l["datetime_str"],
                l["track_id"],
                l["plate_number"],
                f"{l['confidence']:.2f}",
                "YES (ALERT)" if l["is_flagged"] else "NO",
                l["alert_category"] or "NORMAL",
                int(l["bbox_area"])
            ]
            for l in logs
        ]
        print(f"\n=== RECENT ANPR DETECTION AUDIT LOGS (Limit: {args.limit}) ===")
        print(tabulate(table, headers=["ID", "Timestamp", "Track ID", "Plate", "Conf", "Flagged", "Category", "BBox Area"], tablefmt="grid"))


if __name__ == "__main__":
    main()
