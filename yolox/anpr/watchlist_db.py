import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple
from loguru import logger


class WatchlistDB:
    """
    Thread-safe SQLite database manager for ANPR Watchlist and Detection Logs.
    Supports vehicle hotlist lookups (e.g. STOLEN, WANTED, VIP) and forensic event logging.
    """

    def __init__(self, db_path: str = "anpr_watchlist.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize SQLite schema if tables do not exist."""
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # 1. Watchlist Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS watchlist (
                        plate_number TEXT PRIMARY KEY,
                        owner_name TEXT,
                        alert_category TEXT NOT NULL,  -- e.g. STOLEN, WANTED, VIP, SUSPICIOUS, UNREGISTERED
                        notes TEXT,
                        created_at REAL DEFAULT (strftime('%s', 'now'))
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_category ON watchlist(alert_category)")

                # 2. Detection Audit Logs Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS anpr_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        datetime_str TEXT,
                        track_id INTEGER,
                        plate_number TEXT,
                        confidence REAL,
                        is_flagged INTEGER DEFAULT 0,
                        alert_category TEXT,
                        bbox_area REAL,
                        crop_path TEXT
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_plate ON anpr_logs(plate_number)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON anpr_logs(timestamp)")
                conn.commit()
            finally:
                conn.close()
        logger.info(f"ANPR Watchlist Database initialized at: {self.db_path}")

    @staticmethod
    def _canonical(text: str) -> str:
        return text.strip().upper().replace(" ", "").replace("-", "").replace("O", "0").replace("I", "1").replace("Z", "2")

    def lookup_plate(self, plate_number: str) -> Optional[Dict]:
        """
        Query if a normalized license plate is on the watchlist, with OCR character variance matching (O/0, I/1).
        """
        if not plate_number:
            return None
        cleaned = plate_number.strip().upper().replace(" ", "").replace("-", "")
        canon = self._canonical(cleaned)

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # 1. Exact lookup
                cursor.execute(
                    "SELECT plate_number, owner_name, alert_category, notes, created_at FROM watchlist WHERE plate_number = ?",
                    (cleaned,),
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)

                # 2. Canonical variant match (e.g. VIP007 <-> VIPOO7)
                cursor.execute("SELECT plate_number, owner_name, alert_category, notes, created_at FROM watchlist")
                all_rows = cursor.fetchall()
                for r in all_rows:
                    if self._canonical(r["plate_number"]) == canon:
                        return dict(r)

                return None
            finally:
                conn.close()

    def add_watchlist_entry(
        self,
        plate_number: str,
        alert_category: str = "WANTED",
        owner_name: str = "",
        notes: str = ""
    ) -> bool:
        """Add or update a license plate in the watchlist."""
        cleaned = plate_number.strip().upper().replace(" ", "").replace("-", "")
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO watchlist (plate_number, owner_name, alert_category, notes, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(plate_number) DO UPDATE SET
                        alert_category=excluded.alert_category,
                        owner_name=excluded.owner_name,
                        notes=excluded.notes
                """, (cleaned, owner_name, alert_category.upper(), notes, time.time()))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to add watchlist entry: {e}")
                return False
            finally:
                conn.close()

    def remove_watchlist_entry(self, plate_number: str) -> bool:
        """Remove a plate from the watchlist."""
        cleaned = plate_number.strip().upper().replace(" ", "").replace("-", "")
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM watchlist WHERE plate_number = ?", (cleaned,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def get_all_watchlist(self) -> List[Dict]:
        """Fetch all entries currently in the watchlist."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT plate_number, owner_name, alert_category, notes, created_at FROM watchlist ORDER BY created_at DESC")
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def log_detection(
        self,
        track_id: int,
        plate_number: str,
        confidence: float,
        is_flagged: bool,
        alert_category: Optional[str] = None,
        bbox_area: float = 0.0,
        crop_path: Optional[str] = None,
    ) -> int:
        """Log an ANPR detection event into the audit table."""
        cleaned = plate_number.strip().upper().replace(" ", "").replace("-", "")
        curr_time = time.time()
        datetime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(curr_time))

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO anpr_logs (
                        timestamp, datetime_str, track_id, plate_number, confidence, is_flagged, alert_category, bbox_area, crop_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    curr_time,
                    datetime_str,
                    track_id,
                    cleaned,
                    float(confidence),
                    1 if is_flagged else 0,
                    alert_category,
                    float(bbox_area),
                    crop_path
                ))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                logger.error(f"Failed to log ANPR detection: {e}")
                return -1
            finally:
                conn.close()

    def get_recent_logs(self, limit: int = 50, flagged_only: bool = False) -> List[Dict]:
        """Fetch recent detection records."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM anpr_logs"
                if flagged_only:
                    query += " WHERE is_flagged = 1"
                query += " ORDER BY id DESC LIMIT ?"
                cursor.execute(query, (limit,))
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def seed_sample_watchlist(self):
        """Seed realistic sample watchlist data for testing and demonstrations."""
        sample_entries = [
            ("MH12AB1234", "STOLEN", "John Doe", "Red sedan reported stolen in Pune"),
            ("KA01MJ5678", "WANTED", "David Miller", "Active warrant for robbery"),
            ("DL3CA9999", "VIP", "Executive Council", "Authorized VIP vehicle with priority access"),
            ("MH02CZ4321", "SUSPICIOUS", "Unknown", "Involved in toll evasion incidents"),
            ("CA78XYZ9", "STOLEN", "Sarah Connor", "Blue SUV reported stolen"),
            ("ABC1234", "WANTED", "James Bond", "Vehicle under surveillance"),
            ("VIP007", "VIP", "Directorate", "Special security clearance")
        ]
        for plate, cat, owner, notes in sample_entries:
            self.add_watchlist_entry(plate, alert_category=cat, owner_name=owner, notes=notes)
        logger.info("Sample watchlist seeded with test plates (STOLEN, WANTED, VIP).")
