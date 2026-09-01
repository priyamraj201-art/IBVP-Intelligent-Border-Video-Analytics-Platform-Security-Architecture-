import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import sqlite3
import threading
import base64
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaceDatabase:
    """
    Thread-safe SQLite database for enrolled facial identities and forensic audit logs.
    Features:
    - AES-256 encrypted biometric embedding storage (Fernet / SHA256 key derivation)
    - Incremental multi-observation embedding averaging for progressive identity refinement
    - High-speed FAISS vector index search with NumPy vector batch fallback
    """

    def __init__(self, db_path: str = "frs_faces.db", embedding_dim: int = 512):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._lock = threading.Lock()

        # Encryption Key Setup
        self._cipher = None
        self._init_encryption()

        # Initialize SQLite Schema
        self._init_db()

    def _init_encryption(self):
        """Derive 32-byte Fernet key from environment master key."""
        if HAS_CRYPTOGRAPHY:
            master_key_str = os.environ.get("IBVAP_MASTER_KEY", "ibvap_secure_border_master_key_2026")
            derived_key = base64.urlsafe_b64encode(hashlib.sha256(master_key_str.encode()).digest())
            self._cipher = Fernet(derived_key)
        else:
            logger.warning("cryptography package not installed. Facial embeddings will be stored as raw bytes.")

    def _encrypt_embedding(self, emb: np.ndarray) -> bytes:
        raw_bytes = emb.astype(np.float32).tobytes()
        if self._cipher is not None:
            return self._cipher.encrypt(raw_bytes)
        return raw_bytes

    def _decrypt_embedding(self, blob: bytes) -> np.ndarray:
        if self._cipher is not None:
            try:
                decrypted = self._cipher.decrypt(blob)
                return np.frombuffer(decrypted, dtype=np.float32)
            except Exception:
                # If decrypt fails, try reading as raw bytes
                return np.frombuffer(blob, dtype=np.float32)
        return np.frombuffer(blob, dtype=np.float32)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize SQLite tables and indexes."""
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # 1. Enrolled Face Identities Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_identities (
                        person_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        category TEXT NOT NULL,  -- WANTED / STAFF / VIP / SUSPECT / UNKNOWN_REPEAT
                        notes TEXT,
                        embedding_blob BLOB NOT NULL,
                        enrolled_at REAL,
                        enrolled_by TEXT DEFAULT 'admin',
                        face_count INTEGER DEFAULT 1
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_category ON face_identities(category)")

                # 2. Recognition Audit Logs Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS frs_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        datetime_str TEXT,
                        track_id INTEGER,
                        person_id TEXT,
                        name TEXT,
                        confidence REAL,
                        category TEXT,
                        is_flagged INTEGER DEFAULT 0,
                        camera_id INTEGER DEFAULT 0,
                        bbox_area REAL DEFAULT 0.0
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_frs_logs_person ON frs_logs(person_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_frs_logs_timestamp ON frs_logs(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_frs_logs_flagged ON frs_logs(is_flagged)")
                conn.commit()
            finally:
                conn.close()
        logger.info(f"FRS Database initialized at: {self.db_path}")

    def enroll_face(
        self,
        person_id: str,
        name: str = "",
        embedding: np.ndarray = None,
        category: str = "UNKNOWN",
        notes: str = "",
        enrolled_by: str = "admin",
        full_name: str = None,
    ) -> bool:
        """
        Enroll a new face or incrementally refine an existing identity with a new observation.

        :param person_id: Unique identifier (e.g. 'SUSPECT_007').
        :param name: Full name of person.
        :param embedding: 512-dimensional float32 vector.
        :param category: Identity category (WANTED / STAFF / VIP / SUSPECT / UNKNOWN_REPEAT).
        :param notes: Remarks or case notes.
        :param enrolled_by: Operator username.
        :return: True on success.
        """
        if full_name is not None and not name:
            name = full_name
        elif not name:
            name = person_id
        if embedding is None:
            return False

        emb = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        else:
            return False

        person_id = person_id.strip().upper()
        category = category.strip().upper()
        now = time.time()

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT embedding_blob, face_count FROM face_identities WHERE person_id = ?", (person_id,))
                row = cursor.fetchone()

                if row is not None:
                    # Incremental update: Average new embedding with stored observation history
                    old_blob = row["embedding_blob"]
                    old_count = row["face_count"]
                    old_emb = self._decrypt_embedding(old_blob)

                    if len(old_emb) == len(emb):
                        # Weighted cumulative average
                        combined_emb = (old_emb * old_count + emb) / float(old_count + 1)
                        c_norm = np.linalg.norm(combined_emb)
                        if c_norm > 1e-6:
                            combined_emb = combined_emb / c_norm
                        emb = combined_emb
                    new_count = old_count + 1

                    enc_blob = self._encrypt_embedding(emb)
                    cursor.execute("""
                        UPDATE face_identities SET
                            name = ?,
                            category = ?,
                            notes = ?,
                            embedding_blob = ?,
                            enrolled_at = ?,
                            enrolled_by = ?,
                            face_count = ?
                        WHERE person_id = ?
                    """, (name, category, notes, enc_blob, now, enrolled_by, new_count, person_id))
                    logger.info(f"Refined face identity '{person_id}' ({name}) - Total observations: {new_count}")
                else:
                    # New enrollment
                    enc_blob = self._encrypt_embedding(emb)
                    cursor.execute("""
                        INSERT INTO face_identities (
                            person_id, name, category, notes, embedding_blob, enrolled_at, enrolled_by, face_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """, (person_id, name, category, notes, enc_blob, now, enrolled_by))
                    logger.info(f"Enrolled new face identity '{person_id}' ({name}) [{category}]")

                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to enroll face identity {person_id}: {e}")
                return False
            finally:
                conn.close()

    def search_face(self, embedding: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Search watchlist for matching facial identities using cosine similarity.

        :param embedding: 512-dimensional query vector.
        :param top_k: Number of highest-confidence matches to return.
        :return: List of dicts with {'person_id', 'name', 'category', 'confidence', 'notes'}.
        """
        if embedding is None:
            return []

        q_emb = np.asarray(embedding, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 1e-6:
            q_emb = q_emb / q_norm
        else:
            return []

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT person_id, name, category, notes, embedding_blob FROM face_identities")
                rows = cursor.fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        # Decrypt stored embeddings into matrix
        identities = []
        stored_vectors = []
        for r in rows:
            vec = self._decrypt_embedding(r["embedding_blob"])
            if len(vec) == self.embedding_dim:
                v_norm = np.linalg.norm(vec)
                if v_norm > 1e-6:
                    vec = vec / v_norm
                stored_vectors.append(vec)
                identities.append(r)

        if not stored_vectors:
            return []

        matrix = np.vstack(stored_vectors).astype(np.float32)

        # 1. FAISS Search (if available)
        if HAS_FAISS:
            try:
                index = faiss.IndexFlatIP(self.embedding_dim)
                index.add(matrix)
                k_search = min(top_k, len(identities))
                distances, indices = index.search(q_emb.reshape(1, -1), k_search)

                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < len(identities):
                        match_row = identities[idx]
                        results.append({
                            "person_id": match_row["person_id"],
                            "name": match_row["name"],
                            "category": match_row["category"],
                            "confidence": float(np.clip(dist, -1.0, 1.0)),
                            "notes": match_row["notes"],
                        })
                return results
            except Exception as e:
                logger.error(f"FAISS search failed ({e}), falling back to NumPy.")

        # 2. NumPy Batch Cosine Similarity (Dot product on normalized vectors)
        similarities = np.dot(matrix, q_emb)
        ranked_indices = np.argsort(-similarities)[:top_k]

        results = []
        for idx in ranked_indices:
            match_row = identities[idx]
            conf = float(np.clip(similarities[idx], -1.0, 1.0))
            results.append({
                "person_id": match_row["person_id"],
                "name": match_row["name"],
                "category": match_row["category"],
                "confidence": conf,
                "notes": match_row["notes"],
            })
        return results

    def get_all_identities(self) -> List[Dict[str, Any]]:
        """Fetch all enrolled face records without embedding blobs."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT person_id, name, category, notes, enrolled_at, enrolled_by, face_count
                    FROM face_identities
                    ORDER BY enrolled_at DESC
                """)
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def remove_identity(self, person_id: str) -> bool:
        """Remove a face identity from the watchlist."""
        cleaned = person_id.strip().upper()
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM face_identities WHERE person_id = ?", (cleaned,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def log_recognition(
        self,
        track_id: int,
        person_id: str,
        name: str,
        confidence: float,
        is_flagged: bool,
        category: str,
        camera_id: int = 0,
        bbox_area: float = 0.0,
    ) -> int:
        """Log a face recognition event into SQLite audit log."""
        curr_time = time.time()
        datetime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(curr_time))

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO frs_logs (
                        timestamp, datetime_str, track_id, person_id, name, confidence, category, is_flagged, camera_id, bbox_area
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    curr_time,
                    datetime_str,
                    int(track_id),
                    person_id,
                    name,
                    float(confidence),
                    category,
                    1 if is_flagged else 0,
                    int(camera_id),
                    float(bbox_area)
                ))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                logger.error(f"Failed to log face recognition event: {e}")
                return -1
            finally:
                conn.close()

    def get_recent_logs(self, limit: int = 50, flagged_only: bool = False) -> List[Dict[str, Any]]:
        """Fetch recent forensic recognition audit logs."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM frs_logs"
                if flagged_only:
                    query += " WHERE is_flagged = 1"
                query += " ORDER BY id DESC LIMIT ?"
                cursor.execute(query, (limit,))
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def seed_sample_identities(self):
        """Seed realistic sample watchlist identities for demonstration."""
        sample_entries = [
            ("SUSPECT_001", "Johnathan Reynolds", "WANTED", "Active warrant for cross-border contraband smuggling"),
            ("SUSPECT_002", "Vikramaditya Singh", "WANTED", "High-priority INTERPOL Red Notice notice #IN-2025-99"),
            ("VIP_OFFICIAL_1", "Dr. Alok Sharma", "VIP", "Directorate General - Special Border Inspection Team"),
            ("STAFF_GUARD_1", "Commander Rajesh Patil", "STAFF", "Sector 4 Border Security Outpost Commander"),
            ("UNKNOWN_REPEAT_1", "Subject 094", "SUSPECT", "Repeated unauthorized loitering detected near border fence"),
        ]

        np.random.seed(42)
        for pid, name, cat, notes in sample_entries:
            # Generate deterministic unit embedding vector for demonstration
            raw_vec = np.random.randn(self.embedding_dim).astype(np.float32)
            unit_vec = raw_vec / np.linalg.norm(raw_vec)
            self.enroll_face(
                person_id=pid,
                name=name,
                embedding=unit_vec,
                category=cat,
                notes=notes,
                enrolled_by="system_seeder",
            )
        logger.info(f"Sample FRS watchlist seeded with {len(sample_entries)} identities (WANTED, VIP, STAFF, SUSPECT).")


# Convenience alias matching naming specifications
FRSDatabase = FaceDatabase
