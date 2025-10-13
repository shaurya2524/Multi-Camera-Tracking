#!/usr/bin/env python3
"""
Database Module for Multi-Camera Tracking System

This module provides database functionality for storing and retrieving
tracking data, including detections, tracks, and system statistics.

Author: Multi-Camera Tracking System
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DetectionRecord:
    """Database record for a detection"""
    id: Optional[int] = None
    session_id: str = ""
    frame_number: int = 0
    camera_id: int = 0
    local_id: str = ""
    global_id: Optional[int] = None
    bbox_x: int = 0
    bbox_y: int = 0
    bbox_width: int = 0
    bbox_height: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    confidence: float = 0.0
    class_id: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TrackRecord:
    """Database record for a track"""
    id: Optional[int] = None
    session_id: str = ""
    global_id: int = 0
    creation_frame: int = 0
    last_seen_frame: int = 0
    total_detections: int = 0
    active_cameras: str = ""  # JSON string of camera IDs
    avg_confidence: float = 0.0
    track_length: int = 0
    status: str = "active"  # active, lost, completed
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class SessionRecord:
    """Database record for a tracking session"""
    id: Optional[int] = None
    session_id: str = ""
    video1_path: str = ""
    video2_path: str = ""
    total_frames: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    max_global_id: int = 0
    config: str = ""  # JSON string of configuration
    started_at: datetime = None
    ended_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now()


class TrackingDatabase:
    """
    SQLite database manager for tracking system
    
    Provides functionality to store and retrieve tracking data including
    detections, tracks, and session information.
    """
    
    def __init__(self, db_path: str = "tracking_data.db"):
        """
        Initialize database connection
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        video1_path TEXT NOT NULL,
                        video2_path TEXT NOT NULL,
                        total_frames INTEGER DEFAULT 0,
                        total_detections INTEGER DEFAULT 0,
                        total_tracks INTEGER DEFAULT 0,
                        max_global_id INTEGER DEFAULT 0,
                        config TEXT,
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ended_at TIMESTAMP
                    )
                """)
                
                # Create detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        frame_number INTEGER NOT NULL,
                        camera_id INTEGER NOT NULL,
                        local_id TEXT NOT NULL,
                        global_id INTEGER,
                        bbox_x INTEGER NOT NULL,
                        bbox_y INTEGER NOT NULL,
                        bbox_width INTEGER NOT NULL,
                        bbox_height INTEGER NOT NULL,
                        center_x REAL NOT NULL,
                        center_y REAL NOT NULL,
                        confidence REAL NOT NULL,
                        class_id INTEGER NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # Create tracks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        global_id INTEGER NOT NULL,
                        creation_frame INTEGER NOT NULL,
                        last_seen_frame INTEGER NOT NULL,
                        total_detections INTEGER DEFAULT 0,
                        active_cameras TEXT,
                        avg_confidence REAL DEFAULT 0.0,
                        track_length INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # Create indices for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detections_session_frame 
                    ON detections (session_id, frame_number)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detections_global_id 
                    ON detections (global_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tracks_session_global 
                    ON tracks (session_id, global_id)
                """)
                
                conn.commit()
                self.logger.info(f"Database initialized: {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_session(self, session_record: SessionRecord) -> str:
        """
        Create a new tracking session
        
        Args:
            session_record (SessionRecord): Session information
            
        Returns:
            str: Session ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO sessions (
                        session_id, video1_path, video2_path, config, started_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    session_record.session_id,
                    session_record.video1_path,
                    session_record.video2_path,
                    session_record.config,
                    session_record.started_at
                ))
                
                conn.commit()
                self.logger.info(f"Created session: {session_record.session_id}")
                return session_record.session_id
                
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise
    
    def add_detection(self, detection: DetectionRecord) -> int:
        """
        Add a detection record to the database
        
        Args:
            detection (DetectionRecord): Detection information
            
        Returns:
            int: Detection ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO detections (
                        session_id, frame_number, camera_id, local_id, global_id,
                        bbox_x, bbox_y, bbox_width, bbox_height,
                        center_x, center_y, confidence, class_id, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    detection.session_id, detection.frame_number, detection.camera_id,
                    detection.local_id, detection.global_id,
                    detection.bbox_x, detection.bbox_y, detection.bbox_width, detection.bbox_height,
                    detection.center_x, detection.center_y, detection.confidence,
                    detection.class_id, detection.timestamp
                ))
                
                detection_id = cursor.lastrowid
                conn.commit()
                return detection_id
                
        except Exception as e:
            self.logger.error(f"Failed to add detection: {e}")
            raise
    
    def add_track(self, track: TrackRecord) -> int:
        """
        Add or update a track record
        
        Args:
            track (TrackRecord): Track information
            
        Returns:
            int: Track ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if track exists
                cursor.execute("""
                    SELECT id FROM tracks 
                    WHERE session_id = ? AND global_id = ?
                """, (track.session_id, track.global_id))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing track
                    cursor.execute("""
                        UPDATE tracks SET
                            last_seen_frame = ?,
                            total_detections = ?,
                            active_cameras = ?,
                            avg_confidence = ?,
                            track_length = ?,
                            status = ?,
                            updated_at = ?
                        WHERE session_id = ? AND global_id = ?
                    """, (
                        track.last_seen_frame, track.total_detections,
                        track.active_cameras, track.avg_confidence,
                        track.track_length, track.status, track.updated_at,
                        track.session_id, track.global_id
                    ))
                    track_id = existing[0]
                else:
                    # Insert new track
                    cursor.execute("""
                        INSERT INTO tracks (
                            session_id, global_id, creation_frame, last_seen_frame,
                            total_detections, active_cameras, avg_confidence,
                            track_length, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        track.session_id, track.global_id, track.creation_frame,
                        track.last_seen_frame, track.total_detections,
                        track.active_cameras, track.avg_confidence,
                        track.track_length, track.status,
                        track.created_at, track.updated_at
                    ))
                    track_id = cursor.lastrowid
                
                conn.commit()
                return track_id
                
        except Exception as e:
            self.logger.error(f"Failed to add/update track: {e}")
            raise
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a tracking session
        
        Args:
            session_id (str): Session ID
            
        Returns:
            dict: Session statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get session info
                cursor.execute("""
                    SELECT * FROM sessions WHERE session_id = ?
                """, (session_id,))
                session_data = cursor.fetchone()
                
                if not session_data:
                    return {}
                
                # Get detection count
                cursor.execute("""
                    SELECT COUNT(*) FROM detections WHERE session_id = ?
                """, (session_id,))
                total_detections = cursor.fetchone()[0]
                
                # Get track count
                cursor.execute("""
                    SELECT COUNT(*) FROM tracks WHERE session_id = ?
                """, (session_id,))
                total_tracks = cursor.fetchone()[0]
                
                # Get active tracks
                cursor.execute("""
                    SELECT COUNT(*) FROM tracks 
                    WHERE session_id = ? AND status = 'active'
                """, (session_id,))
                active_tracks = cursor.fetchone()[0]
                
                # Get camera-specific stats
                cursor.execute("""
                    SELECT camera_id, COUNT(*) 
                    FROM detections 
                    WHERE session_id = ? 
                    GROUP BY camera_id
                """, (session_id,))
                camera_stats = dict(cursor.fetchall())
                
                return {
                    'session_id': session_id,
                    'total_detections': total_detections,
                    'total_tracks': total_tracks,
                    'active_tracks': active_tracks,
                    'camera_stats': camera_stats,
                    'session_data': session_data
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get session stats: {e}")
            return {}
    
    def get_track_history(self, session_id: str, global_id: int) -> List[DetectionRecord]:
        """
        Get detection history for a specific track
        
        Args:
            session_id (str): Session ID
            global_id (int): Global track ID
            
        Returns:
            list: List of DetectionRecord objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM detections 
                    WHERE session_id = ? AND global_id = ?
                    ORDER BY frame_number
                """, (session_id, global_id))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                detections = []
                for row in rows:
                    data = dict(zip(columns, row))
                    # Convert timestamp string back to datetime
                    if data['timestamp']:
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    
                    detection = DetectionRecord(**data)
                    detections.append(detection)
                
                return detections
                
        except Exception as e:
            self.logger.error(f"Failed to get track history: {e}")
            return []
    
    def update_session_stats(self, session_id: str, **kwargs):
        """
        Update session statistics
        
        Args:
            session_id (str): Session ID
            **kwargs: Fields to update
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic update query
                fields = []
                values = []
                for key, value in kwargs.items():
                    fields.append(f"{key} = ?")
                    values.append(value)
                
                if fields:
                    query = f"UPDATE sessions SET {', '.join(fields)} WHERE session_id = ?"
                    values.append(session_id)
                    
                    cursor.execute(query, values)
                    conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update session stats: {e}")
    
    def export_session_data(self, session_id: str, output_path: str):
        """
        Export session data to JSON file
        
        Args:
            session_id (str): Session ID
            output_path (str): Output file path
        """
        try:
            stats = self.get_session_stats(session_id)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all detections
                cursor.execute("""
                    SELECT * FROM detections WHERE session_id = ?
                    ORDER BY frame_number, camera_id
                """, (session_id,))
                
                detections = []
                for row in cursor.fetchall():
                    detection_dict = dict(zip([desc[0] for desc in cursor.description], row))
                    detections.append(detection_dict)
                
                # Get all tracks
                cursor.execute("""
                    SELECT * FROM tracks WHERE session_id = ?
                    ORDER BY global_id
                """, (session_id,))
                
                tracks = []
                for row in cursor.fetchall():
                    track_dict = dict(zip([desc[0] for desc in cursor.description], row))
                    tracks.append(track_dict)
                
                export_data = {
                    'session_stats': stats,
                    'detections': detections,
                    'tracks': tracks,
                    'exported_at': datetime.now().isoformat()
                }
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Exported session data to: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to export session data: {e}")
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """
        Remove sessions older than specified days
        
        Args:
            days_old (int): Number of days to keep
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM detections 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE started_at < datetime('now', '-{} days')
                    )
                """.format(days_old))
                
                cursor.execute("""
                    DELETE FROM tracks 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE started_at < datetime('now', '-{} days')
                    )
                """.format(days_old))
                
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE started_at < datetime('now', '-{} days')
                """.format(days_old))
                
                conn.commit()
                self.logger.info(f"Cleaned up sessions older than {days_old} days")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create database instance
    db = TrackingDatabase("test_tracking.db")
    
    # Create test session
    session = SessionRecord(
        session_id="test_session_001",
        video1_path="video1.mp4",
        video2_path="video2.mp4",
        config=json.dumps({"distance_threshold": 50.0})
    )
    
    db.create_session(session)
    
    # Add test detection
    detection = DetectionRecord(
        session_id="test_session_001",
        frame_number=1,
        camera_id=1,
        local_id="det_001",
        global_id=1,
        bbox_x=100, bbox_y=100, bbox_width=50, bbox_height=80,
        center_x=125.0, center_y=140.0,
        confidence=0.85,
        class_id=0
    )
    
    db.add_detection(detection)
    
    # Add test track
    track = TrackRecord(
        session_id="test_session_001",
        global_id=1,
        creation_frame=1,
        last_seen_frame=1,
        total_detections=1,
        active_cameras=json.dumps([1]),
        avg_confidence=0.85,
        track_length=1
    )
    
    db.add_track(track)
    
    # Get statistics
    stats = db.get_session_stats("test_session_001")
    print("Session Statistics:", json.dumps(stats, indent=2, default=str))
