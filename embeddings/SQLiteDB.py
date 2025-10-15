import sqlite3
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class SQLiteDB:
    """
    SQLite database for storing tracked objects information
    """
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tracked_objects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracked_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER UNIQUE NOT NULL,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL,
                total_frames INTEGER NOT NULL,
                bbox_history TEXT NOT NULL,  -- JSON string of bbox coordinates
                confidence_history TEXT NOT NULL,  -- JSON string of confidence scores
                frame_numbers TEXT NOT NULL,  -- JSON string of frame numbers
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create video_info table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT UNIQUE NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                fps REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                processing_fps REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_tracked_object(self, track_id: int, class_id: int, class_name: str, 
                          bbox_history: List[List[int]], confidence_history: List[float], 
                          frame_numbers: List[int]) -> bool:
        """
        Add or update tracked object information
        
        Args:
            track_id: Unique track ID
            class_id: Object class ID
            class_name: Object class name
            bbox_history: List of bounding box coordinates
            confidence_history: List of confidence scores
            frame_numbers: List of frame numbers where object was detected
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if track_id already exists
            cursor.execute('SELECT id FROM tracked_objects WHERE track_id = ? AND class_id = ?', (track_id, class_id))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute('''
                    UPDATE tracked_objects 
                    SET class_id = ?, class_name = ?, last_frame = ?, total_frames = ?,
                        bbox_history = ?, confidence_history = ?, frame_numbers = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE track_id = ?
                ''', (class_id, class_name, max(frame_numbers), len(frame_numbers),
                      json.dumps(bbox_history), json.dumps(confidence_history),
                      json.dumps(frame_numbers), track_id))
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO tracked_objects 
                    (track_id, class_id, class_name, first_frame, last_frame, total_frames,
                     bbox_history, confidence_history, frame_numbers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (track_id, class_id, class_name, min(frame_numbers), max(frame_numbers),
                      len(frame_numbers), json.dumps(bbox_history), json.dumps(confidence_history),
                      json.dumps(frame_numbers)))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error adding tracked object: {e}")
            return False
    
    def get_tracked_object(self, track_id: int) -> Optional[Dict]:
        """
        Get tracked object information by track_id
        
        Args:
            track_id: Track ID to search for
            
        Returns:
            Dictionary with object information or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM tracked_objects WHERE track_id = ?
            ''', (track_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'track_id': row[1],
                    'class_id': row[2],
                    'class_name': row[3],
                    'first_frame': row[4],
                    'last_frame': row[5],
                    'total_frames': row[6],
                    'bbox_history': json.loads(row[7]),
                    'confidence_history': json.loads(row[8]),
                    'frame_numbers': json.loads(row[9]),
                    'created_at': row[10],
                    'updated_at': row[11]
                }
            return None
            
        except Exception as e:
            print(f"Error getting tracked object: {e}")
            return None
    
    def get_all_tracked_objects(self) -> List[Dict]:
        """
        Get all tracked objects
        
        Returns:
            List of dictionaries with object information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tracked_objects ORDER BY track_id')
            rows = cursor.fetchall()
            conn.close()
            
            objects = []
            for row in rows:
                objects.append({
                    'id': row[0],
                    'track_id': row[1],
                    'class_id': row[2],
                    'class_name': row[3],
                    'first_frame': row[4],
                    'last_frame': row[5],
                    'total_frames': row[6],
                    'bbox_history': json.loads(row[7]),
                    'confidence_history': json.loads(row[8]),
                    'frame_numbers': json.loads(row[9]),
                    'created_at': row[10],
                    'updated_at': row[11]
                })
            
            return objects
            
        except Exception as e:
            print(f"Error getting all tracked objects: {e}")
            return []
    
    def add_video_info(self, video_path: str, width: int, height: int, 
                      fps: float, total_frames: int, processing_fps: float) -> bool:
        """
        Add video information
        
        Args:
            video_path: Path to video file
            width: Video width
            height: Video height
            fps: Video FPS
            total_frames: Total number of frames
            processing_fps: Processing FPS used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO video_info 
                (video_path, width, height, fps, total_frames, processing_fps)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (video_path, width, height, fps, total_frames, processing_fps))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error adding video info: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM video_info WHERE video_path = ?', (video_path,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'video_path': row[1],
                    'width': row[2],
                    'height': row[3],
                    'fps': row[4],
                    'total_frames': row[5],
                    'processing_fps': row[6],
                    'created_at': row[7]
                }
            return None
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None
