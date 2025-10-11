#!/usr/bin/env python3
"""
Main script for embedding-based object tracking and search
"""

import os
import sys
import argparse
from typing import List, Dict

# Add embeddings directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'embeddings'))

from JinaCLIP import JinaCLIP
from embeddings.object_search import ObjectSearch
from AVA.object_detect import ObjectDetectorTracker

def process_video_with_embeddings(video_path: str, faiss_db_path: str = "embeddings.faiss", 
                                 sqlite_db_path: str = "tracked_objects.db"):
    """
    Process video with embedding generation for new track IDs
    
    Args:
        video_path: Path to input video
        faiss_db_path: Path to FAISS database
        sqlite_db_path: Path to SQLite database
    """
    print("Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("JinaCLIP model initialized successfully")
    except Exception as e:
        print(f"Error initializing JinaCLIP model: {e}")
        return None
    
    print("Initializing object detector with embedding support...")
    detector = ObjectDetectorTracker(
        model_path="yolo11n.pt",
        conf_threshold=0.5,
        iou_threshold=0.5,
        tracker_config="config/tracker.yaml",
        embedding_model=embedding_model,
        faiss_db_path=faiss_db_path,
        sqlite_db_path=sqlite_db_path
    )
    
    print(f"Processing video: {video_path}")
    tracked_objects = detector.process_video_tracking_only(video_path)
    
    return embedding_model, detector

def search_and_extract_objects(description: str, video_path: str, output_dir: str,
                              faiss_db_path: str = "embeddings.faiss",
                              sqlite_db_path: str = "tracked_objects.db",
                              k: int = 5, max_images: int = 10):
    """
    Search for objects by description and extract bounding box images
    
    Args:
        description: Text description to search for
        video_path: Path to original video
        output_dir: Directory to save extracted images
        faiss_db_path: Path to FAISS database
        sqlite_db_path: Path to SQLite database
        k: Number of search results to return
        max_images: Maximum images to extract per track
    """
    print("Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("JinaCLIP model initialized successfully")
    except Exception as e:
        print(f"Error initializing JinaCLIP model: {e}")
        return [], []
    
    print("Initializing object search system...")
    search_system = ObjectSearch(faiss_db_path, sqlite_db_path, embedding_model)
    
    print(f"Searching for objects matching: '{description}'")
    search_results, saved_images = search_system.search_and_extract(
        description, video_path, output_dir, k, max_images
    )
    
    return search_results, saved_images

def get_database_statistics(faiss_db_path: str = "embeddings.faiss",
                           sqlite_db_path: str = "tracked_objects.db"):
    """
    Get statistics about the databases
    
    Args:
        faiss_db_path: Path to FAISS database
        sqlite_db_path: Path to SQLite database
    """
    print("Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("JinaCLIP model initialized successfully")
    except Exception as e:
        print(f"Error initializing JinaCLIP model: {e}")
        return
    
    print("Initializing object search system...")
    search_system = ObjectSearch(faiss_db_path, sqlite_db_path, embedding_model)
    
    print("Getting database statistics...")
    stats = search_system.get_track_statistics()
    
    print("\n=== Database Statistics ===")
    print(f"Total tracks: {stats['total_tracks']}")
    print(f"Total frames tracked: {stats['total_frames_tracked']}")
    print(f"Average track length: {stats['average_track_length']:.2f} frames")
    print("\nClass distribution:")
    for class_name, count in stats['class_distribution'].items():
        print(f"  {class_name}: {count} tracks")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Embedding-based Object Tracking and Search')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--description', type=str,
                       help='Text description to search for')
    parser.add_argument('--output-dir', type=str, default='extracted_objects',
                       help='Directory to save extracted images')
    parser.add_argument('--faiss-db', type=str, default='database/embeddings.faiss',
                       help='Path to FAISS database file')
    parser.add_argument('--sqlite-db', type=str, default='database/tracked_objects.db',
                       help='Path to SQLite database file')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of search results to return')
    parser.add_argument('--max-images', type=int, default=1,
                       help='Maximum images to extract per track')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--process-only', action='store_true',
                       help='Only process video without searching')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} does not exist")
        return
    
    # Process video with embeddings
    if args.process_only or not args.description:
        print("Processing video with embedding generation...")
        result = process_video_with_embeddings(args.video, args.faiss_db, args.sqlite_db)
        if result is None:
            print("Failed to process video")
            return
        embedding_model, detector = result
        print("Video processing completed successfully!")
    
    # Show statistics
    if args.stats:
        get_database_statistics(args.faiss_db, args.sqlite_db)
    
    # Search and extract objects
    if args.description:
        print(f"\nSearching for objects matching: '{args.description}'")
        search_results, saved_images = search_and_extract_objects(
            args.description, args.video, args.output_dir,
            args.faiss_db, args.sqlite_db, args.k, args.max_images
        )
        
        if search_results:
            print(f"\nFound {len(search_results)} matching objects:")
            for i, result in enumerate(search_results):
                print(f"  {i+1}. Track ID: {result['track_id']}, "
                      f"Class: {result['class_name']}, "
                      f"Similarity: {result['similarity_score']:.3f}")
            
            print(f"\nExtracted {len(saved_images)} images to {args.output_dir}")
            for image_path in saved_images:
                print(f"  - {image_path}")
        else:
            print("No objects found matching the description")

if __name__ == "__main__":
    main()
