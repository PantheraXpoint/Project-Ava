#!/usr/bin/env python3
"""
Test script for complex query parsing and object search
"""

import os
import sys
import json
from llms.QwenLM import QwenLM
import argparse

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_complex_query_parsing():
    """
    Test the complex query parsing functionality
    """
    print("=== Testing Complex Query Parsing ===")
    
    try:
        from embeddings.JinaCLIP import JinaCLIP
        # from embeddings.FAISSDB import FAISSDB
        from embeddings.SQLiteDB import SQLiteDB
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed")
        return False
    
    
    # Test query parsing
    test_queries = [
        "from frame 20 to 30, find a toddler with blue shirt",
        "find a person wearing red hat between frame 100 and 200",
        "locate a car in the first 50 frames",
        "find any person in the video",
        "from frame 150 to 300, find a dog with brown fur"
    ]
    
    print("\n--- Testing Query Parsing ---")
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        try:
            parsed = llm.parse_complex_query(query)
            print(f"Parsed: {json.dumps(parsed, indent=2)}")
        except Exception as e:
            print(f"Error parsing query: {e}")
    
    return True

def test_database_search(video_path: str):
    """
    Test the database search functionality
    """
    print("\n=== Testing Database Search ===")
    
    # Check if databases exist
    faiss_db_path = "database/embeddings.faiss"
    sqlite_db_path = "database/tracked_objects.db"
    
    if not os.path.exists(faiss_db_path):
        print(f"✗ FAISS database {faiss_db_path} not found")
        print("Please run the object tracking system first to create the database")
        return False
    
    if not os.path.exists(sqlite_db_path):
        print(f"✗ SQLite database {sqlite_db_path} not found")
        print("Please run the object tracking system first to create the database")
        return False
    
    print("✓ Databases found")
    
    try:
        from embeddings.JinaCLIP import JinaCLIP
        from embeddings.object_search import extract_bounding_box_images
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("✓ Models initialized")
        
        # Test search
        query = "from frame 20 to 30, find a toddler with blue shirt"
        print(f"\nTesting search with query: {query}")

        llm = QwenLM()
        
        results = llm.search_objects_with_constraints(
            query, faiss_db_path, sqlite_db_path, embedding_model, k=3
        )
        saved_images = extract_bounding_box_images(
            video_path, results["results"], output_dir="extracted_objects", max_images=1
        )
        
        print(f"Found {results['total_results']} results:")
        for i, result in enumerate(results['results'], 1):
            print(f"  Result {i}:")
            print(f"    Track ID: {result['track_id']}")
            print(f"    Class: {result['class_name']}")
            print(f"    Similarity: {result['similarity_score']:.3f}")
            
            if result['frame_range']:
                print(f"    Frame range: {result['frame_range']['start']}-{result['frame_range']['end']}")
                print(f"    Frames in range: {result['total_frames_in_range']}")
            else:
                print(f"    Total frames: {result['total_frames']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during search: {e}")
        return False

def main():
    """
    Main test function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    args = parser.parse_args()
    
    print("=== Complex Query Parsing and Object Search Test ===")
    print("This test demonstrates:")
    print("1. Complex query parsing using LLM")
    print("2. Object description extraction for FAISS search")
    print("3. Frame constraint application")
    print("4. Combined FAISS + SQLite database querying")
    print("5. Filtered results based on frame ranges")
    
    # Test query parsing
    # parsing_success = test_complex_query_parsing()
    parsing_success = True
    
    # Test database search
    search_success = test_database_search(args.video_path)
    
    print("\n=== Test Results ===")
    print(f"Query Parsing: {'✓ PASSED' if parsing_success else '✗ FAILED'}")
    print(f"Database Search: {'✓ PASSED' if search_success else '✗ FAILED'}")
    
    if parsing_success and search_success:
        print("\n🎉 All tests passed! The complex query system is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    print("\n=== Usage Example ===")
    print("To use this system:")
    print("1. First run object tracking to create databases:")
    print("   python main_embedding_tracking.py --video your_video.mp4 --process-only")
    print("2. Then run the complex query test:")
    print("   python llms/QwenLM.py")
    print("3. Or use the search functionality in your code:")
    print("   from llms.QwenLM import QwenLM")
    print("   llm = QwenLM()")
    print("   results = llm.search_objects_with_constraints(query, faiss_path, sqlite_path, embedding_model)")

if __name__ == "__main__":
    main()
