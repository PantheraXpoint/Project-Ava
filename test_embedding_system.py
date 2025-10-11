#!/usr/bin/env python3
"""
Test script for the embedding-based object tracking system
"""

import os
import sys
import tempfile
import shutil

# Add embeddings directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'embeddings'))

def test_database_functionality():
    """Test FAISS and SQLite database functionality"""
    print("Testing database functionality...")
    
    try:
        from FAISSDB import FAISSDB
        from SQLiteDB import SQLiteDB
        import numpy as np
        
        # Create temporary databases
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_path = os.path.join(temp_dir, "test.faiss")
            sqlite_path = os.path.join(temp_dir, "test.db")
            
            # Test FAISS database
            print("  Testing FAISS database...")
            faiss_db = FAISSDB(faiss_path, embedding_dim=512)
            
            # Add test embedding
            test_embedding = np.random.rand(512).astype(np.float32)
            faiss_id = faiss_db.add_embedding(test_embedding, track_id=1, metadata={'test': 'data'})
            print(f"    Added embedding with FAISS ID: {faiss_id}")
            
            # Search test
            results = faiss_db.search(test_embedding, k=1)
            print(f"    Search returned {len(results)} results")
            
            # Test SQLite database
            print("  Testing SQLite database...")
            sqlite_db = SQLiteDB(sqlite_path)
            
            # Add test tracked object
            success = sqlite_db.add_tracked_object(
                track_id=1,
                class_id=0,
                class_name="person",
                bbox_history=[[100, 100, 200, 200]],
                confidence_history=[0.9],
                frame_numbers=[1]
            )
            print(f"    Added tracked object: {success}")
            
            # Retrieve test
            obj = sqlite_db.get_tracked_object(1)
            print(f"    Retrieved object: {obj is not None}")
            
        print("  Database tests passed!")
        return True
        
    except Exception as e:
        print(f"  Database test failed: {e}")
        return False

def test_embedding_model():
    """Test JinaCLIP embedding model"""
    print("Testing JinaCLIP embedding model...")
    
    try:
        from JinaCLIP import JinaCLIP
        import numpy as np
        from PIL import Image
        
        # Initialize model
        model = JinaCLIP("jinaai/jina-clip-v1")
        print("    Model initialized successfully")
        
        # Test text embedding
        text_embedding = model.get_text_features(["a person walking"])
        print(f"    Text embedding shape: {text_embedding.shape}")
        
        # Test image embedding
        test_image = Image.new('RGB', (224, 224), color='red')
        image_embedding = model.get_image_features([test_image])
        print(f"    Image embedding shape: {image_embedding.shape}")
        
        print("  Embedding model tests passed!")
        return True
        
    except Exception as e:
        print(f"  Embedding model test failed: {e}")
        return False

def test_object_search():
    """Test object search functionality"""
    print("Testing object search functionality...")
    
    try:
        from object_search import ObjectSearch
        from JinaCLIP import JinaCLIP
        import numpy as np
        
        # Create temporary databases
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_path = os.path.join(temp_dir, "test.faiss")
            sqlite_path = os.path.join(temp_dir, "test.db")
            
            # Initialize model and search system
            model = JinaCLIP("jinaai/jina-clip-v1")
            search_system = ObjectSearch(faiss_path, sqlite_path, model)
            
            # Test search (should return empty results since no data)
            results = search_system.search_by_description("a person", k=5)
            print(f"    Search returned {len(results)} results")
            
            # Test statistics
            stats = search_system.get_track_statistics()
            print(f"    Statistics: {stats}")
            
        print("  Object search tests passed!")
        return True
        
    except Exception as e:
        print(f"  Object search test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running embedding system tests...\n")
    
    tests = [
        ("Database Functionality", test_database_functionality),
        ("Embedding Model", test_embedding_model),
        ("Object Search", test_object_search)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"=== {test_name} ===")
        if test_func():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("All tests passed! The embedding system is ready to use.")
    else:
        print("Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
