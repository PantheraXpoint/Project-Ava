import torch
import json
import os
import sys
from llms.BaseModel import BaseLanguageModel
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

# Add embeddings directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))

# from FAISSDB import FAISSDB
from SQLiteDB import SQLiteDB
from JinaCLIP import JinaCLIP

class QwenLM(BaseLanguageModel):
    def __init__(self, model_type="Qwen/Qwen2.5-14B-Instruct-AWQ", tp=1):
        """
        Initialize the QwenLM model.

        Args:
            model_type (str): The type or name of the model.
            tp (int): The number of GPUs to use.
        """
        self.pipe = pipeline(model_type, 
                backend_config=TurbomindEngineConfig(session_len=8192*4, tp=tp, cache_max_entry_count=0.3))
    
    def generate_response(self, inputs, max_new_tokens=512, temperature=0.5):
        """
        Generate a response based on the inputs.

        Args:
            inputs (dict): Input data containing text
            {
                "text": str,
            }

        Returns:
            str: Generated response.
        """
        assert "text" in inputs.keys(), "Please provide a text prompt."
        gen_config = GenerationConfig(do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature)
    
        response = self.pipe(inputs["text"], gen_config=gen_config)
        text_response = response.text
        
        return text_response

    def batch_generate_response(self, batch_inputs, max_batch_size=64, max_new_tokens=512, temperature=0.5):
        prompts = []
        responses = []
        gen_config = GenerationConfig(do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature)
        
        for inputs in batch_inputs:
            prompts.append(inputs["text"])
        
        for i in range(0, len(prompts), max_batch_size):
            responses.extend(self.pipe(prompts[i:i+max_batch_size], gen_config=gen_config))
        
        responses = [response.text for response in responses]
            
        return responses

    def parse_complex_query(self, query: str):
        """
        Parse a complex query into object description and frame constraints
        
        Args:
            query: Complex query like "from frame 20 to 30, find a toddler with blue shirt"
            
        Returns:
            dict: Parsed query with object description and frame constraints
        """
        prompt = f"""
        Parse the following query about video object search into structured components:
        
        Query: "{query}"
        
        Extract:
        1. Object description (what to search for)
        2. Frame range (start_frame, end_frame) if specified
        3. Any other constraints
        
        Return as JSON format:
        {{
            "object_description": "description of the object to search for",
            "frame_range": {{"start": start_frame, "end": end_frame}} or null,
            "constraints": ["any additional constraints"]
        }}
        
        If no frame range is specified, set frame_range to null.
        """
        
        response = self.generate_response({"text": prompt})
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                return parsed
        except:
            pass
        
        # Fallback parsing if JSON extraction fails
        return {
            "object_description": query,
            "frame_range": None,
            "constraints": []
        }

    def search_objects_with_constraints(self, query: str, faiss_db_path: str, sqlite_db_path: str, 
                                       embedding_model: JinaCLIP, k: int = 5):
        """
        Search for objects using complex query with frame constraints
        
        Args:
            query: Complex query string
            faiss_db_path: Path to FAISS database
            sqlite_db_path: Path to SQLite database
            embedding_model: JinaCLIP embedding model
            k: Number of results to return
            
        Returns:
            dict: Search results with frame information
        """
        # Parse the complex query
        parsed_query = self.parse_complex_query(query)
        print(f"Parsed query: {parsed_query}")
        
        # Initialize databases
        faiss_db = FAISSDB(faiss_db_path, embedding_model.embedding_dim)
        sqlite_db = SQLiteDB(sqlite_db_path)
        
        # Search for objects using description
        object_description = parsed_query["object_description"]
        description_embedding = embedding_model.get_text_features([object_description])[0]
        
        # Search in FAISS database
        faiss_results = faiss_db.search(description_embedding, k)
        
        # Get detailed information from SQLite
        results = []
        for faiss_id, similarity_score, metadata in faiss_results:
            track_id = metadata['track_id']
            sqlite_info = sqlite_db.get_tracked_object(track_id)
            
            if sqlite_info:
                # Apply frame constraints if specified
                frame_range = parsed_query.get("frame_range")
                if frame_range:
                    start_frame = frame_range.get("start", 0)
                    end_frame = frame_range.get("end", float('inf'))
                    
                    # Filter frames within the specified range
                    filtered_frames = []
                    filtered_bboxes = []
                    filtered_confidences = []
                    
                    for i, frame_num in enumerate(sqlite_info['frame_numbers']):
                        if start_frame <= frame_num <= end_frame:
                            filtered_frames.append(frame_num)
                            filtered_bboxes.append(sqlite_info['bbox_history'][i])
                            filtered_confidences.append(sqlite_info['confidence_history'][i])
                    
                    if filtered_frames:  # Only include if there are frames in the range
                        result = {
                            'track_id': track_id,
                            'similarity_score': similarity_score,
                            'class_name': sqlite_info['class_name'],
                            'class_id': sqlite_info['class_id'],
                            'frame_range': frame_range,
                            'filtered_frames': filtered_frames,
                            'filtered_bboxes': filtered_bboxes,
                            'filtered_confidences': filtered_confidences,
                            'total_frames_in_range': len(filtered_frames),
                            'all_frames': sqlite_info['frame_numbers'],
                            'faiss_metadata': metadata
                        }
                        results.append(result)
                else:
                    # No frame constraints - return all frames
                    result = {
                        'track_id': track_id,
                        'similarity_score': similarity_score,
                        'class_name': sqlite_info['class_name'],
                        'class_id': sqlite_info['class_id'],
                        'frame_range': None,
                        'all_frames': sqlite_info['frame_numbers'],
                        'all_bboxes': sqlite_info['bbox_history'],
                        'all_confidences': sqlite_info['confidence_history'],
                        'total_frames': sqlite_info['total_frames'],
                        'faiss_metadata': metadata
                    }
                    results.append(result)
        
        return {
            'query': query,
            'parsed_query': parsed_query,
            'results': results,
            'total_results': len(results)
        }


if __name__ == "__main__":
    """
    Test complex query parsing and object search
    """
    print("=== Complex Query Parsing and Object Search Test ===")
    
    # Initialize QwenLM
    print("Initializing QwenLM...")
    llm = QwenLM()
    
    # Initialize embedding model
    print("Initializing JinaCLIP embedding model...")
    try:
        embedding_model = JinaCLIP("jinaai/jina-clip-v1")
        print("JinaCLIP model initialized successfully")
    except Exception as e:
        print(f"Error initializing JinaCLIP model: {e}")
        print("Please ensure the model is available and dependencies are installed")
        exit(1)
    
    # Test queries
    test_queries = [
        "from frame 20 to 30, find a toddler with blue shirt",
        "find a person wearing red hat between frame 100 and 200",
        "locate a car in the first 50 frames",
        "find any person in the video",
        "from frame 150 to 300, find a dog with brown fur"
    ]
    
    # Database paths
    faiss_db_path = "embeddings.faiss"
    sqlite_db_path = "tracked_objects.db"
    
    print(f"\nUsing databases:")
    print(f"FAISS: {faiss_db_path}")
    print(f"SQLite: {sqlite_db_path}")
    
    # Check if databases exist
    if not os.path.exists(faiss_db_path):
        print(f"Warning: FAISS database {faiss_db_path} not found")
        print("Please run the object tracking system first to create the database")
        exit(1)
    
    if not os.path.exists(sqlite_db_path):
        print(f"Warning: SQLite database {sqlite_db_path} not found")
        print("Please run the object tracking system first to create the database")
        exit(1)
    
    print("\n=== Testing Complex Query Parsing ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")
        
        # Parse the query
        parsed = llm.parse_complex_query(query)
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
        
        # Search for objects
        print(f"\nSearching for objects...")
        try:
            results = llm.search_objects_with_constraints(
                query, faiss_db_path, sqlite_db_path, embedding_model, k=3
            )
            
            print(f"Found {results['total_results']} results:")
            for j, result in enumerate(results['results'], 1):
                print(f"  Result {j}:")
                print(f"    Track ID: {result['track_id']}")
                print(f"    Class: {result['class_name']}")
                print(f"    Similarity: {result['similarity_score']:.3f}")
                
                if result['frame_range']:
                    print(f"    Frame range: {result['frame_range']['start']}-{result['frame_range']['end']}")
                    print(f"    Frames in range: {result['total_frames_in_range']}")
                    print(f"    Filtered frames: {result['filtered_frames'][:5]}{'...' if len(result['filtered_frames']) > 5 else ''}")
                else:
                    print(f"    Total frames: {result['total_frames']}")
                    print(f"    All frames: {result['all_frames'][:5]}{'...' if len(result['all_frames']) > 5 else ''}")
                
                print()
        except Exception as e:
            print(f"Error during search: {e}")
    
    print("\n=== Test Complete ===")
    print("This test demonstrates:")
    print("1. Complex query parsing using LLM")
    print("2. Object description extraction for FAISS search")
    print("3. Frame constraint application")
    print("4. Combined FAISS + SQLite database querying")
    print("5. Filtered results based on frame ranges")