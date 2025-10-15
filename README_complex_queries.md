# Complex Query Parsing and Object Search

This system extends the embedding-based object tracking with **intelligent query parsing** using LLM to handle complex natural language queries about video objects.

## Features

- **Natural Language Queries**: Parse complex queries like "from frame 20 to 30, find a toddler with blue shirt"
- **LLM-Powered Parsing**: Uses QwenLM to extract object descriptions and frame constraints
- **Dual Database Search**: Combines FAISS (embeddings) and SQLite (tracking data) for comprehensive results
- **Frame Range Filtering**: Automatically filters results based on specified frame ranges
- **Intelligent Matching**: Finds objects using semantic similarity and temporal constraints

## How It Works

### 1. Query Parsing
```
Input: "from frame 20 to 30, find a toddler with blue shirt"
Output: {
  "object_description": "a toddler with blue shirt",
  "frame_range": {"start": 20, "end": 30},
  "constraints": []
}
```

### 2. Object Search Process
1. **Extract Description**: "a toddler with blue shirt" → FAISS embedding search
2. **Find Similar Objects**: Get track IDs from FAISS database
3. **Apply Frame Constraints**: Filter SQLite results to frames 20-30
4. **Return Filtered Results**: Only objects appearing in specified frame range

### 3. Database Integration
- **FAISS Database**: Stores embeddings for semantic similarity search
- **SQLite Database**: Stores detailed tracking information (frames, bboxes, confidence)
- **Combined Query**: Uses both databases for comprehensive results

## Usage

### Basic Usage

```python
from llms.QwenLM import QwenLM
from embeddings.JinaCLIP import JinaCLIP

# Initialize models
llm = QwenLM()
embedding_model = JinaCLIP("jinaai/jina-clip-v1")

# Search with complex query
results = llm.search_objects_with_constraints(
    query="from frame 20 to 30, find a toddler with blue shirt",
    faiss_db_path="embeddings.faiss",
    sqlite_db_path="tracked_objects.db",
    embedding_model=embedding_model,
    k=5
)

print(f"Found {results['total_results']} results")
for result in results['results']:
    print(f"Track ID: {result['track_id']}")
    print(f"Class: {result['class_name']}")
    print(f"Frames in range: {result['total_frames_in_range']}")
```

### Test the System

```bash
# Run the test script
python test_complex_query.py

# Or run the main QwenLM test
python llms/QwenLM.py
```

## Query Examples

### Frame Range Queries
```
"from frame 20 to 30, find a toddler with blue shirt"
"find a person wearing red hat between frame 100 and 200"
"locate a car in the first 50 frames"
"from frame 150 to 300, find a dog with brown fur"
```

### General Queries
```
"find any person in the video"
"locate all cars"
"find objects with red color"
```

### Complex Descriptions
```
"find a person wearing a blue shirt and black pants"
"locate a small child with blonde hair"
"find a vehicle with white color"
```

## API Reference

### `parse_complex_query(query: str)`
Parses a natural language query into structured components.

**Parameters:**
- `query`: Natural language query string

**Returns:**
```python
{
    "object_description": "description of the object to search for",
    "frame_range": {"start": start_frame, "end": end_frame} or null,
    "constraints": ["any additional constraints"]
}
```

### `search_objects_with_constraints(query, faiss_db_path, sqlite_db_path, embedding_model, k=5)`
Searches for objects using complex query with frame constraints.

**Parameters:**
- `query`: Complex query string
- `faiss_db_path`: Path to FAISS database
- `sqlite_db_path`: Path to SQLite database
- `embedding_model`: JinaCLIP embedding model
- `k`: Number of results to return

**Returns:**
```python
{
    'query': original_query,
    'parsed_query': parsed_components,
    'results': [
        {
            'track_id': unique_track_id,
            'similarity_score': similarity_score,
            'class_name': object_class,
            'frame_range': frame_constraints,
            'filtered_frames': frames_in_range,
            'total_frames_in_range': count,
            # ... additional metadata
        }
    ],
    'total_results': number_of_results
}
```

## Result Structure

### With Frame Constraints
```python
{
    'track_id': 123,
    'similarity_score': 0.85,
    'class_name': 'person',
    'frame_range': {'start': 20, 'end': 30},
    'filtered_frames': [22, 25, 28],
    'filtered_bboxes': [[x1, y1, x2, y2], ...],
    'total_frames_in_range': 3,
    'all_frames': [1, 5, 22, 25, 28, 35, 40],  # All frames for this track
    'faiss_metadata': {...}
}
```

### Without Frame Constraints
```python
{
    'track_id': 123,
    'similarity_score': 0.85,
    'class_name': 'person',
    'frame_range': None,
    'all_frames': [1, 5, 22, 25, 28, 35, 40],
    'all_bboxes': [[x1, y1, x2, y2], ...],
    'total_frames': 7,
    'faiss_metadata': {...}
}
```

## Prerequisites

1. **Databases**: Must have processed videos to create FAISS and SQLite databases
2. **Models**: QwenLM and JinaCLIP models must be available
3. **Dependencies**: All required packages installed

## Setup

1. **Process Videos First**:
```bash
python main_embedding_tracking.py --video your_video.mp4 --process-only
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Test the System**:
```bash
python test_complex_query.py
```

## Advanced Usage

### Custom Query Parsing
```python
# Parse query without searching
parsed = llm.parse_complex_query("from frame 20 to 30, find a toddler with blue shirt")
print(parsed['object_description'])  # "a toddler with blue shirt"
print(parsed['frame_range'])         # {"start": 20, "end": 30}
```

### Batch Processing
```python
queries = [
    "find a person in the first 100 frames",
    "locate a car between frame 200 and 300",
    "find any animal in the video"
]

results = []
for query in queries:
    result = llm.search_objects_with_constraints(
        query, faiss_db_path, sqlite_db_path, embedding_model
    )
    results.append(result)
```

### Custom Frame Ranges
```python
# The system automatically handles various frame range formats:
"from frame 20 to 30"           # Exact range
"between frame 100 and 200"     # Alternative format
"in the first 50 frames"        # Start from beginning
"in the last 100 frames"        # End at video end
```

## Troubleshooting

### Common Issues

1. **"Database not found"**: Run object tracking first to create databases
2. **"Model not available"**: Ensure QwenLM and JinaCLIP models are installed
3. **"No results found"**: Check if objects exist in the specified frame range
4. **"Parsing failed"**: The LLM might not understand the query format

### Debug Mode
```python
# Enable verbose output
results = llm.search_objects_with_constraints(
    query, faiss_db_path, sqlite_db_path, embedding_model, k=5
)
print(f"Parsed query: {results['parsed_query']}")
print(f"Total results: {results['total_results']}")
```

## Performance Notes

- **Query Parsing**: ~1-2 seconds per query (depends on LLM model)
- **Database Search**: ~0.1-0.5 seconds (depends on database size)
- **Frame Filtering**: ~0.01-0.1 seconds (depends on result size)
- **Total Time**: ~1-3 seconds per complex query

## Future Enhancements

- **Temporal Queries**: "find objects that appear before/after frame X"
- **Multi-Object Queries**: "find a person and a car in the same frame"
- **Attribute Queries**: "find objects with high confidence"
- **Spatial Queries**: "find objects in the top-left corner"
- **Motion Queries**: "find objects that are moving fast"

This system provides a powerful interface for querying video content using natural language, making it easy to find specific objects at specific times in your video data.
