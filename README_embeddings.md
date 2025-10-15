# Embedding-Based Object Tracking and Search System

This system extends the existing object detection and tracking functionality with embedding-based search capabilities using JinaCLIP and FAISS.

## Features

- **Object Detection & Tracking**: Uses YOLOv11 with built-in tracking
- **Embedding Generation**: Automatically generates embeddings for new track IDs using JinaCLIP
- **FAISS Database**: Stores and searches embeddings efficiently
- **SQLite Database**: Stores detailed tracking information
- **Text-based Search**: Search for objects using natural language descriptions
- **Image Extraction**: Extract bounding box images from video based on search results

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_embeddings.txt
```

2. For better performance with CUDA, install faiss-gpu instead of faiss-cpu:
```bash
pip install faiss-gpu
```

## Usage

### 1. Process Video with Embeddings

Process a video and generate embeddings for new track IDs:

```bash
python main_embedding_tracking.py --video path/to/video.mp4 --process-only
```

Or use the original object detection script with embedding support:

```bash
python AVA/object_detect.py --video path/to/video.mp4 --tracking-only --enable-embeddings
```

### 2. Search and Extract Objects

Search for objects by description and extract bounding box images:

```bash
python main_embedding_tracking.py \
    --video path/to/video.mp4 \
    --description "a person wearing red shirt" \
    --output-dir extracted_objects \
    --k 5 \
    --max-images 10
```

### 3. View Database Statistics

Get statistics about tracked objects:

```bash
python main_embedding_tracking.py --video path/to/video.mp4 --stats
```

## API Usage

### Initialize the System

```python
from embeddings.JinaCLIP import JinaCLIP
from embeddings.FAISSDB import FAISSDB
from embeddings.SQLiteDB import SQLiteDB
from embeddings.object_search import ObjectSearch

# Initialize embedding model
embedding_model = JinaCLIP("jinaai/jina-clip-v1")

# Initialize databases
faiss_db = FAISSDB("embeddings.faiss", embedding_model.embedding_dim)
sqlite_db = SQLiteDB("tracked_objects.db")

# Initialize search system
search_system = ObjectSearch("embeddings.faiss", "tracked_objects.db", embedding_model)
```

### Process Video with Embeddings

```python
from AVA.object_detect import ObjectDetectorTracker

# Initialize detector with embedding support
detector = ObjectDetectorTracker(
    model_path="yolo11n.pt",
    embedding_model=embedding_model,
    faiss_db_path="embeddings.faiss",
    sqlite_db_path="tracked_objects.db"
)

# Process video
tracked_objects = detector.process_video_tracking_only("path/to/video.mp4")
```

### Search for Objects

```python
# Search by description
results = search_system.search_by_description("a person walking", k=5)

# Extract bounding box images
saved_images = search_system.extract_bounding_box_images(
    "path/to/video.mp4", results, "output_dir", max_images=10
)
```

### Complete Search and Extract Pipeline

```python
# One-step search and extract
results, saved_images = search_system.search_and_extract(
    "a person wearing blue shirt",
    "path/to/video.mp4",
    "extracted_objects",
    k=5,
    max_images=10
)
```

## Database Schema

### FAISS Database
- Stores embeddings for efficient similarity search
- Maps track IDs to embeddings
- Supports cosine similarity search

### SQLite Database

#### tracked_objects table
- `track_id`: Unique track identifier
- `class_id`: Object class ID
- `class_name`: Object class name
- `first_frame`: First frame where object appears
- `last_frame`: Last frame where object appears
- `total_frames`: Total number of frames tracked
- `bbox_history`: JSON array of bounding box coordinates
- `confidence_history`: JSON array of confidence scores
- `frame_numbers`: JSON array of frame numbers

#### video_info table
- `video_path`: Path to video file
- `width`: Video width
- `height`: Video height
- `fps`: Video FPS
- `total_frames`: Total number of frames
- `processing_fps`: Processing FPS used

## File Structure

```
Project-AVAS/
├── embeddings/
│   ├── BaseEmbeddingModel.py
│   ├── JinaCLIP.py
│   ├── FAISSDB.py
│   ├── SQLiteDB.py
│   └── object_search.py
├── AVA/
│   └── object_detect.py (modified)
├── main_embedding_tracking.py
├── test_embedding_system.py
├── requirements_embeddings.txt
└── README_embeddings.md
```

## Testing

Run the test script to verify the system:

```bash
python test_embedding_system.py
```

## Performance Notes

- **Embedding Generation**: Only generates embeddings for new track IDs to avoid redundancy
- **Processing Speed**: Embedding generation adds some overhead but is done asynchronously
- **Memory Usage**: FAISS database is memory-efficient for large-scale similarity search
- **Storage**: Embeddings are stored in binary format for efficient disk usage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in JinaCLIP or use CPU version
2. **FAISS Import Error**: Install faiss-cpu or faiss-gpu
3. **Model Download Issues**: Ensure internet connection for first-time model download

### Performance Optimization

1. Use `faiss-gpu` for faster similarity search on CUDA-enabled systems
2. Adjust processing FPS in `object_detect.py` for balance between speed and accuracy
3. Use smaller embedding dimensions if memory is limited

## Examples

### Example 1: Basic Video Processing

```bash
# Process video and generate embeddings
python main_embedding_tracking.py --video sample_video.mp4 --process-only
```

### Example 2: Search for Specific Objects

```bash
# Search for people and extract images
python main_embedding_tracking.py \
    --video sample_video.mp4 \
    --description "a person" \
    --output-dir people_images \
    --k 10 \
    --max-images 5
```

### Example 3: Search for Specific Attributes

```bash
# Search for objects with specific attributes
python main_embedding_tracking.py \
    --video sample_video.mp4 \
    --description "a person wearing a hat" \
    --output-dir hat_people \
    --k 3
```

## Advanced Usage

### Custom Embedding Models

You can use different embedding models by modifying the `JinaCLIP` initialization:

```python
# Use different JinaCLIP model
embedding_model = JinaCLIP("jinaai/jina-clip-v2")  # If available
```

### Custom Database Paths

```python
# Use custom database paths
faiss_db = FAISSDB("custom_embeddings.faiss", embedding_dim=512)
sqlite_db = SQLiteDB("custom_tracking.db")
```

### Batch Processing

For processing multiple videos:

```python
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video_path in video_paths:
    detector.process_video_tracking_only(video_path)
```

## License

This project extends the existing AVA system with embedding-based search capabilities.
