#!/usr/bin/env python3
"""
Script to merge multiple JSON result files while preserving the order of video_id and question_id.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSON file and return the parsed data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def merge_results(files: List[str], output_file: str) -> None:
    """
    Merge multiple JSON result files into a single file.
    The merged data will be sorted by video_id first, then by question_id.
    """
    print("Starting merge process...")
    
    # Load all data from the files
    all_data = []
    for file_path in files:
        data = load_json_file(file_path)
        all_data.extend(data)
    
    print(f"Total entries loaded: {len(all_data)}")
    
    # Sort by video_id first, then by question_id
    print("Sorting data by video_id and question_id...")
    all_data.sort(key=lambda x: (x['video_id'], x['question_id']))
    
    # Check for duplicates
    seen = set()
    duplicates = []
    unique_data = []
    
    for item in all_data:
        key = (item['video_id'], item['question_id'])
        if key in seen:
            duplicates.append(key)
            print(f"Warning: Duplicate found - video_id: {item['video_id']}, question_id: {item['question_id']}")
        else:
            seen.add(key)
            unique_data.append(item)
    
    if duplicates:
        print(f"Found {len(duplicates)} duplicate entries. Keeping only the first occurrence of each.")
        print(f"Final unique entries: {len(unique_data)}")
    else:
        print("No duplicates found.")
        unique_data = all_data
    
    # Write merged data to output file
    print(f"Writing merged data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=4, ensure_ascii=False)
    
    print(f"Merge completed successfully!")
    print(f"Output file: {output_file}")
    print(f"Total entries in merged file: {len(unique_data)}")
    
    # Print summary statistics
    video_ids = sorted(set(item['video_id'] for item in unique_data))
    print(f"Video ID range: {min(video_ids)} to {max(video_ids)}")
    print(f"Number of unique videos: {len(video_ids)}")


def main():
    """Main function to handle command line arguments and execute merge."""
    if len(sys.argv) < 3:
        print("Usage: python merge_results.py <output_file> <input_file1> [input_file2] ...")
        print("Example: python merge_results.py merged_results.json file1.json file2.json file3.json file4.json")
        sys.exit(1)
    
    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    
    # Check if all input files exist
    for file_path in input_files:
        if not Path(file_path).exists():
            print(f"Error: Input file {file_path} does not exist")
            sys.exit(1)
    
    print(f"Input files: {input_files}")
    print(f"Output file: {output_file}")
    print("-" * 50)
    
    merge_results(input_files, output_file)


if __name__ == "__main__":
    main()
