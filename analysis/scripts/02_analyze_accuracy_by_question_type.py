#!/usr/bin/env python3
"""
Script to analyze accuracy by question type from the merged results.
Groups results by question_type and calculates accuracy metrics for each type.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple


def load_merged_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the merged JSON data."""
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


def analyze_accuracy_by_type(data: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Analyze accuracy by question type.
    Returns a tuple with (results_by_type, overall_metrics).
    """
    # Group data by question type
    type_data = defaultdict(list)
    
    for item in data:
        for question_type in item['question_type']:
            type_data[question_type].append(item)
    
    # Calculate overall accuracy (unique questions only)
    total_unique_questions = len(data)
    total_correct_answers = sum(1 for item in data if item['answer'] == item['response'])
    overall_accuracy = total_correct_answers / total_unique_questions if total_unique_questions > 0 else 0
    
    # Calculate overall per-video accuracy
    video_accuracy = {}
    for item in data:
        video_id = item['video_id']
        if video_id not in video_accuracy:
            video_accuracy[video_id] = {'correct': 0, 'total': 0}
        video_accuracy[video_id]['total'] += 1
        if item['answer'] == item['response']:
            video_accuracy[video_id]['correct'] += 1
    
    # Calculate average per-video accuracy
    per_video_accuracies = [acc['correct']/acc['total'] for acc in video_accuracy.values()]
    overall_avg_per_video_accuracy = sum(per_video_accuracies) / len(per_video_accuracies) if per_video_accuracies else 0
    
    overall_metrics = {
        'total_questions': total_unique_questions,
        'correct_answers': total_correct_answers,
        'accuracy': overall_accuracy,
        'unique_videos': len(video_accuracy),
        'avg_per_video_accuracy': overall_avg_per_video_accuracy
    }
    
    # Calculate accuracy for each type
    results = {}
    
    for question_type, items in type_data.items():
        total_questions = len(items)
        correct_answers = sum(1 for item in items if item['answer'] == item['response'])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Get unique video count for this type
        unique_videos = len(set(item['video_id'] for item in items))
        
        # Calculate per-video accuracy
        video_accuracy = {}
        for item in items:
            video_id = item['video_id']
            if video_id not in video_accuracy:
                video_accuracy[video_id] = {'correct': 0, 'total': 0}
            video_accuracy[video_id]['total'] += 1
            if item['answer'] == item['response']:
                video_accuracy[video_id]['correct'] += 1
        
        # Calculate average per-video accuracy
        per_video_accuracies = [acc['correct']/acc['total'] for acc in video_accuracy.values()]
        avg_per_video_accuracy = sum(per_video_accuracies) / len(per_video_accuracies) if per_video_accuracies else 0
        
        results[question_type] = {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'unique_videos': unique_videos,
            'avg_per_video_accuracy': avg_per_video_accuracy,
            'per_video_accuracy': video_accuracy
        }
    
    return results, overall_metrics


def print_detailed_results(results: Dict[str, Dict[str, Any]], overall_metrics: Dict[str, Any]) -> None:
    """Print detailed accuracy results."""
    print("\n" + "="*80)
    print("ACCURACY ANALYSIS BY QUESTION TYPE")
    print("="*80)
    
    # Print overall accuracy prominently at the top
    overall_accuracy_pct = overall_metrics['accuracy'] * 100
    overall_avg_per_video_pct = overall_metrics['avg_per_video_accuracy'] * 100
    print(f"\n🎯 OVERALL ACCURACY: {overall_accuracy_pct:.2f}% ({overall_metrics['correct_answers']}/{overall_metrics['total_questions']} questions)")
    print(f"🎯 OVERALL AVG/VIDEO: {overall_avg_per_video_pct:.2f}% (across {overall_metrics['unique_videos']} videos)")
    print("="*80)
    
    # Sort by accuracy (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'Question Type':<25} {'Questions':<10} {'Correct':<10} {'Accuracy':<10} {'Videos':<8} {'Avg/Video':<10}")
    print("-" * 80)
    
    for question_type, metrics in sorted_results:
        accuracy_pct = metrics['accuracy'] * 100
        avg_per_video_pct = metrics['avg_per_video_accuracy'] * 100
        
        print(f"{question_type:<25} {metrics['total_questions']:<10} {metrics['correct_answers']:<10} "
              f"{accuracy_pct:<9.2f}% {metrics['unique_videos']:<8} {avg_per_video_pct:<9.2f}%")
    
    print("-" * 80)
    print(f"{'OVERALL':<25} {overall_metrics['total_questions']:<10} {overall_metrics['correct_answers']:<10} {overall_accuracy_pct:<9.2f}% {overall_metrics['unique_videos']:<8} {overall_avg_per_video_pct:<9.2f}%")
    print("="*80)


def print_per_video_breakdown(results: Dict[str, Dict[str, Any]]) -> None:
    """Print per-video accuracy breakdown for each question type."""
    print("\n" + "="*80)
    print("PER-VIDEO ACCURACY BREAKDOWN")
    print("="*80)
    
    for question_type, metrics in sorted(results.items()):
        print(f"\n{question_type.upper()}:")
        print(f"Total videos: {metrics['unique_videos']}")
        print(f"Average per-video accuracy: {metrics['avg_per_video_accuracy']*100:.2f}%")
        
        # Show some examples of per-video performance
        video_accuracies = [(vid, acc['correct']/acc['total']) for vid, acc in metrics['per_video_accuracy'].items()]
        video_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 5 videos by accuracy:")
        for i, (video_id, acc) in enumerate(video_accuracies[:5]):
            print(f"  Video {video_id}: {acc*100:.2f}%")
        
        if len(video_accuracies) > 5:
            print("Bottom 5 videos by accuracy:")
            for i, (video_id, acc) in enumerate(video_accuracies[-5:]):
                print(f"  Video {video_id}: {acc*100:.2f}%")


def save_results_to_json(results: Dict[str, Dict[str, Any]], overall_metrics: Dict[str, Any], output_file: str) -> None:
    """Save detailed results to a JSON file."""
    # Convert to JSON-serializable format
    json_results = {
        'overall_accuracy': {
            'total_questions': overall_metrics['total_questions'],
            'correct_answers': overall_metrics['correct_answers'],
            'accuracy': overall_metrics['accuracy'],  # Store as decimal for consistency
            'accuracy_percentage': overall_metrics['accuracy'] * 100
        },
        'by_question_type': {}
    }
    
    for question_type, metrics in results.items():
        json_results['by_question_type'][question_type] = {
            'total_questions': metrics['total_questions'],
            'correct_answers': metrics['correct_answers'],
            'accuracy': metrics['accuracy'],
            'unique_videos': metrics['unique_videos'],
            'avg_per_video_accuracy': metrics['avg_per_video_accuracy'],
            'per_video_accuracy': {
                str(vid): {
                    'correct': acc['correct'],
                    'total': acc['total'],
                    'accuracy': acc['correct']/acc['total']
                }
                for vid, acc in metrics['per_video_accuracy'].items()
            }
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_accuracy_by_type.py <merged_json_file>")
        print("Example: python analyze_accuracy_by_type.py merged_query_SA_lvbench_qwenlm.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    print(f"Analyzing accuracy by question type from: {input_file}")
    print("-" * 50)
    
    # Load data
    data = load_merged_data(input_file)
    
    # Analyze accuracy by type
    results, overall_metrics = analyze_accuracy_by_type(data)
    
    # Print results
    print_detailed_results(results, overall_metrics)
    print_per_video_breakdown(results)
    
    # Save detailed results
    output_file = input_file.replace('.json', '_accuracy_analysis.json')
    save_results_to_json(results, overall_metrics, output_file)
    
    print(f"\nAnalysis completed!")


if __name__ == "__main__":
    main()
