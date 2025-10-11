#!/usr/bin/env python3
"""
Script to create visualizations of the accuracy analysis results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def load_accuracy_results(file_path: str) -> dict:
    """Load the accuracy analysis results."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        sys.exit(1)


def create_accuracy_bar_chart(results: dict, output_file: str = "accuracy_by_type.png"):
    """Create a bar chart showing accuracy by question type."""
    # Handle both old and new JSON format
    if 'by_question_type' in results:
        question_data = results['by_question_type']
        overall_acc = results['overall_accuracy']['accuracy_percentage']
    else:
        question_data = results
        overall_acc = sum(results[qtype]['correct_answers'] for qtype in results) / sum(results[qtype]['total_questions'] for qtype in results) * 100
    
    question_types = list(question_data.keys())
    accuracies = [question_data[qtype]['accuracy'] * 100 for qtype in question_types]
    
    # Sort by accuracy
    sorted_data = sorted(zip(question_types, accuracies), key=lambda x: x[1], reverse=True)
    question_types, accuracies = zip(*sorted_data)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(question_types)), accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Question Type', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(f'Model Accuracy by Question Type\nOverall Accuracy: {overall_acc:.1f}%', fontsize=14, fontweight='bold')
    plt.xticks(range(len(question_types)), question_types, rotation=45, ha='right')
    plt.ylim(0, max(accuracies) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add overall accuracy line
    plt.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy: {overall_acc:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {output_file}")
    plt.close()


def create_question_count_chart(results: dict, output_file: str = "question_count_by_type.png"):
    """Create a chart showing question count by type."""
    # Handle both old and new JSON format
    if 'by_question_type' in results:
        question_data = results['by_question_type']
    else:
        question_data = results
    
    question_types = list(question_data.keys())
    counts = [question_data[qtype]['total_questions'] for qtype in question_types]
    
    # Sort by count
    sorted_data = sorted(zip(question_types, counts), key=lambda x: x[1], reverse=True)
    question_types, counts = zip(*sorted_data)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(question_types)), counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Question Type', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Questions', fontsize=12, fontweight='bold')
    plt.title('Question Count by Type', fontsize=14, fontweight='bold')
    plt.xticks(range(len(question_types)), question_types, rotation=45, ha='right')
    plt.ylim(0, max(counts) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Question count chart saved to: {output_file}")
    plt.close()


def create_accuracy_vs_count_scatter(results: dict, output_file: str = "accuracy_vs_count.png"):
    """Create a scatter plot of accuracy vs question count."""
    # Handle both old and new JSON format
    if 'by_question_type' in results:
        question_data = results['by_question_type']
    else:
        question_data = results
    
    question_types = list(question_data.keys())
    accuracies = [question_data[qtype]['accuracy'] * 100 for qtype in question_types]
    counts = [question_data[qtype]['total_questions'] for qtype in question_types]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(counts, accuracies, s=100, alpha=0.7, c=accuracies, cmap='viridis')
    
    # Add labels for each point
    for i, qtype in enumerate(question_types):
        plt.annotate(qtype, (counts[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Number of Questions', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy vs Question Count by Type', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_file}")
    plt.close()


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python visualize_accuracy.py <accuracy_analysis_json_file>")
        print("Example: python visualize_accuracy.py merged_query_SA_lvbench_qwenlm_accuracy_analysis.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    print(f"Creating visualizations from: {input_file}")
    
    # Load results
    results = load_accuracy_results(input_file)
    
    # Create visualizations
    create_accuracy_bar_chart(results)
    create_question_count_chart(results)
    create_accuracy_vs_count_scatter(results)
    
    print("\nAll visualizations created successfully!")


if __name__ == "__main__":
    main()
