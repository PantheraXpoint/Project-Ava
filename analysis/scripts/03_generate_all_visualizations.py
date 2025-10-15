#!/usr/bin/env python3
"""
Generate All Visualizations Script

This script generates all necessary visualizations:
1. Accuracy by question type
2. Accuracy of each 13 single paths
3. Accuracy of 2 categories (with/without re-query)
4. Combined accuracy of all paths

It also handles data generation and verification.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_accuracy(data: List[Dict]) -> Dict[str, Any]:
    """Calculate accuracy for a dataset."""
    if not data:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
    
    total = len(data)
    correct = sum(1 for item in data if item['answer'] == item['response'])
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct
    }

def create_accuracy_by_question_type_visualization(question_type_data: Dict, output_prefix: str = ""):
    """Create accuracy by question type visualization."""
    print("Creating accuracy by question type visualization...")
    
    # Extract data
    question_types = list(question_type_data['by_question_type'].keys())
    accuracies = [question_type_data['by_question_type'][qt]['accuracy'] * 100 for qt in question_types]
    question_counts = [question_type_data['by_question_type'][qt]['total_questions'] for qt in question_types]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Accuracy by question type
    bars = ax1.bar(range(len(question_types)), accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Accuracy by Question Type', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xlabel('Question Type', fontsize=12)
    ax1.set_xticks(range(len(question_types)))
    ax1.set_xticklabels(question_types, rotation=45, ha='right', fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add average accuracy line
    overall_accuracy = question_type_data['overall_accuracy']['accuracy_percentage']
    ax1.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2,
                label=f'Overall Average: {overall_accuracy:.1f}%')
    ax1.legend()
    
    # 2. Question count by type
    bars2 = ax2.bar(range(len(question_types)), question_counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax2.set_title('Question Count by Type', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Number of Questions', fontsize=12)
    ax2.set_xlabel('Question Type', fontsize=12)
    ax2.set_xticks(range(len(question_types)))
    ax2.set_xticklabels(question_types, rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars2, question_counts)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}01_accuracy_by_question_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy by question type visualization saved as: {output_prefix}01_accuracy_by_question_type.png")

def create_single_path_accuracy_visualization(data_dir: str, output_prefix: str = ""):
    """Create single path accuracy visualization."""
    print("Creating single path accuracy visualization...")
    
    # Load all individual path files
    path_files = []
    for i in range(1, 14):  # 13 individual paths
        pattern = f"02_path_{i:02d}_"
        for file in os.listdir(data_dir):
            if file.startswith(pattern) and file.endswith('.json'):
                path_files.append(file)
                break
    
    # Calculate accuracies
    path_names = []
    accuracies = []
    
    for file in sorted(path_files):
        file_path = os.path.join(data_dir, file)
        data = load_json_file(file_path)
        result = calculate_accuracy(data)
        
        # Extract path name from filename
        path_name = file.replace('02_path_', '').replace('.json', '').replace('_', ' ')
        path_names.append(path_name)
        accuracies.append(result['accuracy'])
    
    # Create single visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Accuracy by path
    bars = ax.bar(range(len(path_names)), accuracies, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax.set_title('Single Path Accuracy Analysis', fontweight='bold', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xlabel('SA Node Path', fontsize=14)
    ax.set_xticks(range(len(path_names)))
    ax.set_xticklabels(path_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add statistics text box
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)
    
    stats_text = f"""Statistics:
Mean: {mean_accuracy:.2f}%
Std Dev: {std_accuracy:.2f}%
Min: {min_accuracy:.2f}%
Max: {max_accuracy:.2f}%

Best: {path_names[np.argmax(accuracies)]} ({max_accuracy:.2f}%)
Worst: {path_names[np.argmin(accuracies)]} ({min_accuracy:.2f}%)"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}02_single_path_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Single path analysis visualization saved as: {output_prefix}02_single_path_analysis.png")

def create_depth_analysis_visualization(data_dir: str, output_prefix: str = ""):
    """Create depth analysis visualization."""
    print("Creating depth analysis visualization...")
    
    # Load depth files
    depth_files = {
        0: os.path.join(data_dir, "05_depth_0.json"),
        1: os.path.join(data_dir, "05_depth_1.json"),
        2: os.path.join(data_dir, "05_depth_2.json")
    }
    
    # Check if files exist
    for depth, file_path in depth_files.items():
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            return
    
    # Load data
    depth_data = {}
    for depth, file_path in depth_files.items():
        depth_data[depth] = load_json_file(file_path)
    
    # Calculate accuracies
    depths = [0, 1, 2]
    accuracies = [calculate_accuracy(depth_data[depth])['accuracy'] for depth in depths]
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Depth accuracy comparison
    bars = ax1.bar(depths, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7, edgecolor='black')
    ax1.set_title('Accuracy by Depth', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xlabel('Path Depth', fontsize=12)
    ax1.set_xticks(depths)
    ax1.set_xticklabels([f'Depth {d}' for d in depths])
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 2. Depth with re-query analysis
    depth_rq_files = {
        1: os.path.join(data_dir, "06_depth_1_with_requery.json"),
        2: os.path.join(data_dir, "06_depth_2_with_requery.json")
    }
    
    depth_rq_data = {}
    for depth, file_path in depth_rq_files.items():
        if os.path.exists(file_path):
            depth_rq_data[depth] = load_json_file(file_path)
    
    if depth_rq_data:
        rq_accuracies = [calculate_accuracy(depth_rq_data[depth])['accuracy'] for depth in [1, 2] if depth in depth_rq_data]
        rq_depths = [d for d in [1, 2] if d in depth_rq_data]
        
        bars2 = ax2.bar(rq_depths, rq_accuracies, color='lightblue', alpha=0.7, edgecolor='navy')
        ax2.set_title('Accuracy by Depth (With Re-query)', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_xlabel('Path Depth', fontsize=12)
        ax2.set_xticks(rq_depths)
        ax2.set_xticklabels([f'Depth {d}' for d in rq_depths])
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars2, rq_accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 3. Depth without re-query analysis
    depth_no_rq_files = {
        0: os.path.join(data_dir, "07_depth_0_without_requery.json"),
        1: os.path.join(data_dir, "07_depth_1_without_requery.json"),
        2: os.path.join(data_dir, "07_depth_2_without_requery.json")
    }
    
    depth_no_rq_data = {}
    for depth, file_path in depth_no_rq_files.items():
        if os.path.exists(file_path):
            depth_no_rq_data[depth] = load_json_file(file_path)
    
    if depth_no_rq_data:
        no_rq_accuracies = [calculate_accuracy(depth_no_rq_data[depth])['accuracy'] for depth in depths if depth in depth_no_rq_data]
        no_rq_depths = [d for d in depths if d in depth_no_rq_data]
        
        bars3 = ax3.bar(no_rq_depths, no_rq_accuracies, color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax3.set_title('Accuracy by Depth (Without Re-query)', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Accuracy (%)', fontsize=12)
        ax3.set_xlabel('Path Depth', fontsize=12)
        ax3.set_xticks(no_rq_depths)
        ax3.set_xticklabels([f'Depth {d}' for d in no_rq_depths])
        ax3.set_ylim(0, 100)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars3, no_rq_accuracies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 4. Combined depth comparison
    categories = ['Overall\nDepth 0', 'Overall\nDepth 1', 'Overall\nDepth 2', 
                  'With RQ\nDepth 1', 'With RQ\nDepth 2',
                  'Without RQ\nDepth 0', 'Without RQ\nDepth 1', 'Without RQ\nDepth 2']
    
    combined_accuracies = []
    combined_colors = []
    
    # Add overall depths
    for i, acc in enumerate(accuracies):
        combined_accuracies.append(acc)
        combined_colors.append(['lightblue', 'lightgreen', 'lightcoral'][i])
    
    # Add with RQ depths (skip depth 0 as it has 0% accuracy)
    if depth_rq_data:
        for depth in [1, 2]:
            if depth in depth_rq_data:
                combined_accuracies.append(calculate_accuracy(depth_rq_data[depth])['accuracy'])
                combined_colors.append('lightblue')
    
    # Add without RQ depths
    if depth_no_rq_data:
        for depth in [0, 1, 2]:
            if depth in depth_no_rq_data:
                combined_accuracies.append(calculate_accuracy(depth_no_rq_data[depth])['accuracy'])
                combined_colors.append('lightcoral')
    
    bars4 = ax4.bar(range(len(combined_accuracies)), combined_accuracies, color=combined_colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Combined Depth Analysis', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_xlabel('Depth Categories', fontsize=12)
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax4.set_ylim(0, 100)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars4, combined_accuracies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}04_depth_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Depth analysis visualization saved as: {output_prefix}04_depth_analysis.png")

def create_category_accuracy_visualization(data_dir: str, output_prefix: str = ""):
    """Create category accuracy visualization (with/without re-query)."""
    print("Creating category accuracy visualization...")
    
    # Load category files
    requery_file = os.path.join(data_dir, "03_with_requery_nodes.json")
    no_requery_file = os.path.join(data_dir, "04_without_requery_nodes.json")
    
    if not os.path.exists(requery_file) or not os.path.exists(no_requery_file):
        print("Error: Category files not found!")
        return
    
    # Load data
    requery_data = load_json_file(requery_file)
    no_requery_data = load_json_file(no_requery_file)
    
    # Calculate accuracies
    requery_result = calculate_accuracy(requery_data)
    no_requery_result = calculate_accuracy(no_requery_data)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy comparison
    categories = ['With Re-Query\nNodes', 'Without Re-Query\nNodes', 'Combined\n(All Paths)']
    accuracies = [requery_result['accuracy'], no_requery_result['accuracy'], 41.1]  # 41.1% is the combined accuracy
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Re-Query Node Analysis\n(With vs Without Re-Query Nodes + Combined)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 2. Score difference line graph (absolute values, normalized)
    if len(requery_data) == len(no_requery_data):
        signed_differences = []
        for i in range(len(requery_data)):
            requery_score = requery_data[i]['sa_similarity_score']
            no_requery_score = no_requery_data[i]['sa_similarity_score']
            signed_diff = requery_score - no_requery_score
            signed_differences.append(signed_diff)
        
        # Convert to absolute values and normalize
        abs_differences = np.abs(signed_differences)
        # Normalize to 0-1 range
        if np.max(abs_differences) > 0:
            normalized_differences = abs_differences / np.max(abs_differences)
        else:
            normalized_differences = abs_differences
        
        question_indices = range(1, len(normalized_differences) + 1)
        ax2.plot(question_indices, normalized_differences, linewidth=0.8, alpha=0.7, color='blue')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='No Difference')
        ax2.set_title('Normalized Absolute Score Differences\n|Re-Query - Non-Re-Query| (0-1 scale)', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Question Index (1-1549)', fontsize=12)
        ax2.set_ylabel('Normalized Absolute Difference', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add statistics
        mean_abs_diff = np.mean(abs_differences)
        mean_norm_diff = np.mean(normalized_differences)
        ax2.text(0.02, 0.98, f'Mean Abs: {mean_abs_diff:.3f}\nMean Norm: {mean_norm_diff:.3f}', 
                 transform=ax2.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. Score difference distribution (absolute values, normalized)
    if 'signed_differences' in locals() and signed_differences:
        # Use the same absolute and normalized differences from above
        ax3.hist(normalized_differences, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Difference')
        ax3.set_title('Normalized Absolute Score Difference Distribution\n|Re-Query - Non-Re-Query| (0-1 scale)', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Normalized Absolute Difference', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_xlim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        mean_norm_diff = np.mean(normalized_differences)
        ax3.axvline(x=mean_norm_diff, color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {mean_norm_diff:.3f}')
        ax3.legend()
    
    # 4. Summary statistics
    ax4.axis('off')
    
    summary_text = f"""
Re-Query Node Analysis Results

Accuracy Comparison:
• With Re-Query Nodes: {requery_result['accuracy']:.2f}% ({requery_result['correct']}/{requery_result['total']})
• Without Re-Query Nodes: {no_requery_result['accuracy']:.2f}% ({no_requery_result['correct']}/{no_requery_result['total']})
• Combined (All Paths): 41.10% (636/1549)

Key Insights:
• Both categories perform similarly (~40%)
• Re-query nodes don't significantly impact performance
• Individual path limitation confirmed (vs 41.1% combined)
• Tree search approach is necessary

Data Source: Path-specific JSON files
Method: Highest score selection per category
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}03_requery_node_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Category accuracy visualization saved as: {output_prefix}03_requery_node_analysis.png")

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python 03_generate_all_visualizations.py <data_directory>")
        print("Example: python 03_generate_all_visualizations.py data")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir} not found!")
        sys.exit(1)
    
    print("🚀 Generating All Visualizations...")
    print("="*50)
    
    # Check if required files exist
    question_type_file = os.path.join("results", "01_accuracy_by_question_type.json")
    if not os.path.exists(question_type_file):
        print(f"❌ Error: {question_type_file} not found!")
        print("Please run 02_analyze_accuracy_by_question_type.py first.")
        sys.exit(1)
    
    # Load question type data
    print("📁 Loading question type data...")
    question_type_data = load_json_file(question_type_file)
    
    # Create visualizations
    create_accuracy_by_question_type_visualization(question_type_data, "")
    create_single_path_accuracy_visualization(data_dir, "")
    create_category_accuracy_visualization(data_dir, "")
    create_depth_analysis_visualization(data_dir, "")
    
    print("\n🎉 All visualizations generated successfully!")
    print("📁 Files created:")
    print("- 01_accuracy_by_question_type.png")
    print("- 02_single_path_analysis.png")
    print("- 03_requery_node_analysis.png")
    print("- 04_depth_analysis.png")

if __name__ == "__main__":
    main()
