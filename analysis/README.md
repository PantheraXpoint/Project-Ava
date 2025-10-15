# AVA Analysis Suite

This directory contains all analysis scripts and results for the AVA (Agentic Video Assistant) project. The analysis is organized into three main folders with clear, descriptive naming conventions.

## 📁 Directory Structure

```
analysis/
├── scripts/          # Python analysis scripts (numbered by execution order)
├── data/            # Input data files
├── results/         # Generated analysis results
└── README.md        # This file
```

## 🔧 Scripts (`analysis/scripts/`)

### Core Analysis Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_merge_query_results.py` | Merge multiple JSON files | Multiple JSON files | `data/01_merged_query_results.json` |
| `02_analyze_accuracy_by_question_type.py` | Calculate accuracy by question type | `data/01_merged_query_results.json` | `results/01_accuracy_by_question_type.json` |
| `03_generate_all_visualizations.py` | Generate all visualizations | `data/` directory | 3 PNG visualization files |

## 📊 Data Files (`analysis/data/`)

### 01_merged_query_results.json
**Source**: Output from `01_merge_query_results.py`
**Content**: Consolidated query results from all video questions
**Fields**: video_id, question_id, question, answer, response, question_type, highest_sa_score_overall

### 02_path_01_* to 02_path_13_* (Individual Path Files)
**Source**: Generated from AVA cache analysis
**Content**: Results for each individual SA node path using highest score selection
**Fields**: video_id, question_id, question, answer, response, question_type, sa_similarity_score, path, highest_score_of_path, highest_score_overall
**Accuracy Range**: 37.4% - 40.6% (individual paths are limited)

### 03_with_requery_nodes.json
**Source**: Generated from AVA cache analysis
**Content**: Results using highest score from paths WITH re-query nodes (6 paths)
**Fields**: video_id, question_id, question, answer, response, question_type, sa_similarity_score, path, highest_score_of_category, highest_score_overall
**Accuracy**: 40.22% (623/1549)

### 04_without_requery_nodes.json
**Source**: Generated from AVA cache analysis
**Content**: Results using highest score from paths WITHOUT re-query nodes (7 paths)
**Fields**: video_id, question_id, question, answer, response, question_type, sa_similarity_score, path, highest_score_of_category, highest_score_overall
**Accuracy**: 40.09% (621/1549)

## 📈 Results Files (`analysis/results/`)

### 01_accuracy_by_question_type.json
**Source**: Output from `02_analyze_accuracy_by_question_type.py`
**Content**: Accuracy metrics grouped by question type
**Key Metrics**: 
- Overall accuracy: 41.06%
- Per-question-type accuracy
- Per-video accuracy breakdown

## 🚀 Quick Start

### Basic Analysis Workflow

```bash
# From the analysis directory
cd analysis

# 1. Merge results (if needed)
python scripts/01_merge_query_results.py data/01_merged_query_results.json file1.json file2.json file3.json file4.json

# 2. Basic accuracy analysis
python scripts/02_analyze_accuracy_by_question_type.py data/01_merged_query_results.json

# 3. Generate all visualizations
python scripts/03_generate_all_visualizations.py data
```

## 🎯 Key Visualizations

The `03_generate_all_visualizations.py` script creates 3 essential visualizations:

1. **`01_accuracy_by_question_type.png`** - Shows accuracy breakdown by question type with overall average line
2. **`02_single_path_analysis.png`** - Shows single path performance analysis with coverage metrics
3. **`03_requery_node_analysis.png`** - Shows re-query node analysis with score difference distributions

## 🔍 Key Findings

1. **Overall Accuracy**: 41.1% using highest-scoring responses
2. **SA Node Paths**: 13 unique reasoning patterns identified
3. **Re-query Analysis**: Non-re-query paths perform similarly to re-query paths (~40% accuracy)
4. **Single Path Limitation**: Individual paths achieve only 37-41% accuracy when used exclusively
5. **Tree Search Essential**: Multi-path approach is necessary for reasonable performance
6. **Scoring Problem**: Algorithm finds correct answers but doesn't select them optimally due to scoring issues

## 📁 File Naming Convention

- **Scripts**: `XX_descriptive_name.py` (numbered by execution order)
- **Data**: `XX_descriptive_name.json` (input data files)
- **Results**: `XX_descriptive_name.json` (analysis results)
- **Visualizations**: `XX_descriptive_name.png` (chart files)

All files are organized in logical folders with clear purposes.

## 📝 Notes

- All scripts are designed to be run independently
- Input/output file paths are hardcoded for consistency
- Results are saved in JSON format for easy analysis
- Scripts include detailed progress reporting and error handling
- The analysis focuses on real algorithm performance (41.1% accuracy) without exact match interventions