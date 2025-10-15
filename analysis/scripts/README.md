# AVA Analysis Scripts

This directory contains the essential analysis scripts for the AVA (Agentic Video Assistant) project, organized in execution order.

> **📖 For complete documentation, see the main [README.md](../README.md)**

## 📁 Script Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_merge_query_results.py` | Merge multiple JSON files | Multiple JSON files | `data/01_merged_query_results.json` |
| `02_analyze_accuracy_by_question_type.py` | Calculate accuracy by question type | `data/01_merged_query_results.json` | `results/01_accuracy_by_question_type.json` |
| `03_generate_all_visualizations.py` | Generate all visualizations | `data/` directory | 3 PNG visualization files |

## 🚀 Quick Start

```bash
# From the analysis directory
cd analysis

# 1. Merge results (if needed)
python scripts/01_merge_query_results.py data/01_merged_query_results.json file1.json file2.json

# 2. Analyze accuracy by question type
python scripts/02_analyze_accuracy_by_question_type.py data/01_merged_query_results.json

# 3. Generate all visualizations
python scripts/03_generate_all_visualizations.py data
```

## 📝 Notes

- All scripts are designed to be run independently
- Input/output file paths are hardcoded for consistency
- Results are saved in JSON format for easy analysis
- Scripts include detailed progress reporting and error handling