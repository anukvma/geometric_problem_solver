# Geometric Problem Solver

A scalable AI-powered system for solving geometric problems using vision-language models (Google Gemini), knowledge graphs (Neo4j), and multi-hop reasoning.

## 🎯 Overview

This system converts geometric diagram images into structured knowledge graphs and uses LLM-driven reasoning to solve geometry problems with multiple choice questions. It supports both numeric and parametric/algebraic problems.

### Key Features

- **Vision-to-Graph**: Extracts geometric structures from images using Gemini Vision API
- **Knowledge Base**: Stores geometric theorems and shapes in Neo4j graph database
- **Multi-hop Reasoning**: Chains multiple reasoning steps to solve complex problems
- **Multimodal Support**: Can reason with text-only knowledge graphs OR include images
- **Parametric Problems**: Handles algebraic expressions and symbolic mathematics
- **Graph Caching**: Avoids re-extracting graphs with namespace isolation
- **Evaluation Framework**: Comprehensive evaluation with comparison capabilities
- **Robust Error Handling**: Retry logic with progressive token limits

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Components](#-components)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Evaluation](#-evaluation)
- [Example Results](#-example-results)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)

---

## 🏗️ System Architecture

```
┌─────────────────────┐
│   Input Image       │
│  (Geometric Diagram)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Gemini Vision API                  │
│  - Extracts points, edges, shapes   │
│  - Identifies relationships         │
│  - Handles parametric values        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Neo4j Knowledge Graph              │
│  - Points with angles               │
│  - Edges as relationships           │
│  - Shapes with properties           │
│  - Theorems attached                │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  LLM-Driven Reasoning Engine        │
│  - Multi-hop reasoning              │
│  - Theorem application              │
│  - Algebraic manipulation           │
│  - Choice selection                 │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Solution Output    │
│  - Selected answer  │
│  - Reasoning steps  │
│  - Confidence score │
└─────────────────────┘
```

---

## 📦 Components

### Core Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| **`knowledge_base_builder.py`** | Builds geometric knowledge base in Neo4j | - Initialize theorems<br>- Store shapes<br>- Manage connections |
| **`image_to_graph_builder.py`** | Extracts graph structure from images | - Vision API extraction<br>- Graph creation<br>- Caching & isolation<br>- Truncation handling |
| **`geometric_problem_solver.py`** | Main reasoning engine | - Multi-hop reasoning<br>- Theorem application<br>- Solution generation<br>- Retry with progressive tokens |
| **`evaluate_solver.py`** | Evaluation framework | - Dataset loading<br>- Batch evaluation<br>- Comparison mode<br>- Results reporting |
| **`requirements.txt`** | Python dependencies |


## 🚀 Installation

### Prerequisites

- Python 3.8+
- Neo4j Database (running locally or remote)
- Google Gemini API key

### Step 1: Install Neo4j

**Option A: Docker (Recommended)**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**Option B: Local Installation**
- Download from https://neo4j.com/download/
- Set password

### Step 2: Install Python Dependencies

```bash
# Clone the repository
cd geomteric-problem-solver

# Install dependencies
pip install -r requirements.txt
```


## ⚙️ Configuration

Update API keys and Neo4j credentials in evaluation scripts:

```python
# In evaluate_solver_gemini.py, geometric_problem_solver1_gemini.py, etc.
config = {
    'neo4j_uri': 'bolt://localhost:7687',
    'neo4j_user': 'neo4j',
    'neo4j_password': 'password',        # UPDATE THIS
    'gemini_api_key': 'YOUR_API_KEY'     # UPDATE THIS
}
```

**Dataset Path:**
```python
DATASET_PATH = "images/geo3k/train"  # Update to your dataset location
```

---

## 🎬 Quick Start

### 1. Initialize Knowledge Base

```bash
python knowledge_base_builder.py
```

This creates the foundational geometric theorems in Neo4j.


### 2. Run Evaluation

```bash
python evaluate_solver.py
```

**Interactive menu:**
```
1. Knowledge Graph Only (default)
2. Knowledge Graph + Image
3. Comparison (both modes)
```

**Current settings:**
- Random problem selection
- 50 problems (configurable)
- Random seed: 42 (reproducible)

---


## 📊 Evaluation

### Evaluation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Knowledge Graph Only** | Uses only extracted graph structure | Baseline performance |
| **Knowledge Graph + Image** | Includes diagram image in prompt | Multimodal reasoning |
| **Comparison** | Runs both modes on same problems | Research & analysis |


### Metrics Reported

- **Accuracy**: Percentage of correct answers
- **Success Rate**: Percentage solved without errors
- **Average Confidence**: Mean confidence across problems
- **Average Solve Time**: Mean time per problem
- **Average Steps**: Mean reasoning steps per solution

### Output Files

Evaluation creates three files in `evaluation_results/`:

1. **`evaluation_results_TIMESTAMP.json`**
   - Complete detailed results
   - All problem solutions
   - Reasoning chains

2. **`evaluation_summary_TIMESTAMP.csv`**
   - Problem-by-problem summary
   - Quick spreadsheet view

3. **`evaluation_report_TIMESTAMP.txt`**
   - Human-readable report
   - Overall metrics
   - Problem breakdowns

**Comparison mode adds:**
- `comparison_results_TIMESTAMP.json`
- `comparison_report_TIMESTAMP.txt`


## 🐛 Troubleshooting

### Common Issues

#### 1. Neo4j Connection Error

**Error:** `Failed to establish connection`

**Solution:**
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart Neo4j
docker restart neo4j

# Verify credentials in config
```

#### 2. Gemini API Rate Limit

**Error:** `429 Resource Exhausted`

**Solution:**
- Reduce `max_problems` in evaluation
- Add delays between API calls
- Check API quota at https://console.cloud.google.com/

#### 3. Graph Extraction Failure

**Error:** `Failed to extract graph structure`

**Solution:**
- Check image file exists and is readable
- Increase token limits in `image_to_graph_builder_openai_gemini.py`
- Use fallback extraction
- Check logs for specific errors

#### 4. JSON Parsing Error

**Error:** `JSON decode error`

**Solution:**
- System now has automatic retry with higher tokens
- Check logs for which extraction method worked
- Verify Gemini API response format
- Review prompt for JSON format instructions

#### 5. TypeError in Evaluation

**Error:** `'>=' not supported between instances of 'TypeError' and 'int'`

**Solution:**
- Fixed in latest version with robust type checking
- Automatic retry handles malformed responses
- Safe dictionary access prevents crashes

### Performance Tips

**Faster Evaluation:**
1. Use cached graphs (automatic)
2. Reduce `max_problems` for testing
3. Use `include_image=False` (faster than multimodal)
4. Clean up old graphs: `python manage_graphs.py delete-old --days 7`

**Better Accuracy:**
1. Use `include_image=True` (multimodal reasoning)
2. Increase token limits to 65000
3. Verify graph extraction quality
4. Review and improve theorems in knowledge base


## 📝 Dataset Format

Expected dataset structure:

```
images/geo3k/train/
├── 1/
│   ├── img_diagram.png    # Geometric diagram image
│   └── data.json          # Problem data
├── 2/
│   ├── img_diagram.png
│   └── data.json
└── ...
```

**data.json format:**
```json
{
  "compact_text": "Use parallelogram MNPR to find y.",
  "choices": ["10", "15", "20", "25"],
  "answer": "B"
  ...
}
```


