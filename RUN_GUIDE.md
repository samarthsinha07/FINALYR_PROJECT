# 🚀 Crisis-Fusion: Setup & Execution Guide

> **🌐 Live Deployed Application:** [https://disastermanager.streamlit.app/](https://disastermanager.streamlit.app/)

This document provides step-by-step instructions to run the **Disaster Management and Recommendation System** dashboard, prepare data, train model checkpoints, and evaluate multimodal deep learning pipelines on the CrisisMMD benchmark.

---

## 📋 Directory Structure

```text
CRISIS DASHBOARD/
├── data/
│   ├── CrisisMMD_v2.0/                   <-- Linked dataset (images + tsv splits)
│   └── processed/                        <-- Preprocessed task CSVs & label maps
│       ├── task1a/                       <-- Informativeness (informative vs not_informative)
│       ├── task1b/                       <-- Disaster Type (earthquake, flood, hurricane, wildfire, etc.)
│       ├── task2/                        <-- Humanitarian Categories (8 classes)
│       ├── task3/                        <-- Damage Severity (little/no, mild, severe)
│       └── stats.csv                     <-- Partition sample distributions
├── figures/                              <-- Official confusion matrices & evaluation metrics
│   ├── report/                           <-- Summary comparison charts & tables
│   ├── task1b_score_fusion_metrics.json
│   ├── task2_score_fusion_metrics.json
│   └── task3_score_fusion_metrics.json
├── app.py                                <-- Complete Streamlit Web Dashboard (6 modules)
├── model_engine.py                       <-- Cached PyTorch & HuggingFace inference engine
├── baselines.py                          <-- Unimodal training & evaluation script
├── score_fusion.py                       <-- Multimodal score fusion pipeline
├── evaluate.py                           <-- Multimodal cross-modal ablation evaluation
├── evaluate_score_fusion.py              <-- Score fusion test set evaluation script
├── fusion_model.py                       <-- Cross-modal attention neural architecture
├── improve_task3.py                      <-- Class weighting & augmentation experiments
├── prepare_data.py                       <-- Data preparation and agreement filtering
├── demo_task2.py                         <-- Gradio demo for Task 2
├── requirements.txt                      <-- Python package dependencies
├── results.csv                           <-- Experimental accuracy & F1 results log
├── ablation_results.csv                  <-- Task 1a cross-modal ablation log
├── task3_improvement_results.csv         <-- Task 3 class-imbalance mitigation log
└── RUN_GUIDE.md                          <-- This guide
```

---

## ⚙️ Environment Setup

### 1. Python Virtual Environment
Python 3.10 is the recommended environment for compatibility with PyTorch and Transformers.

```bash
# Using uv (fastest)
uv venv .venv --python 3.10
.venv\Scripts\activate

# Or using standard python
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🗄️ Dataset Preparation

The dataset is partitioned into 4 tasks with agreement filtering (ensuring text and image annotations align):

```bash
python prepare_data.py
```

This populates `data/processed/` with `train.csv`, `dev.csv`, `test.csv`, and `label_map.json` for each task.

---

## 🖥️ Launching the Web Dashboard

To launch the complete interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The application will start on `http://localhost:8501`.

### Dashboard Modules:
1. **📊 Overview & Telemetry:** Dataset metrics, benchmark accuracies, and checkpoint registry.
2. **🔮 Live Multimodal Prediction:** Real-time inference with text + image inputs, confidence scores, and probability graphs.
3. **🖼️ Vision & Image Modality Analysis:** Unimodal image feature analysis and modality comparisons.
4. **📈 Model Evaluation & Benchmarks:** Confusion matrices, ablation metrics, and CSV downloads.
5. **⚠️ Damage Severity Analysis (Task 3):** Class imbalance deep dive and mitigation benchmark results.
6. **📑 Academic Research & Paper Comparison:** Shetty et al. (Springer 2025) benchmark vs. experimental project results.

---

## 🏋️ Training & Evaluating Checkpoints

### 1. Train Unimodal Baselines
```bash
# Text models (bert | roberta | bertweet)
python baselines.py --task task1a --model bert --epochs 3

# Vision models (resnet50 | densenet121 | vit)
python baselines.py --task task1a --model resnet50 --epochs 3
```

### 2. Run Score-Level Fusion Evaluation
```bash
python evaluate_score_fusion.py --task task1a --text-model roberta --weight 0.5
```

### 3. Train Cross-Modal Attention Fusion Model
```bash
python fusion_model.py --task task1a --epochs 5
```

### 4. Run Task 3 Imbalance Mitigation
```bash
python improve_task3.py --weights --augment
```

---

## 📦 Checkpoint Naming Convention
Place or save checkpoints in the workspace root:
- `best_{task}_{model}.pt` (e.g. `best_task1a_bert.pt`, `best_task1a_resnet50.pt`)
- `best_{task}_fusion.pt` for cross-modal models (e.g. `best_task1a_fusion.pt`)
