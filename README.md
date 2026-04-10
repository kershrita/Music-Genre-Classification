# Music Genre Classification System

> Production-oriented machine learning pipeline for multi-class music genre prediction using robust preprocessing, engineered audio features, and reproducible evaluation.

## Overview

This project was developed by an Applied AI Engineer (Jun 2023 - Jul 2023) in association with SHAI For AI | شاي للذكاء الاصطناعي.

It delivers an end-to-end genre classification workflow that starts from dataset ingestion and ends with prediction artifacts ready for downstream use.

### Role and Ownership

- Built the full preprocessing and feature-engineering pipeline.
- Implemented and compared multi-class classification models.
- Evaluated and optimized model behavior for reliable genre prediction.
- Produced reusable outputs for downstream analytics and integration.

### Problem Statement

Automatically classify music tracks into genre classes using track-level audio descriptors and supervised learning.

Current implementation uses pre-extracted descriptors from tabular datasets; the architecture keeps feature extraction modular so raw-audio pipelines (for example, MFCC and spectral extraction) can be plugged in without redesigning downstream components.

### Why This System Matters

- Reduces manual catalog tagging effort for music platforms.
- Enables scalable metadata enrichment for recommendation and search systems.
- Provides a reproducible ML baseline for future audio-intelligence products.

### Real-World Use Cases

- Streaming catalog auto-tagging and quality control.
- Cold-start support in music recommendation systems.
- Playlist generation and content organization by genre.
- Analytical segmentation of large music libraries.

## Architecture

The system follows a modular ML pipeline:

1. Ingestion Layer:
	- Load training and inference datasets.
	- Validate schema and required fields.

2. Data Quality and Preparation Layer:
	- Normalize mixed duration units (ms/min).
	- Impute missing values using feature-specific strategies (mean, median, mode).
	- Remove high-cardinality text identifiers that do not generalize (artist and track names).

3. Feature Engineering Layer:
	- Add interaction feature (`tempo_loudness`).
	- Apply nonlinear transformations (log, sqrt, cbrt, reciprocal) to stabilize distributions.
	- Standardize numerical float features.
	- Remove weak or redundant predictors based on analysis.

4. Modeling Layer:
	- Build stratified train/test split for balanced evaluation behavior.
	- Benchmark models using LazyPredict.
	- Tune SVC with GridSearchCV.
	- Train multi-class XGBoost model.

5. Evaluation and Delivery Layer:
	- Score candidate models with classification accuracy.
	- Generate final predictions for inference dataset.
	- Export results to `submission.csv`.

### Architecture Diagram

![Music Genre Classification System Workflow](assets/Music%20Genre%20Classification%20System%20Workflow.png)

The workflow visualizes the production path from ingestion to prediction artifact generation, with an iterative loop from evaluation back into feature engineering and model refinement.

## Features

- End-to-end notebook workflow from ingestion to final prediction export.
- Feature-specific null handling and normalization strategies.
- Engineered interaction features and distribution-aware transformations.
- Stratified splitting and comparative model benchmarking.
- Hyperparameter tuning for SVC and multi-class XGBoost training.
- Plot export support for EDA and feature diagnostics.

## Technical Highlights

- Designed a reusable `wrangle()` stage to keep preprocessing consistent across train and test paths.
- Applied transformation-by-distribution logic instead of one-size-fits-all preprocessing.
- Combined rapid model screening (LazyPredict) with targeted optimization (GridSearchCV on SVC).
- Used an explicit multi-class XGBoost setup (`multi:softmax`, `num_class=12`) for scalable classification.
- Produced reproducible output artifacts (`submission.csv`) suitable for evaluation workflows.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (`SimpleImputer`, `StandardScaler`, `train_test_split`, `GridSearchCV`, `SVC`, `accuracy_score`)
- XGBoost
- LazyPredict
- Matplotlib, Seaborn
- Jupyter Notebook

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/Music-Genre-Classification.git
cd Music-Genre-Classification/Model
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lazypredict jupyter
```

### Run

```bash
jupyter notebook "Music Genre Classification.ipynb"
```

Run all cells in order to:

- Execute preprocessing and feature engineering.
- Train/evaluate candidate models.
- Generate inference predictions and export `submission.csv`.

## Results

Current implementation outputs:

- Comparative model benchmark from LazyPredict.
- Tuned SVC evaluation via cross-validation and hold-out accuracy.
- Multi-class XGBoost inference pipeline.
- Final prediction artifact: `Model/submission.csv`.
- Architecture diagram: `assets/Music Genre Classification System Workflow.png`.
- Saved EDA/analysis visuals in `assets/images/`.

## Model Details

- Task Type: Multi-class classification.
- Target Variable: `Class`.
- Candidate Models: SVC, XGBoost, plus additional baseline classifiers from LazyPredict.
- Evaluation Method: Accuracy on stratified hold-out split.
- Training Design: Preprocessing + feature engineering + model selection + inference export.
