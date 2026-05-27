# Mental Health MLOps Project

A production-grade MLOps pipeline for depression detection from text.

## Problem Statement
Binary text classification to detect depression signals from user-written text.

## Dataset
- 7,731 samples | 2 classes (Depression / Non-Depression)
- Balanced dataset (~50/50 split)

## MLOps Stack
- Data Versioning : DVC
- Experiment Tracking : MLflow
- Serving : FastAPI
- Containerization : Docker
- CI/CD : GitHub Actions
- Monitoring : Evidently AI

## Project Structure
data/       → raw and processed datasets
src/        → all source code
pipelines/  → end-to-end pipeline scripts
configs/    → YAML configuration files
tests/      → unit and integration tests
models/     → saved model artifacts
reports/    → evaluation and monitoring reports
EOF