# Comparative Analysis of Classical Machine Learning Models for Disease Risk Prediction

## Overview
This repository contains an independent research study comparing classical machine learning models for non-clinical disease risk prediction using structured medical data.

The goal of this work is to evaluate whether simpler, interpretable models can match or outperform more complex algorithms when dataset size is limited — a common scenario in healthcare applications.

## Models Evaluated
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest

## Key Findings
- Logistic Regression achieved the highest accuracy (81.2%)
- Random Forest performed competitively but did not outperform the linear model
- KNN showed lower robustness due to sensitivity to feature scaling and noise
- Results highlight that increased model complexity does not always lead to better performance

## Dataset
- Public diabetes dataset (768 samples, 8 features)
- Includes demographic and physiological indicators
- Missing and invalid values handled via median imputation
- Features standardized prior to training

## Experimental Setup
- Train-test split: 80/20
- Evaluation metric: Accuracy
- Implemented using Python and scikit-learn

## Repository Contents
- `research report.md` – Full research paper with methodology and analysis
- `model_training.py` – Reproducible experiment code
- `results/` – Experimental outputs and visualizations

## How to Run
```bash
pip install -r requirements.txt
python model_training.py
