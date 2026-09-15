# Aspect-Based Sentiment Analysis

## Overview

This repository implements an aspect-based sentiment analysis pipeline for the
SemEval-2014 Laptop Reviews dataset.

The system extracts aspect terms from review sentences and predicts sentiment
polarity using transformer-based models with controlled evaluation.

## Repository Structure

```
src/
  train_absa.py          # Reproducible training and evaluation pipeline

datasets/
  SemEval XML datasets

absa_reports/
  Evaluation outputs, figures, and analysis artifacts
```

## Methodology

The pipeline includes:

- SemEval XML parsing
- aspect-aware preprocessing
- transformer-based sentiment classification
- training/validation monitoring
- error analysis
- robustness evaluation

## Evidence

Generated artifacts include:

- confusion matrices
- training curves
- class distribution analysis
- error analysis outputs
- seed comparison results

## Running

Install dependencies and execute:

```bash
python src/train_absa.py
```

## Notes

The original exploratory notebook was converted into a Python implementation
to improve reproducibility and maintainability.
