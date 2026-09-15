# Transformer-Based Aspect-Based Sentiment Analysis

## Overview

This repository implements an end-to-end Aspect-Based Sentiment Analysis
(ABSA) pipeline for the SemEval-2014 Laptop Reviews benchmark.

The objective is to identify aspect terms in review sentences and
classify the sentiment polarity associated with each aspect.

The project explores transformer-based language models for fine-grained
sentiment understanding, including data preprocessing, model training,
evaluation, and error analysis.

------------------------------------------------------------------------

## Key Contributions

-   Implemented an aspect-aware sentiment classification pipeline for
    SemEval-2014 ABSA.
-   Developed preprocessing workflows for XML-based review annotations
    and aspect-polarity extraction.
-   Fine-tuned transformer-based models for aspect-conditioned sentiment
    prediction.
-   Performed systematic evaluation through classification metrics,
    confusion analysis, and robustness experiments.
-   Conducted error analysis to investigate model limitations and
    failure cases.

------------------------------------------------------------------------

## Methodology

The pipeline consists of:

1.  **Dataset Processing**
    -   Parsing SemEval-2014 Laptop Reviews annotations.
    -   Extracting aspect terms and associated sentiment labels.
    -   Preparing transformer-compatible input representations.
2.  **Model Training**
    -   Fine-tuning transformer-based language models for aspect
        sentiment classification.
    -   Evaluating model behavior across multiple training
        configurations.
3.  **Evaluation**
    -   Measuring classification performance using standard metrics.
    -   Analyzing prediction errors and sentiment confusion patterns.
    -   Comparing experimental variations.

------------------------------------------------------------------------

## Repository Structure

``` text
.
├── src/
│   └── train_absa.py
│       Main reproducible training and evaluation pipeline
│
├── notebooks/
│   └── SemEval2014_ABSA_E2E.ipynb
│       Original experimental notebook containing development workflow,
│       exploratory analysis, and experiment records
│
├── datasets/
│   SemEval-2014 Laptop Reviews dataset files
│
├── absa_reports/
│   ├── report_figs/
│   │   Generated evaluation and analysis visualizations
│   │
│   └── absa_out/
│       Model outputs, reports, and error analysis artifacts
│
└── README.md
```

------------------------------------------------------------------------

## Experimental Notebook

The repository includes the original Jupyter notebook used during model
development:

`notebooks/SemEval2014_ABSA_E2E.ipynb`

The notebook contains:

-   exploratory data analysis
-   preprocessing experiments
-   training experiments
-   evaluation workflows
-   visualization and analysis outputs

The cleaned implementation is provided separately in:

``` text
src/train_absa.py
```

to improve reproducibility and maintainability.

------------------------------------------------------------------------

## Evidence and Analysis

The repository preserves experiment artifacts including:

-   training and validation curves
-   confusion matrices
-   classification reports
-   error analysis outputs
-   robustness evaluation results

These artifacts document not only final performance, but also the
evaluation process used to understand model behavior.

------------------------------------------------------------------------

## Running the Project

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the training pipeline:

``` bash
python src/train_absa.py
```

------------------------------------------------------------------------

## Dataset

This project uses the SemEval-2014 Task 4 Laptop Reviews dataset for
aspect-based sentiment analysis.

Dataset annotations include:

-   review sentences
-   aspect terms
-   sentiment polarity labels

------------------------------------------------------------------------

## Future Improvements

Potential extensions include:

-   experimenting with larger instruction-tuned language models
-   incorporating contextual retrieval for aspect reasoning
-   evaluating cross-domain generalization
-   investigating causal and explainable sentiment representations

------------------------------------------------------------------------

## Acknowledgments

Developed as an experimental study of transformer-based language
understanding and fine-grained sentiment analysis.
