# Suspicious Hotel Review Detection

Leakage-controlled semantic graph and explainable machine learning framework for suspicious hotel review detection under weak supervision.

## Overview

This repository contains the reproducible research code associated with the manuscript:

**Suspicious Hotel Review Detection Using Leakage-Controlled Semantic Graphs and Explainable Machine Learning**

The study combines semantic review representations, similarity graphs, graph-derived structural features, and XGBoost classification. The primary evaluation uses strict fold-specific graph construction to prevent structural information leakage.

## Main methodology

- Primary source dataset: 2,011 Booking.com hotel reviews after preprocessing.
- Text representation: `all-MiniLM-L6-v2` (384-dimensional embeddings).
- Similarity threshold for graph construction: cosine similarity >= 0.80.
- Weak labels are generated using the adopted heuristic rules and are not independently verified.
- Five-fold stratified cross-validation.
- Training graph is constructed using training reviews only.
- Validation reviews are projected onto the fixed training graph using validation-to-training relationships only.
- Validation-to-validation edges are not constructed.
- Primary graph features:
  - Betweenness Centrality
  - Clustering Coefficient
  - Triangle Count
  - Eigenvector Centrality
  - Component Size
- Review-level features:
  - Review Score
  - Token Count
  - Character Count
- Primary classifier: XGBoost.
- Eight imbalance conditions are evaluated: None, Class Weight, Random Oversampling, SMOTE, Borderline-SMOTE, SVM-SMOTE, KMeans-SMOTE, and ADASYN.
- SHAP is used for model explanation.
- The Ott et al. 1,600-review benchmark is evaluated separately using MiniLM representations.
- A separate external hotel-review collection is used only for cross-domain behaviour analysis because verified labels are unavailable.

## Repository structure

```text
.
├── data/
│   └── README.md
├── results/
│   ├── figures/
│   ├── eight_method_summary.csv
│   ├── fold_protocol_summary.csv
│   ├── final_model_summary.csv
│   └── experiment_metadata.json
├── suspicious_reviews/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── serialization.py
│   ├── strict_foldwise_8_imbalance_methods.py
│   ├── phase3_ott_benchmark.py
│   └── phase4_external_case_studies.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Data availability

Raw review datasets are intentionally **not included** in this public repository.

The Booking.com review data were collected from a source platform and are not redistributed because of applicable source-platform and data-use restrictions. The external hotel-review collection is likewise not redistributed here. The Ott et al. benchmark should be obtained from its original publication/research resources.

See the `data/README.md` file for the dataset description used in the study.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The final fold-wise experiment also requires `imbalanced-learn`, which is included in `requirements.txt`.

## Running the leakage-controlled experiment

Place the appropriately prepared 2,011-review source dataset at:

```text
data/processed/booking_reviews_traindataset_output.csv
```

Place the corresponding 384-dimensional MiniLM embeddings at:

```text
models/review_embeddings_minilm.npy
```

Then run:

```bash
python -m suspicious_reviews.strict_foldwise_8_imbalance_methods
```

The script constructs the training graph separately within each fold, projects validation reviews onto the fixed training graph, verifies that validation-validation edges are absent, evaluates all eight imbalance conditions, selects the best configuration using suspicious-class F1 followed by the predefined tie-breaking criteria, and retrains the selected configuration on all source reviews.

## Ott benchmark

The Ott et al. benchmark is not included in the repository. After obtaining the dataset and placing it at:

```text
data/ott/deceptive-opinion.csv
```

run:

```bash
python -m suspicious_reviews.phase3_ott_benchmark
```

This benchmark uses the verified deceptive/truthful labels provided by the dataset and does not generate weak labels.

## External cross-domain analysis

The external hotel-review collection is not included. After preparing the retained external dataset according to the manuscript's preprocessing protocol, the case-study script can be run with the appropriate input paths:

```bash
python -m suspicious_reviews.phase4_external_case_studies \
    --input data/processed/reviews_cleaned.csv \
    --source data/processed/booking_reviews_traindataset_output.csv \
    --embeddings models/review_embeddings_minilm.npy
```

The external procedure builds a graph from the source reviews and projects external reviews onto that fixed source graph. External-to-external graph edges are not constructed. Because verified external labels are unavailable, these outputs are model predictions/cross-domain behaviour analysis rather than supervised external validation.

## Reported primary result

The non-resampled XGBoost configuration is the primary selected model in the final experiment. The repository includes the aggregate imbalance comparison and fold-protocol summaries used for reporting.

## Reproducibility note

The repository intentionally excludes raw review text, private data, credentials, and large generated model artifacts. The included scripts and aggregate results are provided to document the experimental protocol and support reproducibility subject to access to the underlying datasets and required pretrained models.

## Citation

If you use this repository, please cite the associated manuscript:

> Vanessa Sharon Soans, Sucharitha Shetty, Prakash K. Aithal. *Suspicious Hotel Review Detection Using Leakage-Controlled Semantic Graphs and Explainable Machine Learning.*

