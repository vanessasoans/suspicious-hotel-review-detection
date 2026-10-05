"""
Phase 3: Ott ground-truth benchmark
-----------------------------------

Purpose
-------
Evaluate the true-label Ott deceptive hotel review corpus using the same
384-dimensional all-MiniLM-L6-v2 text representation used in the project.

Models
------
1. MiniLM embeddings + XGBoost
2. MiniLM embeddings + Linear SVM

Important
---------
- The Ott dataset provides verified deceptive/truthful labels.
- No weak labels are generated here.
- No graph features are used in this benchmark.
- The dataset is balanced (800 truthful, 800 deceptive), so SMOTE is not
  applied.
- Five-fold stratified cross-validation is used.
- Validation folds remain untouched during training.
- Continuous decision scores are used for ROC-AUC and PR-AUC.

Expected input columns
----------------------
deceptive, hotel, polarity, source, text

Expected project location
--------------------------
data/ott/deceptive-opinion.csv

Outputs
-------
results/phase3_ott_benchmark/
    fold_metrics.csv
    pooled_predictions.csv
    summary.csv

Run from the project root:
    python -m suspicious_reviews.phase3_ott_benchmark

Or explicitly provide the CSV:
    python -m suspicious_reviews.phase3_ott_benchmark ^
        --input "data\\ott\\deceptive-opinion.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is required. Install it with: pip install xgboost"
    ) from exc


RANDOM_STATE = 42
N_SPLITS = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Ott ground-truth benchmark."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/ott/deceptive-opinion.csv",
        help="Path to deceptive-opinion.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase3_ott_benchmark",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="SentenceTransformer batch size.",
    )
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = {"deceptive", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()

    # Keep only rows with usable review text and labels.
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["deceptive"] = df["deceptive"].astype(str).str.strip().str.lower()

    df = df[df["text"].str.len() > 0].copy()

    label_map = {
        "truthful": 0,
        "deceptive": 1,
    }
    df["label"] = df["deceptive"].map(label_map)

    if df["label"].isna().any():
        bad = sorted(df.loc[df["label"].isna(), "deceptive"].unique())
        raise ValueError(
            f"Unexpected values in 'deceptive' column: {bad}. "
            "Expected only 'truthful' and 'deceptive'."
        )

    df["label"] = df["label"].astype(int)

    counts = df["label"].value_counts().sort_index()
    print("\nDataset validation")
    print("------------------")
    print(f"Rows retained: {len(df)}")
    print(f"Truthful (0):  {counts.get(0, 0)}")
    print(f"Deceptive (1): {counts.get(1, 0)}")

    if set(counts.index) != {0, 1}:
        raise ValueError("Both truthful and deceptive classes are required.")

    return df.reset_index(drop=True)


def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def print_result(
    model_name: str,
    fold: int,
    metrics: dict,
) -> None:
    print(
        f"{model_name:<26} Fold {fold}: "
        f"Acc={metrics['accuracy']:.4f} | "
        f"BalAcc={metrics['balanced_accuracy']:.4f} | "
        f"P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )


def summarise(
    fold_metrics: pd.DataFrame,
    pooled_predictions: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "roc_auc",
        "pr_auc",
    ]

    for model_name in fold_metrics["model"].unique():
        model_folds = fold_metrics[fold_metrics["model"] == model_name]

        # Pooled out-of-fold metrics
        pooled = pooled_predictions[
            pooled_predictions["model"] == model_name
        ].copy()

        pooled_metrics = calculate_metrics(
            pooled["y_true"].to_numpy(),
            pooled["y_pred"].to_numpy(),
            pooled["y_score"].to_numpy(),
        )

        row = {
            "model": model_name,
            "n_folds": len(model_folds),
            "n_samples": len(pooled),
        }

        for metric in metric_names:
            row[f"pooled_{metric}"] = pooled_metrics[metric]
            row[f"mean_{metric}"] = model_folds[metric].mean()
            row[f"std_{metric}"] = model_folds[metric].std(ddof=1)

        row.update(
            {
                "pooled_tn": pooled_metrics["tn"],
                "pooled_fp": pooled_metrics["fp"],
                "pooled_fn": pooled_metrics["fn"],
                "pooled_tp": pooled_metrics["tp"],
            }
        )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}\n"
            "Place deceptive-opinion.csv under data/ott/ or pass --input."
        )

    print("=" * 80)
    print("PHASE 3 — OTT GROUND-TRUTH BENCHMARK")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Cross-validation: {N_SPLITS}-fold stratified CV")
    print("Text representation: normalized 384-d MiniLM embeddings")
    print("Graph features: none")
    print("SMOTE: not used (balanced ground-truth dataset)")
    print("=" * 80)

    df = pd.read_csv(input_path)
    df = validate_dataset(df)

    texts = df["text"].tolist()
    y = df["label"].to_numpy()

    print("\nGenerating MiniLM embeddings...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"Embedding shape: {embeddings.shape}")

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected 384-dimensional embeddings, got {embeddings.shape[1]}."
        )

    # Save embeddings so reruns do not require recomputing them manually.
    np.save(output_dir / "ott_minilm_embeddings.npy", embeddings)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    prediction_rows = []

    models = {
        "MiniLM + XGBoost": "xgb",
        "MiniLM + Linear SVM": "svm",
    }

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(embeddings, y), start=1
    ):
        X_train = embeddings[train_idx]
        X_val = embeddings[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        print(
            f"\nFold {fold}: "
            f"train={len(train_idx)} | validation={len(val_idx)}"
        )

        # ------------------------------------------------------------------
        # Model 1: MiniLM + XGBoost
        # ------------------------------------------------------------------
        xgb = build_xgb()
        xgb.fit(X_train, y_train)

        xgb_pred = xgb.predict(X_val).astype(int)
        xgb_score = xgb.predict_proba(X_val)[:, 1]

        xgb_metrics = calculate_metrics(
            y_val, xgb_pred, xgb_score
        )

        xgb_metrics.update(
            {
                "fold": fold,
                "model": "MiniLM + XGBoost",
                "train_size": len(train_idx),
                "validation_size": len(val_idx),
            }
        )
        fold_rows.append(xgb_metrics)
        print_result("MiniLM + XGBoost", fold, xgb_metrics)

        prediction_rows.extend(
            {
                "fold": fold,
                "row_index": int(idx),
                "model": "MiniLM + XGBoost",
                "y_true": int(y_val[j]),
                "y_pred": int(xgb_pred[j]),
                "y_score": float(xgb_score[j]),
            }
            for j, idx in enumerate(val_idx)
        )

        # ------------------------------------------------------------------
        # Model 2: MiniLM + Linear SVM
        # ------------------------------------------------------------------
        svm = LinearSVC(
            C=1.0,
            random_state=RANDOM_STATE,
            max_iter=10000,
        )
        svm.fit(X_train, y_train)

        svm_pred = svm.predict(X_val).astype(int)
        svm_score = svm.decision_function(X_val)

        svm_metrics = calculate_metrics(
            y_val, svm_pred, svm_score
        )

        svm_metrics.update(
            {
                "fold": fold,
                "model": "MiniLM + Linear SVM",
                "train_size": len(train_idx),
                "validation_size": len(val_idx),
            }
        )
        fold_rows.append(svm_metrics)
        print_result("MiniLM + Linear SVM", fold, svm_metrics)

        prediction_rows.extend(
            {
                "fold": fold,
                "row_index": int(idx),
                "model": "MiniLM + Linear SVM",
                "y_true": int(y_val[j]),
                "y_pred": int(svm_pred[j]),
                "y_score": float(svm_score[j]),
            }
            for j, idx in enumerate(val_idx)
        )

    fold_metrics = pd.DataFrame(fold_rows)
    pooled_predictions = pd.DataFrame(prediction_rows)

    summary = summarise(
        fold_metrics,
        pooled_predictions,
    )

    fold_metrics.to_csv(
        output_dir / "fold_metrics.csv",
        index=False,
    )
    pooled_predictions.to_csv(
        output_dir / "pooled_predictions.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("OTT BENCHMARK SUMMARY")
    print("=" * 80)

    for _, row in summary.iterrows():
        print(f"\n{row['model']}")
        print(f"  Pooled Accuracy:          {row['pooled_accuracy']:.6f}")
        print(
            f"  Pooled Balanced Accuracy: {row['pooled_balanced_accuracy']:.6f}"
        )
        print(f"  Pooled Precision:         {row['pooled_precision']:.6f}")
        print(f"  Pooled Recall:            {row['pooled_recall']:.6f}")
        print(f"  Pooled F1:                {row['pooled_f1']:.6f}")
        print(f"  Pooled Macro F1:          {row['pooled_macro_f1']:.6f}")
        print(f"  Pooled ROC-AUC:           {row['pooled_roc_auc']:.6f}")
        print(f"  Pooled PR-AUC:            {row['pooled_pr_auc']:.6f}")
        print(
            "  Five-fold F1:             "
            f"{row['mean_f1']:.6f} ± {row['std_f1']:.6f}"
        )
        print(
            "  Five-fold PR-AUC:         "
            f"{row['mean_pr_auc']:.6f} ± {row['std_pr_auc']:.6f}"
        )

    print("\nFiles saved:")
    print(f"  {output_dir / 'fold_metrics.csv'}")
    print(f"  {output_dir / 'pooled_predictions.csv'}")
    print(f"  {output_dir / 'summary.csv'}")
    print(f"  {output_dir / 'ott_minilm_embeddings.npy'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
