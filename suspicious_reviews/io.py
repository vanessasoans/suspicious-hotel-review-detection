from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "hotel_name",
    "hotel_url",
    "review_score",
    "review_title",
    "positive_text",
    "negative_text",
    "review_date",
    "stay_date",
]


def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset was not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df[REQUIRED_COLUMNS].copy()


def normalize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.insert(0, "review_id", range(1, len(normalized) + 1))

    text_columns = ["hotel_name", "hotel_url", "review_title", "positive_text", "negative_text"]
    for col in text_columns:
        normalized[col] = normalized[col].fillna("").astype(str).str.strip()

    normalized["review_score_numeric"] = normalized["review_score"].apply(parse_review_score)
    normalized["review_date_parsed"] = normalized["review_date"].apply(parse_review_date)
    normalized["stay_date_parsed"] = pd.to_datetime(
        normalized["stay_date"], errors="coerce"
    ).dt.date.astype("string")

    normalized["review_text"] = (
        normalized["review_title"]
        + " "
        + normalized["positive_text"]
        + " "
        + normalized["negative_text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return normalized


def parse_review_score(value: object) -> float | None:
    if pd.isna(value):
        return None
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    return float(numbers[-1]) if numbers else None


def parse_review_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).replace("Reviewed:", "").strip()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

