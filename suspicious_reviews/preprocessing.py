from __future__ import annotations

import re
import string
from functools import lru_cache

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:  # pragma: no cover - dependency declared in requirements.
    nltk = None
    stopwords = None
    wordnet = None
    WordNetLemmatizer = None
    word_tokenize = None


FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
}


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["clean_review_text"] = output["review_text"].fillna("").map(clean_text)
    output["token_count"] = output["clean_review_text"].str.split().map(len)
    output["character_count"] = output["review_text"].fillna("").str.len()
    return output


def clean_text(text: object) -> str:
    lowered = str(text or "").lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = tokenize(no_punctuation)
    stops = get_stopwords()
    lemmas = [lemmatize(token) for token in tokens if token not in stops and len(token) > 1]
    return " ".join(lemmas)


def tokenize(text: str) -> list[str]:
    if nltk is None or word_tokenize is None:
        return re.findall(r"[a-zA-Z]+", text)

    if not resource_available("tokenizers/punkt", "punkt"):
        return re.findall(r"[a-zA-Z]+", text)
    resource_available("tokenizers/punkt_tab", "punkt_tab")
    try:
        return [token for token in word_tokenize(text) if token.isalpha()]
    except LookupError:
        return re.findall(r"[a-zA-Z]+", text)


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    if nltk is None or stopwords is None:
        return FALLBACK_STOPWORDS
    if not resource_available("corpora/stopwords", "stopwords"):
        return FALLBACK_STOPWORDS
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOPWORDS


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer | None:
    if nltk is None or WordNetLemmatizer is None:
        return None
    if not resource_available("corpora/wordnet", "wordnet"):
        return None
    resource_available("corpora/omw-1.4", "omw-1.4")
    return WordNetLemmatizer()


def lemmatize(token: str) -> str:
    lemmatizer = get_lemmatizer()
    if lemmatizer is None or wordnet is None:
        return token
    try:
        return lemmatizer.lemmatize(token)
    except LookupError:
        return token


@lru_cache(maxsize=None)
def resource_available(resource: str, package: str) -> bool:
    if nltk is None:
        return False
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        return False
