from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_object(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(obj, path)
    except ImportError:
        fallback_path = path.with_suffix(".pkl")
        with fallback_path.open("wb") as file:
            pickle.dump(obj, file)

def load_object(path: Path) -> Any:
    """
    Load a saved object from a joblib or pickle file.
    """

    try:
        import joblib

        return joblib.load(path)

    except ImportError:
        with path.open("rb") as file:
            return pickle.load(file)
