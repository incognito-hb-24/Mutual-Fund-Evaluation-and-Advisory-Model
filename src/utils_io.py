from __future__ import annotations
import os, json, random
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ProjectPaths:
    base: str
    raw: str | None = None
    processed: str | None = None
    reports: str | None = None
    models: str | None = None
    notebooks: str | None = None

    @classmethod
    def from_base(cls, base: str) -> "ProjectPaths":
        raw = f"{base}/data/raw"
        processed = f"{base}/data/processed"
        reports = f"{base}/reports"
        models = f"{base}/models"
        notebooks = f"{base}/notebooks"
        return cls(base, raw, processed, reports, models, notebooks)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        if p: os.makedirs(p, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
