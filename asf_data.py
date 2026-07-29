"""
Load Apache Software Foundation (ASF) Jira issues from MongoDB BSON export (Zenodo).

Dataset: https://zenodo.org/records/5665896
Expected file: data/issues.bson.gz (and optionally priorities.bson.gz if priority is referenced by id).
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from bson import decode_file_iter
from bson.objectid import ObjectId

from config import (
    ASF_ISSUES_BSON_GZ,
    ASF_PRIORITIES_BSON_GZ,
    ASF_SAMPLE_CSV,
    ASF_MAX_SCAN_ISSUES,
    SAMPLE_SIZE,
    RANDOM_STATE,
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Some exports nest text
        for key in ("value", "text", "body", "content"):
            if key in value:
                return _as_text(value[key])
        return str(value)
    return str(value)


def _priority_to_str(doc: dict, priority_map: dict | None) -> str | None:
    p = doc.get("priority")
    if p is None:
        return None
    if isinstance(p, dict):
        return p.get("name") or p.get("value") or _as_text(p)
    if isinstance(p, str) and p.strip():
        return p.strip()
    if isinstance(p, ObjectId) and priority_map is not None:
        entry = priority_map.get(str(p))
        if isinstance(entry, dict):
            return entry.get("name") or entry.get("value")
        if isinstance(entry, str):
            return entry
    return None


def load_priority_map(path: Path) -> dict:
    """Build ObjectId (str) -> document map from priorities.bson.gz."""
    if not path.exists():
        return {}
    out: dict = {}
    with gzip.open(path, "rb") as f:
        for doc in decode_file_iter(f):
            oid = doc.get("_id")
            if oid is not None:
                out[str(oid)] = doc
    return out


def iter_issues_from_bson(bson_gz_path: Path) -> Iterator[dict]:
    with gzip.open(bson_gz_path, "rb") as f:
        for doc in decode_file_iter(f):
            yield doc


def issues_to_rows(
    docs: Iterator[dict],
    priority_map: dict | None,
    max_docs: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i, doc in enumerate(docs):
        if i >= max_docs:
            break
        summary = _as_text(doc.get("summary"))
        description = _as_text(doc.get("description"))
        priority = _priority_to_str(doc, priority_map)
        if not summary.strip() or not priority:
            continue
        rows.append(
            {
                "summary": summary.strip(),
                "description": description.strip(),
                "priority": str(priority).strip(),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["summary", "description", "priority"])
    return pd.DataFrame(rows)


def apply_priority_label_map(df: pd.DataFrame, priority_label_map: dict[str, str]) -> pd.DataFrame:
    """Map raw priority names to canonical labels; drop unmapped rows."""
    out = df.copy()
    out["priority"] = out["priority"].map(priority_label_map)
    return out.dropna(subset=["priority"]).reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Stratified sample of n rows by priority; if too few rows, return df."""
    df = df.drop_duplicates(subset=["summary", "description", "priority"]).reset_index(drop=True)
    if len(df) <= n:
        return df
    y = df["priority"].astype(str)
    if y.nunique() < 2:
        return df.sample(n=n, random_state=random_state).reset_index(drop=True)
    # StratifiedShuffleSplit: exact n train indices
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=random_state)
    idx = np.arange(len(df))
    train_idx, _ = next(sss.split(idx, y))
    return df.iloc[train_idx].reset_index(drop=True)


def build_asf_sample_csv(
    issues_path: Path | None = None,
    priorities_path: Path | None = None,
    out_csv: Path | None = None,
    sample_size: int = SAMPLE_SIZE,
    max_scan: int = ASF_MAX_SCAN_ISSUES,
    random_state: int = RANDOM_STATE,
    priority_label_map: dict[str, str] | None = None,
) -> Path:
    """
    Stream issues from BSON, keep up to max_scan valid rows, stratified sample sample_size, save CSV.
    """
    issues_path = issues_path or ASF_ISSUES_BSON_GZ
    priorities_path = priorities_path or ASF_PRIORITIES_BSON_GZ
    out_csv = out_csv or ASF_SAMPLE_CSV

    if not issues_path.exists():
        raise FileNotFoundError(
            f"Missing ASF dump: {issues_path}\n"
            "Download `issues.bson.gz` from Zenodo record 5665896 (Apache Jira Issue Tracking Dataset) "
            "and place it in the data/ folder."
        )

    priority_map = load_priority_map(priorities_path) if priorities_path.exists() else None
    if priority_map:
        print(f"Loaded {len(priority_map)} priority documents for id resolution.")
    else:
        print("No priorities.bson.gz found; using embedded/string priority fields only.")

    print(f"Scanning up to {max_scan:,} issues from {issues_path} ...")
    df = issues_to_rows(iter_issues_from_bson(issues_path), priority_map, max_scan)
    print(f"Valid issues with summary + priority: {len(df):,}")

    if priority_label_map:
        before = len(df)
        df = apply_priority_label_map(df, priority_label_map)
        print(f"After priority label mapping: {len(df):,} (dropped {before - len(df):,} unmapped)")

    sampled = stratified_sample(df, sample_size, random_state)
    sampled.to_csv(out_csv, index=False)
    print(f"Wrote stratified sample n={len(sampled)} -> {out_csv}")
    print(sampled["priority"].value_counts())
    return out_csv


def load_prepared_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or ASF_SAMPLE_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python prepare_asf_data.py\n"
            "(Requires issues.bson.gz from Zenodo 5665896.)"
        )
    return pd.read_csv(path)
