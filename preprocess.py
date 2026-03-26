"""Data loading and text preprocessing for Jira ticket prioritization."""
import re
import pandas as pd
import numpy as np
from pathlib import Path

from config import ASF_SAMPLE_CSV, RANDOM_STATE, SAMPLE_SIZE


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower case and standard names."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # Common aliases
    if "summary" not in df.columns and "title" in df.columns:
        df = df.rename(columns={"title": "summary"})
    if "description" not in df.columns and "body" in df.columns:
        df = df.rename(columns={"body": "description"})
    if "priority" not in df.columns and "issue_priority" in df.columns:
        df = df.rename(columns={"issue_priority": "priority"})
    return df


def clean_text(text: str) -> str:
    """Remove HTML, boilerplate, and excessive whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove hex-like tokens (e.g. memory addresses)
    text = re.sub(r"\b[0-9a-fA-F]{8,}\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jira_data(csv_path: Path | None = None) -> pd.DataFrame:
    """Load Jira data from CSV or generate synthetic sample (demo only)."""
    path = csv_path or ASF_SAMPLE_CSV
    if path.exists():
        df = pd.read_csv(path, nrows=100_000)
    else:
        df = _generate_sample_data(n=SAMPLE_SIZE)
    df = normalize_column_names(df)
    if "summary" not in df.columns:
        raise ValueError("CSV must have a 'summary' or 'title' column")
    if "priority" not in df.columns:
        raise ValueError("CSV must have a 'priority' column")
    if "description" not in df.columns:
        df["description"] = ""
    return df


def _generate_sample_data(n: int = 8000) -> pd.DataFrame:
    """Generate synthetic Jira-like data for testing (no real data)."""
    np.random.seed(RANDOM_STATE)
    priorities = ["Highest", "High", "Medium", "Low", "Lowest"]
    # Rough distribution (imbalanced)
    p_weights = [0.05, 0.15, 0.45, 0.25, 0.10]
    priority_list = np.random.choice(priorities, size=n, p=p_weights)

    summaries_high = [
        "Critical: database connection pool exhausted",
        "Production outage: API returning 500",
        "Security vulnerability in auth module",
        "Data loss risk: replication lag",
        "Blocker: deployment pipeline broken",
    ]
    summaries_med = [
        "Improve error messages in login flow",
        "Refactor payment service",
        "Add logging for request tracing",
        "Update dependency to fix CVE",
        "Performance degradation under load",
    ]
    summaries_low = [
        "Typo in documentation",
        "Minor UI alignment issue",
        "Optional feature request",
        "Code style cleanup",
    ]

    def sample_summary(prio: str) -> str:
        if prio in ("Highest", "High"):
            base = np.random.choice(summaries_high)
        elif prio == "Medium":
            base = np.random.choice(summaries_med)
        else:
            base = np.random.choice(summaries_low)
        return base + f" [{np.random.randint(1, 9999)}]"

    rows = []
    for i in range(n):
        p = priority_list[i]
        summary = sample_summary(p)
        desc_len = np.random.randint(0, 200) if np.random.rand() > 0.3 else 0
        description = (
            "Steps to reproduce: 1. Open app 2. Click submit. Expected: success. Actual: error."
            * (desc_len // 80 + 1)
        )[:desc_len] if desc_len else ""
        rows.append({"summary": summary, "description": description, "priority": p})

    return pd.DataFrame(rows)


def prepare_text(df: pd.DataFrame) -> pd.Series:
    """Combine summary and description and clean."""
    summary = df["summary"].fillna("").astype(str).apply(clean_text)
    desc = df["description"].fillna("").astype(str).apply(clean_text)
    combined = summary + " " + desc
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()


def prepare_labels(df: pd.DataFrame) -> pd.Series:
    """Extract and clean priority labels."""
    labels = df["priority"].astype(str).str.strip()
    # Normalize common variants
    labels = labels.replace(
        {"Blocker": "Highest", "Critical": "High", "Major": "Medium", "Minor": "Low"}
    )
    return labels
