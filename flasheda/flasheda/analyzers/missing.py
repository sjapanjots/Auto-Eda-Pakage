"""
flasheda.analyzers.missing
~~~~~~~~~~~~~~~~~~~~~~~~~~
Missing-value analysis on the fixed sample.
Detects per-column null rates and flags high-missing columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


WARN_THRESHOLD = 0.20   # 20%+ missing → flag
CRITICAL_THRESHOLD = 0.50  # 50%+ missing → critical


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    per_col: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round(null_count / n * 100, 2) if n > 0 else 0.0
        severity = "ok"
        if null_pct >= CRITICAL_THRESHOLD * 100:
            severity = "critical"
        elif null_pct >= WARN_THRESHOLD * 100:
            severity = "warn"

        per_col[col] = {
            "null_count": null_count,
            "null_pct": null_pct,
            "severity": severity,
        }

    total_cells = n * len(df.columns)
    total_nulls = sum(v["null_count"] for v in per_col.values())
    overall_null_pct = round(total_nulls / total_cells * 100, 2) if total_cells > 0 else 0.0

    warn_cols = [c for c, v in per_col.items() if v["severity"] in ("warn", "critical")]
    critical_cols = [c for c, v in per_col.items() if v["severity"] == "critical"]

    # Detect rows with any null (approximate missing-row pattern)
    rows_with_any_null = int(df.isna().any(axis=1).sum())
    rows_with_any_null_pct = round(rows_with_any_null / n * 100, 2) if n > 0 else 0.0

    return {
        "per_column": per_col,
        "overall_null_pct": overall_null_pct,
        "warn_columns": warn_cols,
        "critical_columns": critical_cols,
        "rows_with_any_null": rows_with_any_null,
        "rows_with_any_null_pct": rows_with_any_null_pct,
    }