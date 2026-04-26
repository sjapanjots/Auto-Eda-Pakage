"""
flasheda.analyzers.numeric
~~~~~~~~~~~~~~~~~~~~~~~~~~
Statistical summary for numeric columns.
Detects outliers using IQR, computes skewness & kurtosis,
and flags heavily skewed distributions.
All computed on the fixed sample — constant time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


SKEW_THRESHOLD = 1.0   # |skew| > 1 → flag


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    results: Dict[str, Dict[str, Any]] = {}

    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 4:
            results[col] = {"error": "too few non-null values"}
            continue

        q1, q50, q75 = float(np.percentile(s, 25)), float(np.percentile(s, 50)), float(np.percentile(s, 75))
        iqr = q75 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        outliers = s[(s < lower_fence) | (s > upper_fence)]
        outlier_pct = round(len(outliers) / len(s) * 100, 2)

        skewness = float(s.skew())
        kurt = float(s.kurtosis())
        mean_val = float(s.mean())
        std_val = float(s.std())
        min_val = float(s.min())
        max_val = float(s.max())

        skew_flag = abs(skewness) > SKEW_THRESHOLD

        results[col] = {
            "count": int(len(s)),
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "q25": round(q1, 4),
            "median": round(q50, 4),
            "q75": round(q75, 4),
            "max": round(max_val, 4),
            "iqr": round(iqr, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurt, 4),
            "outlier_count": int(len(outliers)),
            "outlier_pct": outlier_pct,
            "skewed": skew_flag,
            "zero_variance": std_val < 1e-10,
        }

    skewed_cols = [c for c, v in results.items() if v.get("skewed")]
    high_outlier_cols = [c for c, v in results.items() if v.get("outlier_pct", 0) > 5]

    return {
        "columns": results,
        "numeric_col_names": num_cols,
        "skewed_columns": skewed_cols,
        "high_outlier_columns": high_outlier_cols,
    }