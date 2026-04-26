"""
flasheda
~~~~~~~~
Constant-time Exploratory Data Analysis.

Usage
-----
    import flasheda
    report = flasheda.analyze(df)          # one line
    report.show()                          # rich console
    report.save_html("report.html")        # browser report

The analysis always takes roughly the same time regardless of
dataset size, because every analyzer works on a fixed-size
reservoir sample (default: 5,000 rows).

Public API
----------
  analyze(source, n=5000, show=False, random_state=42) → EDAReport
  set_sample_size(n)
  get_sample_size() → int
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union, Optional

import pandas as pd

from .sampler import sample, get_sample_size, set_sample_size
from .analyzers import overview, missing, numeric, categorical, correlation
from .report import EDAReport

__version__ = "0.1.0"
__all__ = ["analyze", "set_sample_size", "get_sample_size", "EDAReport"]


def analyze(
    source: Union[pd.DataFrame, str, Path],
    n: Optional[int] = None,
    *,
    show: bool = False,
    random_state: int = 42,
) -> EDAReport:
    """
    Perform constant-time EDA on any dataset.

    Parameters
    ----------
    source       : pandas DataFrame or path to a CSV / Parquet file
    n            : sample size (default: flasheda.get_sample_size() == 5000)
    show         : if True, print the rich console summary immediately
    random_state : RNG seed for reproducible sampling

    Returns
    -------
    EDAReport
        A structured report object with .show(), .to_html(),
        .save_html(), .to_dict(), and .to_json() methods.

    Example
    -------
    >>> import flasheda
    >>> report = flasheda.analyze(df)
    >>> report.show()
    >>> report.save_html()
    """
    t0 = time.perf_counter()
    n = n or get_sample_size()

    # ── Step 1: Reservoir sample (constant time regardless of dataset size) ──
    original_shape: tuple
    if isinstance(source, pd.DataFrame):
        original_shape = source.shape
    else:
        # For files we don't know the total rows until we read;
        # set a placeholder updated after sampling.
        original_shape = (-1, -1)

    sampled = sample(source, n=n, random_state=random_state)

    if original_shape == (-1, -1):
        # Best we can do for streamed files: report sampled shape
        original_shape = sampled.shape

    # ── Step 2: Run all analyzers in parallel ────────────────────────────────
    tasks = {
        "overview":    lambda: overview.analyze(sampled, original_shape),
        "missing":     lambda: missing.analyze(sampled),
        "numeric":     lambda: numeric.analyze(sampled),
        "categorical": lambda: categorical.analyze(sampled),
        "correlation": lambda: correlation.analyze(sampled),
    }

    results: dict = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        future_map = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {"error": str(exc)}

    # ── Step 3: Collect warnings ─────────────────────────────────────────────
    warnings: list[str] = []
    miss = results.get("missing", {})
    if miss.get("critical_columns"):
        warnings.append(f"Columns with >50% missing: {', '.join(miss['critical_columns'])}")
    num = results.get("numeric", {})
    if num.get("skewed_columns"):
        warnings.append(f"Highly skewed columns (|skew| > 1): {', '.join(num['skewed_columns'])}")
    cat = results.get("categorical", {})
    if cat.get("likely_id_columns"):
        warnings.append(f"Likely ID columns (consider dropping): {', '.join(cat['likely_id_columns'])}")
    corr = results.get("correlation", {})
    if corr.get("numeric_strong_pairs"):
        pairs = corr["numeric_strong_pairs"]
        warnings.append(f"{len(pairs)} strongly correlated numeric pair(s) found (|r| ≥ 0.8)")

    elapsed = round(time.perf_counter() - t0, 3)

    report = EDAReport(
        overview=results.get("overview", {}),
        missing=results.get("missing", {}),
        numeric=results.get("numeric", {}),
        categorical=results.get("categorical", {}),
        correlation=results.get("correlation", {}),
        sample_size=len(sampled),
        elapsed_seconds=elapsed,
        original_shape=original_shape,
        warnings=warnings,
    )

    if show:
        report.show()

    return report