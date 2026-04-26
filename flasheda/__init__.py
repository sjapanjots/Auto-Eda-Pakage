from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union, Optional

import pandas as pd

from .sampler import sample, get_sample_size, set_sample_size
from .analyzers import overview, missing, numeric, categorical, correlation
from .report import EDAReport

__version__ = "0.1.2"
__all__ = ["analyze", "set_sample_size", "get_sample_size", "EDAReport"]


def analyze(
    source: Union[pd.DataFrame, str, Path],
    n: Optional[int] = None,
    *,
    show: bool = False,
    random_state: int = 42,
    save_pdf: bool = True,                        # ← NEW: auto-save PDF
    pdf_path: str = "flasheda_report.pdf",        # ← NEW: PDF file name
) -> EDAReport:
    """
    Perform constant-time EDA on any dataset.

    Parameters
    ----------
    source       : pandas DataFrame or path to CSV / Parquet file
    n            : sample size (default 5000)
    show         : print rich console summary
    random_state : RNG seed
    save_pdf     : automatically save PDF report (default True)
    pdf_path     : path/name of the PDF file to save

    Example
    -------
    >>> report = flasheda.analyze(df)
    # flasheda_report.pdf is saved automatically
    """
    t0 = time.perf_counter()
    n = n or get_sample_size()

    if isinstance(source, pd.DataFrame):
        original_shape = source.shape
    else:
        original_shape = (-1, -1)

    sampled = sample(source, n=n, random_state=random_state)

    if original_shape == (-1, -1):
        original_shape = sampled.shape

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

    warnings: list[str] = []
    miss = results.get("missing", {})
    if miss.get("critical_columns"):
        warnings.append(f"Columns with >50% missing: {', '.join(miss['critical_columns'])}")
    num = results.get("numeric", {})
    if num.get("skewed_columns"):
        warnings.append(f"Highly skewed columns: {', '.join(num['skewed_columns'])}")
    cat = results.get("categorical", {})
    if cat.get("likely_id_columns"):
        warnings.append(f"Likely ID columns: {', '.join(cat['likely_id_columns'])}")
    corr = results.get("correlation", {})
    if corr.get("numeric_strong_pairs"):
        warnings.append(f"{len(corr['numeric_strong_pairs'])} strongly correlated pair(s) found")

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

    # ── Auto-save PDF immediately after analysis ───────────────────────────
    if save_pdf:
        report.save_pdf(pdf_path)

    return report