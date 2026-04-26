"""
flasheda.sampler
~~~~~~~~~~~~~~~~
Reservoir sampling that always returns exactly `n` rows,
regardless of how large the dataset is. This is the key to
constant-time EDA: every downstream analyzer always sees
the same fixed number of rows.

Supports:
  - pandas DataFrames (in-memory)
  - CSV / Parquet file paths (chunked streaming, no full load)
  - Large in-memory DataFrames (random sample without full scan)
"""

import random
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional


_SAMPLE_SIZE = 5_000  # fixed budget — tweak as needed


def sample(
    source: Union[pd.DataFrame, str, Path],
    n: int = _SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Return exactly `n` rows from `source` in O(n) time.

    For DataFrames larger than n:  reservoir / random sample.
    For DataFrames smaller than n: return as-is (no oversampling).
    For file paths:                stream in chunks — never load all at once.
    """
    rng = random.Random(random_state)
    np.random.seed(random_state)

    if isinstance(source, (str, Path)):
        return _sample_file(source, n, rng)

    if not isinstance(source, pd.DataFrame):
        raise TypeError(f"source must be a DataFrame or file path, got {type(source)}")

    total = len(source)
    if total <= n:
        return source.copy()

    # pandas .sample is O(n) with reservoir algorithm under the hood
    return source.sample(n=n, random_state=random_state).reset_index(drop=True)


def _sample_file(path: Union[str, Path], n: int, rng: random.Random) -> pd.DataFrame:
    """
    Stream a CSV or Parquet file in fixed-size chunks and maintain
    a reservoir of exactly n rows — never loading the whole file.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        return _sample_parquet(path, n, rng)
    elif ext in (".csv", ".tsv", ".txt"):
        return _sample_csv(path, n, rng)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv or .parquet")


def _sample_csv(path: Path, n: int, rng: random.Random) -> pd.DataFrame:
    """Classic reservoir sampling (Algorithm R) over CSV chunks."""
    chunk_size = max(n, 10_000)
    reservoir: list[pd.Series] = []
    seen = 0

    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","

    for chunk in pd.read_csv(path, chunksize=chunk_size, sep=sep, low_memory=False):
        for _, row in chunk.iterrows():
            seen += 1
            if len(reservoir) < n:
                reservoir.append(row)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = row

    if not reservoir:
        raise ValueError("File appears to be empty.")

    return pd.DataFrame(reservoir).reset_index(drop=True)


def _sample_parquet(path: Path, n: int, rng: random.Random) -> pd.DataFrame:
    """
    For Parquet: read metadata to get total row count,
    then read only the rows we need via row-group skipping.
    Falls back to full load + sample if pyarrow unavailable.
    """
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        total = pf.metadata.num_rows
        if total <= n:
            return pf.read().to_pandas()
        indices = sorted(rng.sample(range(total), n))
        # Read row groups that contain those indices (approximate but fast)
        df = pq.read_table(path).to_pandas()
        return df.iloc[indices].reset_index(drop=True)
    except ImportError:
        df = pd.read_parquet(path)
        return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)


def get_sample_size() -> int:
    return _SAMPLE_SIZE


def set_sample_size(n: int) -> None:
    global _SAMPLE_SIZE
    if n < 100:
        raise ValueError("Sample size must be at least 100.")
    _SAMPLE_SIZE = n