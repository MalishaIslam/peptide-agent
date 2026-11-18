"""CSV in, results out. One bad row never kills the job."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .core import PeptideError, compute_all

SEQUENCE_COLUMNS = ("sequence", "seq", "peptide", "peptide_sequence")


def find_sequence_column(df: pd.DataFrame) -> str:
    """Locate the sequence column case-insensitively, or say what was found."""
    lookup = {c.lower().strip(): c for c in df.columns}
    for cand in SEQUENCE_COLUMNS:
        if cand in lookup:
            return lookup[cand]
    raise PeptideError(
        f"no sequence column found. Looked for {SEQUENCE_COLUMNS}; "
        f"the file has {list(df.columns)}"
    )


def process_frame(df: pd.DataFrame, column: str | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute descriptors for every row.

    Returns (results, errors). Results keeps the original columns and appends
    the descriptors. Errors lists the row number, the input, and why it failed.
    """
    col = column or find_sequence_column(df)
    good, bad = [], []

    for i, raw in enumerate(df[col]):
        try:
            good.append({"_row": i, **compute_all(raw).to_dict()})
        except PeptideError as e:
            bad.append({"row": i + 2, "input": raw, "error": str(e)})  # +2 = 1-indexed + header

    results = pd.DataFrame(good)
    if not results.empty:
        # Re-attach the caller's original columns by position, not by index,
        # so a filtered or non-default index cannot misalign them.
        original = df.reset_index(drop=True).iloc[results["_row"]].reset_index(drop=True)
        extra = [c for c in original.columns if c != col]
        results = pd.concat(
            [original[extra].reset_index(drop=True), results.drop(columns=["_row"])],
            axis=1,
        )

    return results, pd.DataFrame(bad, columns=["row", "input", "error"])


def process_csv(path: str | Path, column: str | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    return process_frame(pd.read_csv(path), column)
