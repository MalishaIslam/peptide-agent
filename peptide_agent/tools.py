"""The three tools the agent can call.

Each returns plain JSON-serialisable data and reports failure as data rather
than raising, so a bad call teaches the model to retry instead of killing
the loop.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .batch import process_csv
from .core import PeptideError, compute_all

_DATASET: pd.DataFrame | None = None
_DATASET_PATH: Path | None = None

NUMERIC_FIELDS = (
    "length", "mol_weight", "net_charge", "gravy",
    "hydrophobic_moment", "isoelectric_point",
    "aromaticity", "fraction_charged",
)


def load_dataset(path: str | Path) -> dict:
    """Compute descriptors for a CSV once and hold it for later queries."""
    global _DATASET, _DATASET_PATH
    results, errors = process_csv(path)
    _DATASET, _DATASET_PATH = results, Path(path)
    return {
        "loaded": len(results),
        "failed": len(errors),
        "columns": list(results.columns),
        "errors": errors.to_dict("records")[:10],
    }


# --- tool 1 -----------------------------------------------------------------

def compute_properties(sequence: str) -> dict:
    """Physicochemical descriptors for a single peptide sequence."""
    try:
        return {"ok": True, **compute_all(sequence).to_dict()}
    except PeptideError as e:
        return {"ok": False, "error": str(e)}


# --- tool 2 -----------------------------------------------------------------

def query_dataset(filters: list[dict] | None = None,
                  sort_by: str | None = None,
                  descending: bool = True,
                  limit: int = 20) -> dict:
    """Filter the loaded dataset.

    filters: [{"field": "net_charge", "op": ">", "value": 4}, ...]
    Supported ops: > >= < <= == !=
    """
    if _DATASET is None:
        return {"ok": False, "error": "no dataset loaded; call load_dataset first"}

    df = _DATASET
    ops = {
        ">": lambda s, v: s > v, ">=": lambda s, v: s >= v,
        "<": lambda s, v: s < v, "<=": lambda s, v: s <= v,
        "==": lambda s, v: s == v, "!=": lambda s, v: s != v,
    }

    for f in filters or []:
        field, op, value = f.get("field"), f.get("op"), f.get("value")
        if field not in df.columns:
            return {"ok": False,
                    "error": f"unknown field {field!r}; available: {list(df.columns)}"}
        if op not in ops:
            return {"ok": False, "error": f"unknown operator {op!r}; use {list(ops)}"}
        df = df[ops[op](df[field], value)]

    if sort_by:
        if sort_by not in df.columns:
            return {"ok": False, "error": f"unknown sort field {sort_by!r}"}
        df = df.sort_values(sort_by, ascending=not descending)

    shown = df.head(limit)
    keep = list(shown.columns)
    return {
        "ok": True,
        "matched": len(df),
        "returned": len(shown),
        "total_in_dataset": len(_DATASET),
        "rows": shown[keep].to_dict("records"),
    }


def dataset_summary() -> dict:
    """Shape of the loaded dataset and the range of each numeric field."""
    if _DATASET is None:
        return {"ok": False, "error": "no dataset loaded"}
    stats = {}
    for f in NUMERIC_FIELDS:
        if f in _DATASET.columns:
            s = _DATASET[f]
            stats[f] = {"min": round(float(s.min()), 3),
                        "max": round(float(s.max()), 3),
                        "mean": round(float(s.mean()), 3)}
    return {"ok": True, "n_peptides": len(_DATASET),
            "columns": list(_DATASET.columns), "ranges": stats}


# --- tool 3 -----------------------------------------------------------------

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, max_results: int = 5) -> dict:
    """Search PubMed and return titles, journals, years and PMIDs."""
    try:
        params = {"db": "pubmed", "term": query, "retmax": str(max_results),
                  "retmode": "json", "sort": "relevance"}
        if key := os.environ.get("NCBI_API_KEY"):
            params["api_key"] = key

        with urllib.request.urlopen(
            f"{_EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}", timeout=20
        ) as r:
            ids = json.load(r)["esearchresult"].get("idlist", [])

        if not ids:
            return {"ok": True, "query": query, "count": 0, "articles": []}

        sp = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        if key := os.environ.get("NCBI_API_KEY"):
            sp["api_key"] = key
        with urllib.request.urlopen(
            f"{_EUTILS}/esummary.fcgi?{urllib.parse.urlencode(sp)}", timeout=20
        ) as r:
            summary = json.load(r)["result"]

        articles = [{
            "pmid": i,
            "title": summary[i].get("title", ""),
            "journal": summary[i].get("source", ""),
            "year": (summary[i].get("pubdate", "") or "")[:4],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
        } for i in ids if i in summary]

        return {"ok": True, "query": query, "count": len(articles), "articles": articles}

    except Exception as e:
        return {"ok": False, "error": f"PubMed request failed: {type(e).__name__}: {e}"}


# --- schemas handed to the model -------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "compute_properties",
        "description": (
            "Compute physicochemical descriptors for ONE linear peptide of "
            "standard amino acids (L- or D-residues; they give identical "
            "values here). "
            "Returns length, molecular weight, net charge at pH 7, "
            "GRAVY hydrophobicity, hydrophobic moment (amphipathicity), "
            "isoelectric point, aromaticity and fraction of charged residues. "
            "IMPORTANT: net charge counts IONIZABLE SIDE CHAINS ONLY -- the N- and "
            "C-terminal groups are excluded, so the value is comparable across "
            "free, amidated and acylated peptides but is about 1 unit lower at "
            "pH 7 than the conventional figure. Always say so when reporting it. "
            "isoelectric_point is null when the charge curve never crosses zero "
            "(e.g. an all-basic peptide); report that as 'no pI under this "
            "convention', never as zero or as a missing value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sequence": {"type": "string",
                                        "description": "One-letter amino acid sequence, e.g. GLFDIVKKVVGALGSL"}},
            "required": ["sequence"],
        },
    },
    {
        "name": "query_dataset",
        "description": (
            "Filter and sort the loaded peptide dataset by any computed descriptor. "
            "Use this for questions about 'my peptides' or 'the dataset'. "
            "Call with no filters to see the whole set. Combine filters to narrow."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "array",
                    "description": "Conditions combined with AND",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "enum": list(NUMERIC_FIELDS)},
                            "op": {"type": "string", "enum": [">", ">=", "<", "<=", "==", "!="]},
                            "value": {"type": "number"},
                        },
                        "required": ["field", "op", "value"],
                    },
                },
                "sort_by": {"type": "string", "enum": list(NUMERIC_FIELDS)},
                "descending": {"type": "boolean"},
                "limit": {"type": "integer", "description": "Max rows to return (default 20)"},
            },
        },
    },
    {
        "name": "dataset_summary",
        "description": "Size of the loaded dataset and the min/max/mean of every descriptor. "
                       "Call this first when you need to choose sensible filter thresholds.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_pubmed",
        "description": "Search PubMed for published literature. Use for questions about what is "
                       "known or reported about a peptide, a family, or an activity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "PubMed search terms"},
                "max_results": {"type": "integer", "description": "Default 5"},
            },
            "required": ["query"],
        },
    },
]

TOOL_FUNCTIONS = {
    "compute_properties": compute_properties,
    "query_dataset": query_dataset,
    "dataset_summary": dataset_summary,
    "search_pubmed": search_pubmed,
}
