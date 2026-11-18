"""Peptide analysis agent: descriptor computation, dataset querying and
literature search, exposed as tools to an LLM agent."""

__version__ = "0.2.0"

from .core import (
    PeptideError, Properties, compute_all, has_d_residues, validate,
)

__all__ = ["PeptideError", "Properties", "compute_all", "has_d_residues",
           "validate"]
