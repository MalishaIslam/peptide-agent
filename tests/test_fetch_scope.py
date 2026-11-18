"""Tests for the DBAASP scope filter.

These pin the rules that keep out-of-scope records out of the dataset. They
use record shapes taken from real DBAASP API responses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fetch_dataset import in_scope  # noqa: E402


def rec(**kw):
    base = {"id": 1, "dbaaspId": "DBAASPS_1", "name": "", "sequence": "GLFDIVKK",
            "complexity": "monomer", "nTerminus": None, "cTerminus": None,
            "synthesisType": "Synthetic"}
    base.update(kw)
    return base


def test_keeps_a_clean_monomer():
    keep, reason = in_scope(rec())
    assert keep, reason


def test_keeps_d_amino_acids():
    """Lowercase marks a D-residue. Charge, hydrophobicity and mass are
    identical for either enantiomer, so these are in scope."""
    keep, reason = in_scope(rec(sequence="KVvvKWVvKvVK"))
    assert keep, reason


def test_keeps_amidated_c_terminus():
    """Terminal caps are kept and recorded, not filtered: net charge comes
    from side chains only, so it does not depend on them."""
    keep, reason = in_scope(rec(cTerminus="AMD"))
    assert keep, reason


def test_keeps_acylated_n_terminus():
    keep, reason = in_scope(rec(nTerminus="C16"))
    assert keep, reason


def test_keeps_a_capped_d_peptide():
    keep, reason = in_scope(rec(sequence="KVvvKWVvKvVK", cTerminus="AMD"))
    assert keep, reason


def test_rejects_empty_sequence_of_a_multimer():
    keep, reason = in_scope(rec(sequence="", complexity="multimer"))
    assert not keep
    assert "no sequence" in reason


def test_rejects_non_standard_residues():
    keep, reason = in_scope(rec(sequence="GLFXIVKK"))
    assert not keep
    assert "non-standard" in reason


def test_rejects_too_short():
    keep, _ = in_scope(rec(sequence="G"))
    assert not keep


def test_whitespace_padded_sequence_is_kept():
    keep, reason = in_scope(rec(sequence="  GLFDIVKK  "))
    assert keep, reason
