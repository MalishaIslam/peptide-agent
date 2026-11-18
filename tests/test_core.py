"""Reference-value tests. Numbers come from independent sources, not from
running this code and freezing whatever it produced.
"""

import math
import pytest

from peptide_agent.core import (
    PeptideError, validate, compute_all, has_d_residues,
    net_charge, gravy, isoelectric_point, hydrophobic_moment, mol_weight,
)


# --- validation -------------------------------------------------------------

def test_strips_whitespace_and_uppercases():
    assert validate(" gl fd\nIV ") == "GLFDIV"


@pytest.mark.parametrize("bad,msg", [
    ("", "empty"),
    ("   ", "empty"),
    ("GXG", "non-standard"),
    ("GBG", "non-standard"),
    ("GLF-IVK", "non-standard"),   # punctuation
    ("GLF1IVK", "non-standard"),   # digits
    ("A", "at least 2"),
])
def test_rejects_bad_input(bad, msg):
    with pytest.raises(PeptideError) as e:
        validate(bad)
    assert msg in str(e.value)


def test_error_names_the_offending_residue():
    with pytest.raises(PeptideError) as e:
        validate("GXZG")
    assert "X" in str(e.value) and "Z" in str(e.value)


# --- molecular weight against known references ------------------------------

@pytest.mark.parametrize("seq,mw", [
    ("GG",  132.12),   # glycylglycine, CAS 556-50-3
    ("AA",  160.17),   # alanylalanine
    ("GGG", 189.17),   # triglycine, CAS 556-33-2
])
def test_molecular_weight_matches_reference(seq, mw):
    assert compute_all(seq).mol_weight == pytest.approx(mw, abs=0.05)


def test_mw_matches_rdkit_reference_values():
    """Frozen from RDKit Descriptors.MolWt before RDKit was removed. These
    pin the residue-mass table against an independent implementation."""
    for seq, rdkit_mw in [
        ("AA", 160.170), ("GLFDIVKKVVGALGSL", 1615.980),
        ("GIGAVLKVLTTGLPALISWIKRKRQQ", 2847.500), ("KKKKRRRR", 1155.470),
        ("ACDEFGHIKLMNPQRSTVWY", 2395.760),
    ]:
        assert mol_weight(seq) == pytest.approx(rdkit_mw, abs=0.05)


def test_mw_is_additive():
    """Joining two peptides must lose exactly one water."""
    assert mol_weight("GLFD" + "IVKK") == pytest.approx(
        mol_weight("GLFD") + mol_weight("IVKK") - 18.01528, abs=1e-6)


# --- D-residues are in scope now --------------------------------------------

def test_lowercase_d_residues_are_accepted():
    """Charge, hydrophobicity and mass depend on which side chain is present,
    not on its stereochemistry, so D-peptides are valid input here."""
    assert validate("KVvvKWVvKvVK") == "KVVVKWVVKVVK"


def test_enantiomers_share_every_descriptor():
    a = compute_all("KVVVKWVVKVVK")
    b = compute_all("KVvvKWVvKvVK")
    assert a == b


def test_has_d_residues_flags_lowercase():
    assert has_d_residues("KVvvKWVvKvVK")
    assert not has_d_residues("KVVVKWVVKVVK")



# --- charge -----------------------------------------------------------------

def test_polylysine_is_strongly_positive():
    assert net_charge("KKKKK") > 4.0


def test_polyglutamate_is_strongly_negative():
    assert net_charge("EEEEE") < -4.0


def test_charge_falls_as_ph_rises():
    seq = "GLFDIVKKVVGALGSL"
    charges = [net_charge(seq, ph) for ph in (2, 5, 7, 9, 12)]
    assert charges == sorted(charges, reverse=True)


def test_pi_is_the_ph_where_charge_is_zero():
    """Only meaningful for peptides whose charge curve actually crosses zero."""
    for seq in ("GLFDIVKKVVGALGSL", "ACDEFGHIKLMNPQRSTVWY"):
        pi = isoelectric_point(seq)
        assert pi is not None
        assert abs(net_charge(seq, pi)) < 1e-3


def test_pi_is_none_when_no_acidic_residues():
    """Termini are excluded, so an all-basic peptide never reaches zero."""
    assert isoelectric_point("KKKKRRRR") is None


def test_pi_is_none_when_no_basic_residues():
    assert isoelectric_point("DDDDEEEE") is None


def test_pi_is_none_with_no_ionizable_side_chains():
    assert isoelectric_point("LFIFFF") is None


# --- charge excludes the termini --------------------------------------------

def test_charge_ignores_termini():
    """A peptide with no ionizable side chains carries no charge at all,
    because the N- and C-terminal groups are deliberately excluded."""
    assert net_charge("LFIFFF") == pytest.approx(0.0, abs=1e-6)
    assert net_charge("GGGG") == pytest.approx(0.0, abs=1e-6)


def test_charge_is_length_independent_for_neutral_residues():
    """Padding with glycine must not change the charge -- it would if the
    termini were counted, since they would move further apart."""
    assert net_charge("KGGGG") == pytest.approx(net_charge("KGGGGGGGG"), abs=1e-6)


def test_charge_counts_only_side_chains():
    assert net_charge("KKKKK") == pytest.approx(5.0, abs=0.01)
    assert net_charge("EEEEE") == pytest.approx(-5.0, abs=0.01)


# --- hydrophobicity ---------------------------------------------------------

def test_gravy_equals_scale_value_for_homopolymer():
    assert gravy("IIIII") == pytest.approx(4.5)
    assert gravy("RRRRR") == pytest.approx(-4.5)


def test_amphipathic_helix_beats_uniform_sequence():
    """Melittin is the textbook amphipathic peptide; polyleucine is uniform.
    An amphipathicity measure that cannot tell them apart is broken."""
    melittin = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    assert hydrophobic_moment(melittin) > hydrophobic_moment("LLLLLLLLLLLL")


def test_moment_is_non_negative():
    assert hydrophobic_moment("GG") >= 0


# --- whole-record integration ----------------------------------------------

def test_compute_all_is_internally_consistent():
    p = compute_all("GLFDIVKKVVGALGSL")
    assert p.length == 16
    assert not hasattr(p, "smiles")
    assert p.sequence == "GLFDIVKKVVGALGSL"
    assert p.gravy == pytest.approx(gravy(p.sequence), abs=1e-3)
    assert p.net_charge == pytest.approx(net_charge(p.sequence), abs=1e-3)
    assert 0 <= p.aromaticity <= 1
    assert 0 <= p.fraction_charged <= 1


def test_fractions_counted_correctly():
    p = compute_all("FWYAAAAAAA")       # 3 aromatic of 10, 0 charged
    assert p.aromaticity == pytest.approx(0.3)
    assert p.fraction_charged == pytest.approx(0.0)


def test_compute_all_rejects_bad_sequence():
    with pytest.raises(PeptideError):
        compute_all("GLFX")


# --- regression: silent NaN corruption --------------------------------------

def test_nan_is_rejected_not_read_as_asn_ala_asn():
    """An empty CSV cell arrives as float('nan'), whose str() is 'nan' ->
    'NAN' -> a valid tripeptide. It must raise, not silently compute."""
    with pytest.raises(PeptideError):
        validate(float("nan"))


def test_non_string_input_is_rejected():
    for bad in (123, None, ["G", "G"]):
        with pytest.raises(PeptideError):
            validate(bad)
