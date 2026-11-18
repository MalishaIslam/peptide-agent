"""Core peptide descriptors: validation, charge, hydrophobicity, mass.

Scope: linear peptides built from the 20 standard amino acids. L- and
D-residues are both accepted, because every descriptor here depends only on
WHICH side chains are present, not on their stereochemistry: charge comes
from side-chain pKa values, hydrophobicity from per-residue scales, and mass
from per-residue masses. None of those differ between enantiomers.

Pure functions -- no I/O, no printing, no network, no third-party
dependencies. Everything else in the package builds on this.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy scale (J Mol Biol 157:105-132, 1982)
KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Eisenberg consensus scale -- used for the hydrophobic moment, which was
# derived against it. KD is kept separately for the familiar GRAVY number.
EISENBERG_SCALE = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}

# Average (not monoisotopic) residue masses, Da. Validated against RDKit's
# Descriptors.MolWt on a panel of peptides: agreement within 0.03 Da for a
# 20-mer, which is rounding in the published residue masses.
RESIDUE_MASS = {
    "G": 57.0519, "A": 71.0788, "S": 87.0782, "P": 97.1167, "V": 99.1326,
    "T": 101.1051, "C": 103.1388, "L": 113.1594, "I": 113.1594,
    "N": 114.1038, "D": 115.0886, "Q": 128.1307, "K": 128.1741,
    "E": 129.1155, "M": 131.1926, "H": 137.1411, "F": 147.1766,
    "R": 156.1875, "Y": 163.1760, "W": 186.2132,
}
WATER_MASS = 18.01528

PKA = {"K": 10.5, "R": 12.5, "H": 6.0, "D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
POSITIVE_AA = ("K", "R", "H")
NEGATIVE_AA = ("D", "E", "C", "Y")


class PeptideError(ValueError):
    """Raised when a sequence is outside the supported scope."""


@dataclass(frozen=True)
class Properties:
    sequence: str
    length: int
    mol_weight: float
    net_charge: float
    gravy: float
    hydrophobic_moment: float
    isoelectric_point: float | None
    aromaticity: float
    fraction_charged: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate(sequence: str) -> str:
    """Normalise a sequence, or raise PeptideError explaining why it is unusable.

    Returns the cleaned uppercase sequence. Whitespace anywhere is stripped,
    so FASTA-wrapped input works. Lowercase is uppercased: in databases such
    as DBAASP lowercase denotes a D-residue, and that is deliberately
    flattened here because no descriptor in this module distinguishes
    enantiomers. If you later add a structural or stereochemistry-dependent
    descriptor, this is the line that has to change.
    """
    if sequence is None:
        raise PeptideError("sequence is missing")
    # A float NaN (an empty CSV cell) stringifies to "nan", which uppercases
    # to "NAN" -- a valid Asn-Ala-Asn tripeptide. Catch it before it becomes
    # data.
    if isinstance(sequence, float) and math.isnan(sequence):
        raise PeptideError("sequence is empty")
    if not isinstance(sequence, str):
        raise PeptideError(
            f"sequence must be a string, got {type(sequence).__name__}: {sequence!r}"
        )
    seq = "".join(sequence.split()).upper()
    if not seq:
        raise PeptideError("sequence is empty")
    bad = sorted(set(seq) - STANDARD_AA)
    if bad:
        raise PeptideError(
            f"non-standard residue(s) {''.join(bad)}; "
            "only the 20 standard amino acids are supported"
        )
    if len(seq) < 2:
        raise PeptideError("sequence must be at least 2 residues")
    return seq


def has_d_residues(sequence: str) -> bool:
    """True if the raw sequence contains lowercase letters.

    Databases such as DBAASP write D-residues in lowercase. This does not
    affect any descriptor here, but it does affect the biology, so it is
    worth recording alongside the results.
    """
    return any(c.islower() for c in str(sequence))


def mol_weight(sequence: str) -> float:
    """Average molecular weight from residue masses plus one water."""
    seq = validate(sequence)
    return sum(RESIDUE_MASS[aa] for aa in seq) + WATER_MASS


def _charge_at(pka: float, ph: float, positive: bool) -> float:
    return (
        1.0 / (1.0 + 10 ** (ph - pka)) if positive
        else -1.0 / (1.0 + 10 ** (pka - ph))
    )


def net_charge(sequence: str, ph: float = 7.0) -> float:
    """Net charge from IONIZABLE SIDE CHAINS ONLY, by Henderson-Hasselbalch.

    The N- and C-terminal groups are deliberately excluded, so the value is
    independent of how the termini are capped and is therefore comparable
    across free, amidated and acylated peptides. It is NOT the conventional
    net charge, which includes both termini and would be roughly +1 higher at
    pH 7 for a free-terminus peptide. Say which convention you used whenever
    you report these numbers.
    """
    seq = validate(sequence)
    charge = 0.0
    for aa in seq:
        if aa in POSITIVE_AA:
            charge += _charge_at(PKA[aa], ph, True)
        elif aa in NEGATIVE_AA:
            charge += _charge_at(PKA[aa], ph, False)
    return charge


def isoelectric_point(sequence: str) -> float | None:
    """pH at which side-chain net charge is zero, or None if undefined.

    With the termini excluded there is no guarantee the charge curve crosses
    zero: an all-basic peptide (KKKKK) stays positive at every pH, an
    all-acidic one stays negative, and a peptide with no ionizable side
    chains (LFIFFF) is flat at zero. None means "no pI exists under this
    convention", which is information, not a failure.
    """
    seq = validate(sequence)
    lo_q, hi_q = net_charge(seq, 0.0), net_charge(seq, 14.0)
    if lo_q == 0.0 and hi_q == 0.0:
        return None                      # no ionizable side chains at all
    if (lo_q > 0) == (hi_q > 0):
        return None                      # curve never crosses zero
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gravy(sequence: str) -> float:
    """Grand average of hydropathy: mean Kyte-Doolittle value per residue."""
    seq = validate(sequence)
    return sum(KD_SCALE[aa] for aa in seq) / len(seq)


def hydrophobic_moment(sequence: str, angle: float = 100.0, window: int = 11) -> float:
    """Maximum Eisenberg hydrophobic moment over a sliding window.

    angle=100 degrees is the alpha-helix periodicity. This measures
    amphipathicity -- whether hydrophobic residues segregate to one face of
    the helix, which is the main structural determinant of membrane activity.
    """
    seq = validate(sequence)
    w = min(window, len(seq))
    rad = math.radians(angle)
    best = 0.0
    for start in range(len(seq) - w + 1):
        sin_sum = cos_sum = 0.0
        for i, aa in enumerate(seq[start:start + w]):
            h = EISENBERG_SCALE[aa]
            sin_sum += h * math.sin(i * rad)
            cos_sum += h * math.cos(i * rad)
        best = max(best, math.sqrt(sin_sum**2 + cos_sum**2) / w)
    return best


def compute_all(sequence: str) -> Properties:
    """Every descriptor for one sequence."""
    seq = validate(sequence)
    charged = sum(seq.count(aa) for aa in ("D", "E", "K", "R", "H"))
    aromatic = sum(seq.count(aa) for aa in ("F", "W", "Y"))
    pi = isoelectric_point(seq)

    return Properties(
        sequence=seq,
        length=len(seq),
        mol_weight=round(mol_weight(seq), 3),
        net_charge=round(net_charge(seq), 3),
        gravy=round(gravy(seq), 3),
        hydrophobic_moment=round(hydrophobic_moment(seq), 3),
        isoelectric_point=None if pi is None else round(pi, 2),
        aromaticity=round(aromatic / len(seq), 3),
        fraction_charged=round(charged / len(seq), 3),
    )
