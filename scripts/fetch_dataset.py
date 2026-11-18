"""Fetch a scope-compatible peptide set from DBAASP.

    python scripts/fetch_dataset.py --limit 800

Writes data/peptides.csv. The CSV is gitignored on purpose: this script is
what the repository ships, not somebody else's database.

Scope filtering, in order of how much it removes:
  1. complexity = monomer      (API-side; drops multimers, whose top-level
                                sequence field is empty anyway)
  2. passes core.validate()    (20 standard amino acids, length >= 2)

D-residues (which DBAASP writes in lowercase) are KEPT. Charge,
hydrophobicity and mass depend on which side chain is present, not on its
stereochemistry, so they are identical for either enantiomer. The
has_d_residues column records them anyway, because the distinction matters
biologically even though it does not change these numbers.

Terminal modifications are NOT filtered out. They are recorded in the
n_terminus / c_terminus columns instead, because every descriptor here is
computed from side chains only and so does not depend on them. One caveat:
molecular weight is the sum of residue masses plus one water, so for a
capped peptide it describes the uncapped analogue -- about 1 Da low for an
amide, more for an acyl group.

Each rejection is counted and reported so the scope mismatch is visible
rather than silent.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from peptide_agent.core import (  # noqa: E402
    PeptideError, has_d_residues, validate,
)

API = "https://dbaasp.org/peptides"
PAGE = 100


def fetch_page(limit: int, offset: int, timeout: int = 30) -> tuple[list[dict], int]:
    params = {"complexity.value": "monomer", "limit": str(limit), "offset": str(offset)}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.load(r)
    # The API has been seen to use both keys for the record list.
    records = payload.get("data") or payload.get("records") or []
    return records, int(payload.get("totalCount", 0))


def in_scope(rec: dict) -> tuple[bool, str]:
    """Return (keep, reason_if_dropped)."""
    seq = (rec.get("sequence") or "").strip()

    if not seq:
        return False, "no sequence"
    try:
        validate(seq)
    except PeptideError as e:
        return False, str(e).split(";")[0]
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=800,
                    help="max peptides to KEEP (default 800)")
    ap.add_argument("--max-pages", type=int, default=60,
                    help="safety cap on API requests (default 60)")
    ap.add_argument("--out", default="data/peptides.csv")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds between requests (default 0.3)")
    args = ap.parse_args()

    kept: list[dict] = []
    dropped = Counter()
    seen_sequences: set[str] = set()
    offset = total = 0

    print(f"fetching monomers from {API}")

    for page in range(args.max_pages):
        try:
            records, total = fetch_page(PAGE, offset)
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} at offset {offset}: {e.reason}", file=sys.stderr)
            break
        except Exception as e:
            print(f"  request failed at offset {offset}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            break

        if not records:
            break

        for rec in records:
            keep, reason = in_scope(rec)
            if not keep:
                dropped[reason] += 1
                continue

            seq = rec["sequence"].strip().upper()
            if seq in seen_sequences:
                dropped["duplicate sequence"] += 1
                continue
            seen_sequences.add(seq)

            kept.append({
                "peptide_id": rec.get("dbaaspId") or f"DBAASP_{rec.get('id')}",
                "Sequence": seq,
                "name": (rec.get("name") or "").strip(),
                "has_d_residues": has_d_residues(rec["sequence"]),
                "n_terminus": rec.get("nTerminus") or "free",
                "c_terminus": rec.get("cTerminus") or "free",
                "synthesis_type": rec.get("synthesisType") or "",
            })
            if len(kept) >= args.limit:
                break

        offset += PAGE
        print(f"  page {page + 1}: scanned {offset}, kept {len(kept)}")

        if len(kept) >= args.limit or (total and offset >= total):
            break
        time.sleep(args.delay)

    if not kept:
        print("\nnothing kept -- check the API response shape before retrying.",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["peptide_id", "Sequence", "name",
                               "has_d_residues", "n_terminus", "c_terminus",
                               "synthesis_type"])
        w.writeheader()
        w.writerows(kept)

    scanned = sum(dropped.values()) + len(kept)
    print(f"\nwrote {len(kept)} peptides to {out}")
    print(f"scanned {scanned} records (DBAASP reports {total} monomers total)")
    print("\nrejected, by reason:")
    for reason, n in dropped.most_common():
        print(f"  {n:6d}  {reason}")
    capped = sum(1 for r in kept
                 if r["n_terminus"] != "free" or r["c_terminus"] != "free")
    d_count = sum(1 for r in kept if r["has_d_residues"])
    print(f"\n{capped} of {len(kept)} kept peptides have a modified terminus; "
          f"{d_count} contain D-residues.")
    print("Neither affects charge, hydrophobicity or mass, which depend only "
          "on side-chain\nidentity. Both are recorded as columns.")
    print("\nThese rejections are the scope filter working, not errors. "
          "Quote this breakdown in your README.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
