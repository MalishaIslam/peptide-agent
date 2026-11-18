# Peptide Analysis Agent

An LLM agent that answers open-ended questions about peptides by deciding which
tools to call and in what order — computing physicochemical descriptors,
querying a peptide dataset, and searching the published literature.

<!-- Drag your screen recording into the GitHub README editor here. GitHub
     embeds .mov and .mp4 directly; no GIF conversion needed. -->

https://github.com/user-attachments/assets/0a7b9fbe-a34f-4f03-a076-469f0955ddae


https://github.com/user-attachments/assets/b72d6273-d197-4b95-8c61-7565c2e0b397



https://github.com/user-attachments/assets/b4ff499f-388c-4a08-af56-6c369de6803f




Asked *"which peptides in my dataset are strongly amphipathic?"*, the agent
calls `dataset_summary` to find the actual μH range (0.04–1.39, mean 0.59),
picks μH ≥ 0.9 as the top ~8%, adds a length ≥ 11 rule because very short
peptides score artificially high, then calls `query_dataset` to filter.
Nothing in the code chose that sequence — the threshold is not in the source,
and the second call depends on the first call's output.

> **Scope:** linear peptides built from the 20 standard amino acids, L- or
> D-. Non-standard residues are rejected rather than silently approximated.
> Research tool, not for clinical use.

> **Why D-residues are in scope:** every descriptor here depends on *which*
> side chains are present, not on their stereochemistry. Charge comes from
> side-chain pKa values, hydrophobicity from per-residue scales, mass from
> per-residue masses — none differ between enantiomers. DBAASP writes
> D-residues in lowercase; they are uppercased and flagged in a
> `has_d_residues` column, since the distinction matters biologically even
> though it does not change these numbers. Add any structure-dependent
> descriptor and `core.validate()` is the line that has to change.

> **Charge convention:** net charge counts **ionizable side chains only**. The
> N- and C-terminal groups are excluded, so the number is comparable across
> free, amidated and acylated peptides — but it is roughly 1 unit lower at
> pH 7 than the conventional figure, and it will not match ExPASy or
> Biopython. A consequence worth knowing: with the termini gone the charge
> curve need not cross zero, so **pI is `None`** for an all-basic peptide
> (`KKKKRRRR`), an all-acidic one (`DDDDEEEE`), or one with no ionizable side
> chains (`LFIFFF`). `None` means "no pI exists under this convention" — that
> is information, not a failure.

> **Terminal caps:** peptides with modified termini are kept, and the
> modification is recorded in the `n_terminus` / `c_terminus` columns. Charge
> and hydrophobicity are unaffected. Molecular weight is the sum of residue
> masses plus one water, so for a capped peptide it describes the uncapped
> analogue — about 1 Da low for an amide. Check the terminus columns before
> quoting mass.

## What makes it an agent

The tools are deterministic. The agency is in the routing: for a question like

> *"Find the most positively charged peptide in my dataset, then tell me what
> the literature says about cationic antimicrobial peptides."*

nothing in the code decides to query the dataset before searching PubMed, or to
check the dataset's descriptor ranges before picking a threshold. The model
works that out from the tool descriptions and the results it gets back.

```
question ─► model ─► tool call ─► result ─┐
             ▲                             │
             └─────────── loop ────────────┘
                                   └─► prose answer
```

The loop is ~40 lines in `peptide_agent/agent.py`, written by hand rather than
with a framework, because the mechanism is the part worth understanding.

## Tools

The loop runs on **either Anthropic or Gemini** (`BACKENDS.md`), both given
byte-identical tool descriptions, so the eval suite can measure routing
differences between models rather than between prompts.

| Tool | Does |
|---|---|
| `compute_properties` | 8 descriptors for one sequence |
| `query_dataset` | Filter/sort a loaded CSV by any descriptor |
| `dataset_summary` | Dataset size and per-descriptor ranges |
| `search_pubmed` | Literature search via NCBI E-utilities |

All four are also exposed over **MCP** (`peptide_agent/mcp_server.py`), so any
MCP client — Claude Desktop included — can use them directly.

## Descriptors

Length, molecular weight, net charge at pH 7, GRAVY (Kyte–Doolittle),
**hydrophobic moment** (Eisenberg, 100° helical periodicity — the
amphipathicity measure that predicts membrane activity), isoelectric point,
aromaticity, and fraction of charged residues.

All pure Python — **no cheminformatics dependency**. The package needs pandas
and one model SDK (Anthropic or Google); the descriptor layer itself needs
neither.

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                 # 63 tests

python scripts/fetch_dataset.py           # builds data/peptides.csv from DBAASP
export GEMINI_API_KEY=...                 # free: aistudio.google.com
                                          # (or ANTHROPIC_API_KEY)

python demo.py                            # scripted questions, prints traces
streamlit run app.py                      # UI: single / batch / agent
python -m evals.run_evals                 # eval suite
```

See `QUICKSTART.md` for the same thing with expected output at each step.

## Data

The dataset is **fetched, not vendored**. `scripts/fetch_dataset.py` pulls
monomeric peptides from the [DBAASP](https://dbaasp.org/) REST API
(`GET /peptides?complexity.value=monomer`) and filters them to this project's
scope. `data/*.csv` is gitignored — this repository ships the fetch code, not
somebody else's database. Check DBAASP's terms for attribution requirements
before publishing results built on it.

Three filters, applied in order, each counted and reported:

| Filter | Drops |
|---|---|
| `complexity = monomer` | Multimers, whose top-level `sequence` is empty |
| deduplication by sequence | Repeated peptides (see below) |
| `core.validate()` | Non-standard residues, length < 2 |

**A representative run** — 977 records scanned, 800 kept (82%):

| Rejected | n | Why |
|---|---|---|
| Duplicate sequence | 116 | DBAASP stores one row per peptide–target pair, so a peptide tested against six organisms appears six times. Querying it naively double-counts. |
| Non-standard residue `X` | 60 | DBAASP's placeholder for an unspecified residue — no side chain, so no charge. |
| Length < 2 | 1 | |

Of the 800 kept, **427 (53%) carry a modified terminus** and 18 contain
D-residues. Terminal modifications are deliberately *not* filtered: side-chain
charge does not depend on them, and excluding them would discard most of the
database. They are recorded in the `n_terminus` / `c_terminus` columns.


## Validation

Descriptor code is checked against sources outside this repository, not against
its own output:

- **Molecular weight** matches reference compounds (glycylglycine 132.12,
  triglycine 189.17) to within 0.05 Da, is additive (joining two peptides
  loses exactly one water), and is pinned against RDKit’s `Descriptors.MolWt`
  on a five-peptide panel — values frozen into the test suite before RDKit
  was removed as a dependency.
- **Cross-checked against Biopython** (`ProtParam`): molecular weight agrees
  to 0.05 Da, GRAVY agrees exactly.
- **Charge and pI will not match Biopython or ExPASy**, by design: this code
  excludes the termini (see Charge convention above) and uses the pKa set in
  `core.PKA`. The cross-check against Biopython was run on the earlier
  termini-inclusive version, where MW agreed to 0.05 Da and GRAVY agreed
  exactly; those two are unaffected by the charge change.
- **Amphipathicity** sanity check: melittin, the textbook amphipathic peptide,
  scores above polyleucine. A moment implementation that cannot separate those
  two is broken.

## Robustness

A malformed row never kills a batch job. `process_csv` returns `(results,
errors)`; errors carry the 1-indexed CSV row, the offending input, and the
reason.

Three silent-corruption bugs were found building this, all of which produced
plausible numbers and no error:

1. **Empty CSV cells became tripeptides.** pandas reads a blank cell as `NaN`;
   `str(NaN)` is `"nan"`, which uppercases to `"NAN"` — a valid Asn-Ala-Asn.
   Blank rows were scored as real 3-mers. `validate()` now rejects non-string
   input before it becomes data, with a regression test.
2. **D-peptides were silently becoming L-peptides.** DBAASP writes D-residues
   in lowercase (`KVvvKWVvKvVK`), and `validate()` calls `.upper()` for
   convenience — so a D-peptide passed straight through as its all-L analogue:
   same formula, same mass, different molecule. Resolved by establishing that
   no descriptor here is stereochemistry-dependent, and flagging D-content in
   its own column rather than discarding it.
3. **The highest amphipathicity score in the dataset was an artifact.**
   `hydrophobic_moment` normalises by window length `min(11, len(seq))`, so a
   dipeptide is divided by 2 rather than 11. The dataset maximum, μH = 1.394,
   is the dipeptide `RW` — not a membrane-active peptide at all. See
   limitations.

## Evaluation

26 cases in `evals/questions.yaml`, each checking three things independently:

- **Routing** — were the right tools called (`expect_tools`)?
- **Negative routing** — were the wrong ones avoided (`forbid_tools`)? A
  single-sequence question that queries the whole dataset fails even if the
  number is right.
- **Correctness** — do the stated values appear, within tolerance
  (`expect_values`), and is forbidden text absent (`reject_text`)?

Ground truth comes from `peptide_agent.core`, which is itself validated above.
Six cases probe scope discipline: cyclic peptides, D-amino acids, C-terminal
amidation and invalid residues must be refused, not guessed at.

```bash
python -m evals.run_evals            # prints pass rate, writes evals/results.json
```

## Layout

```
peptide_agent/
  core.py         validation and descriptors  (pure, no I/O, no deps)
  batch.py        CSV in → (results, errors)
  tools.py        the four tools + their schemas
  agent.py        the loop, for both backends
  mcp_server.py   same tools over MCP
tests/            63 tests: reference values, scope filter, agent loop
evals/            26-case agent eval suite
scripts/          fetch_dataset.py — builds data/peptides.csv from DBAASP
app.py            Streamlit UI
chat.py           interactive terminal
demo.py           scripted questions for the demo recording
```

`core.py` has no I/O, no printing and no network, so it is testable in
isolation; every other layer is a thin wrapper over it.

## Known limitations

- pI is pKa-set dependent (see Validation).
- **Hydrophobic moment is unreliable below ~7 residues.** It is normalised by
  window length, so very short sequences are divided by a small number and
  score artificially high: the dataset maximum (μH = 1.394) is the dipeptide
  `RW`. Filter on length as well as μH when ranking candidates. Across the
  800-peptide set μH spans 0.04–1.39 with a mean of 0.59.
- **Descriptors cannot distinguish sequence order.** They are composition- and
  periodicity-driven, so `MOTIF-A-GGG-MOTIF-B` and `MOTIF-B-GGG-MOTIF-A` are
  identical to these tools and different molecules in reality. Linker-joined
  permutations collapse to the same μH; treat tied clusters as one entry when
  ranking.
- Hydrophobic moment assumes an α-helix; it is not meaningful for sequences
  that do not adopt one.
- `search_pubmed` is uncredentialed by default (3 req/s). Set `NCBI_API_KEY`
  to raise the limit.
- Descriptors are sequence-derived only — no structure prediction, no docking,
  no activity prediction.
