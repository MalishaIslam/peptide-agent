"""MCP server exposing the peptide tools to any MCP client (Claude Desktop).

Run standalone:        python -m peptide_agent.mcp_server

Claude Desktop config (macOS:
~/Library/Application Support/Claude/claude_desktop_config.json):

    {"mcpServers": {"peptide": {
        "command": "/abs/path/to/.venv/bin/python",
        "args": ["-m", "peptide_agent.mcp_server"],
        "cwd": "/abs/path/to/peptide-agent",
        "env": {"PYTHONPATH": "/abs/path/to/peptide-agent",
                "PEPTIDE_DATASET": "/abs/path/to/data/peptides.csv"}}}}

PYTHONPATH is required. Claude Desktop does not put cwd on the import path,
so without it the server exits with ModuleNotFoundError before the client
ever connects.

NOTE ON DOCSTRINGS: in this mode the client supplies its own system prompt,
so agent.SYSTEM_PROMPT is never seen. These docstrings are the ONLY
instructions that reach the model -- every rule that matters has to live
here. Keep them in sync with tools.TOOL_SCHEMAS, which carries the same
rules for the API path.
"""

from __future__ import annotations

import os
import sys

try:                                    # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ModuleNotFoundError:             # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import tools

INSTRUCTIONS = """Tools for analysing linear peptides of the 20 standard
amino acids.

Rules:
- Never state a descriptor value you did not get from a tool. Do not
  estimate or recall them.
- Net charge counts ionizable SIDE CHAINS ONLY; the termini are excluded.
  Say so whenever you report a charge, and note it will not match ExPASy
  or Biopython.
- isoelectric_point is null when the charge curve never crosses zero.
  Report that as "no pI under this convention", not as zero or missing.
- For questions about "my peptides" or "the dataset", use query_dataset
  rather than computing sequences one at a time.
- For a vague threshold ("strongly amphipathic"), call dataset_summary
  first, then choose a cutoff and justify it.
- Cyclic peptides, non-standard residues and structural or 3D questions
  are out of scope. Say so rather than guessing.
- Cite only literature that search_pubmed actually returned.
- This is a research tool, not a clinical or safety assessment."""

try:
    mcp = _Server("peptide-analysis", instructions=INSTRUCTIONS)
except TypeError:                       # older signature without instructions
    mcp = _Server("peptide-analysis")

# A missing or unreadable dataset must not kill the server: four of the five
# tools do not need one, and a dead server loses all of them. Report the
# problem on stderr (which the client logs) and carry on.
if path := os.environ.get("PEPTIDE_DATASET"):
    try:
        _info = tools.load_dataset(path)
        print(f"loaded {_info['loaded']} peptides from {path} "
              f"({_info['failed']} rejected)", file=sys.stderr)
    except FileNotFoundError:
        print(f"WARNING: no dataset at {path}. Dataset tools will report this; "
              "single-sequence tools still work. Run scripts/fetch_dataset.py.",
              file=sys.stderr)
    except Exception as e:                      # malformed CSV, bad columns
        print(f"WARNING: could not load {path}: {type(e).__name__}: {e}",
              file=sys.stderr)
else:
    print("No PEPTIDE_DATASET set; dataset tools need load_dataset first.",
          file=sys.stderr)


@mcp.tool()
def compute_properties(sequence: str) -> dict:
    """Compute physicochemical descriptors for ONE linear peptide.

    Input: a one-letter amino acid sequence of the 20 standard residues,
    e.g. GLFDIVKKVVGALGSL. L- and D-residues give identical values here, so
    case does not matter.

    Returns length, molecular weight, net charge at pH 7, GRAVY
    hydrophobicity, hydrophobic moment (amphipathicity), isoelectric point,
    aromaticity and fraction of charged residues.

    IMPORTANT, state these when you report the numbers:
    - net_charge counts IONIZABLE SIDE CHAINS ONLY. The N- and C-terminal
      groups are excluded, so the value is comparable across free, amidated
      and acylated peptides but is about 1 unit lower at pH 7 than the
      conventional figure. It will not match ExPASy or Biopython.
    - isoelectric_point is null when the charge curve never crosses zero
      (an all-basic peptide, an all-acidic one, or one with no ionizable
      side chains). Report that as "no pI exists under this convention",
      never as zero and never as a missing value.
    - mol_weight is the sum of residue masses plus one water, so for a
      peptide with a capped terminus it describes the uncapped analogue.

    Never state a descriptor value you did not get from this tool. Do not
    estimate or recall them. Cyclic peptides, non-standard residues and
    structural questions are out of scope: say so rather than guessing.
    """
    return tools.compute_properties(sequence)


@mcp.tool()
def query_dataset(filters: list[dict] | None = None, sort_by: str | None = None,
                  descending: bool = True, limit: int = 20) -> dict:
    """Filter and sort the loaded peptide dataset by any computed descriptor.

    Use this for any question about "my peptides", "the dataset" or "my
    data" -- do not compute sequences one at a time for those.

    filters is a list of AND-combined conditions:
        [{"field": "net_charge", "op": ">", "value": 4},
         {"field": "gravy", "op": "<", "value": 0}]

    Fields: length, mol_weight, net_charge, gravy, hydrophobic_moment,
    isoelectric_point, aromaticity, fraction_charged.
    Operators: > >= < <= == !=

    Call with no filters to see the whole set. If you are unsure what
    threshold is reasonable for a vague word like "high" or "strongly",
    call dataset_summary FIRST to see the actual range, then choose a
    threshold and say why you chose it.
    """
    return tools.query_dataset(filters, sort_by, descending, limit)


@mcp.tool()
def dataset_summary() -> dict:
    """Size of the loaded dataset and the min, max and mean of every
    descriptor.

    Call this before query_dataset whenever the question uses a relative
    term ("strongly amphipathic", "highly cationic", "unusually long") and
    you need a defensible numeric cutoff.
    """
    return tools.dataset_summary()


@mcp.tool()
def search_pubmed(query: str, max_results: int = 5) -> dict:
    """Search PubMed for published literature on a peptide or an activity.

    Returns titles, journals, years, PMIDs and links. Cite only what this
    returns -- never invent a citation, an author or a PMID. If the search
    returns nothing, say so plainly.
    """
    return tools.search_pubmed(query, max_results)


@mcp.tool()
def load_dataset(path: str) -> dict:
    """Load a peptide CSV and compute descriptors for every row.

    Only needed if no dataset was loaded at startup, or to switch files.
    Returns how many rows loaded, how many were rejected, and why.
    """
    return tools.load_dataset(path)


if __name__ == "__main__":
    mcp.run()
