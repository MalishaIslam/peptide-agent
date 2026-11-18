"""Streamlit UI. Deployable publicly at zero cost to the author.

    streamlit run app.py

Three tabs. The first two are pure computation and need no API key, so a
visitor can use them immediately. The third runs the agent and asks the
visitor for their own Anthropic key, which is never stored and never leaves
their session -- so a public deployment costs the author nothing.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from peptide_agent.batch import process_frame
from peptide_agent.core import PeptideError, compute_all

DATASET = Path("data/peptides.csv")

st.set_page_config(page_title="Peptide Agent", page_icon="🧬", layout="wide")
st.title("🧬 Peptide Analysis Agent")
st.caption(
    "Linear peptides, 20 standard amino acids, L or D. Net charge counts "
    "ionizable side chains only — termini excluded — so it reads about "
    "1 unit lower than ExPASy. Research tool, not for clinical use."
)


@st.cache_data(show_spinner="Fetching peptides from DBAASP…")
def load_dataset_cached(limit: int = 400) -> pd.DataFrame:
    """Use the committed CSV if present, else fetch from DBAASP once."""
    if DATASET.exists():
        return pd.read_csv(DATASET)
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "scripts/fetch_dataset.py", "--limit", str(limit)],
        check=True, capture_output=True, timeout=600,
    )
    return pd.read_csv(DATASET)


single, batch, ask = st.tabs(
    ["Single peptide", "Batch CSV", "Ask the agent (needs your API key)"]
)


with single:
    seq = st.text_input("Sequence", "GLFDIVKKVVGALGSL").strip()
    if seq:
        try:
            p = compute_all(seq)
        except PeptideError as e:
            st.error(str(e))
        else:
            a, b, c, d = st.columns(4)
            a.metric("Length", p.length)
            b.metric("MW (Da)", f"{p.mol_weight:,.1f}",
                     help="Residue masses + one water. Low by ~1 Da if a terminus is capped.")
            c.metric("Net charge (pH 7)", f"{p.net_charge:+.2f}",
                     help="Ionizable side chains only; termini excluded.")
            d.metric("pI", "none" if p.isoelectric_point is None
                     else f"{p.isoelectric_point:.2f}",
                     help="'none' means the charge curve never crosses zero "
                          "under this convention — not a missing value.")
            a, b, c, d = st.columns(4)
            a.metric("GRAVY", f"{p.gravy:+.3f}", help="Mean Kyte-Doolittle hydropathy.")
            b.metric("Hydrophobic moment", f"{p.hydrophobic_moment:.3f}",
                     help="Eisenberg μH, 100° helical periodicity. Higher = more amphipathic.")
            c.metric("Aromatic", f"{p.aromaticity:.1%}")
            d.metric("Charged", f"{p.fraction_charged:.1%}")


with batch:
    st.write("Upload a CSV with a sequence column, or browse the DBAASP set.")
    up = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")

    source = None
    if up is not None:
        source = pd.read_csv(up)
    elif st.button("Load DBAASP sample"):
        source = load_dataset_cached()

    if source is not None:
        try:
            results, errors = process_frame(source)
        except PeptideError as e:
            st.error(str(e))
        else:
            a, b = st.columns(2)
            a.metric("Processed", len(results))
            b.metric("Rejected", len(errors))

            if not results.empty:
                st.dataframe(results, use_container_width=True)
                buf = io.StringIO()
                results.to_csv(buf, index=False)
                st.download_button("Download results CSV", buf.getvalue(),
                                   "peptide_properties.csv", "text/csv")
            if not errors.empty:
                st.warning(f"{len(errors)} row(s) could not be processed — "
                           "each with the row number and reason:")
                st.dataframe(errors, use_container_width=True)


with ask:
    st.info(
        "This tab runs the agent, which needs a model API. It uses **your** "
        "key, entered below — held only for this browser session, never "
        "written to disk, never sent anywhere but the provider you pick. "
        "**Gemini keys are free** at aistudio.google.com. "
        "The other two tabs need no key at all.",
        icon="🔑",
    )
    provider = st.radio("Provider", ["Gemini (free tier)", "Anthropic"],
                        horizontal=True)
    is_gemini = provider.startswith("Gemini")
    key = st.text_input(
        "Gemini API key" if is_gemini else "Anthropic API key",
        type="password",
        placeholder="AIza…" if is_gemini else "sk-ant-api03-…",
    )

    q = st.text_area(
        "Question",
        "Which peptides in the dataset are strongly amphipathic? "
        "Pick a sensible threshold and say why.",
        height=90,
    )

    if st.button("Run", type="primary", disabled=not key):
        backend = "gemini" if is_gemini else "anthropic"
        os.environ["GEMINI_API_KEY" if is_gemini else "ANTHROPIC_API_KEY"] = key
        from peptide_agent import agent, tools
        if not DATASET.exists():
            load_dataset_cached()
        tools.load_dataset(str(DATASET))

        with st.spinner("Thinking…"):
            try:
                trace = agent.run(q, backend=backend)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                trace = None

        if trace:
            st.markdown(trace.answer)
            route = " → ".join(trace.tool_sequence) or "(no tools)"
            st.caption(f"Route: {route}  ·  {trace.turns} turn(s)  ·  {trace.backend}")
            with st.expander(f"Tool calls ({len(trace.tool_calls)})"):
                for i, call in enumerate(trace.tool_calls, 1):
                    st.code(f"{i}. {call['name']}({call['input']})", language="python")
    elif not key:
        st.caption("Enter a key above to enable this tab.")
