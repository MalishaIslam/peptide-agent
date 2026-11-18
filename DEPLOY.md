# Deploying the demo (free)

The Streamlit app is built so a public deployment costs the author nothing:

- **Single peptide** and **Batch CSV** are pure Python. No API key, no model,
  no cost. Any visitor can use them immediately.
- **Ask the agent** asks the visitor for their own Anthropic key. It is held
  in session memory only — never written to disk, never logged, never sent
  anywhere but Anthropic.

So the live link works for everyone, and nobody can spend your money.

## Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Sign in at share.streamlit.io with GitHub.
3. New app → pick the repo → main file `app.py` → Deploy.

First load fetches ~400 peptides from DBAASP and caches them, so it is slow
once and fast after. No secrets to configure.

## Hugging Face Spaces

New Space → SDK: Streamlit → push this repo. Same behaviour.

## Note on the dataset

`data/*.csv` is gitignored, so the deployed app has no dataset at first. It
runs `scripts/fetch_dataset.py` on demand and caches the result. This keeps
DBAASP's data out of the repository rather than republishing it.

If a deploy host blocks outbound HTTP, the fetch fails and the dataset tabs
go quiet while the single-peptide tab keeps working. In that case commit a
small sample CSV and check DBAASP's terms for attribution first.

## What a visitor sees without a key

Everything except the agent tab, which shows a short note explaining why a
key is needed. Put a GIF of the agent running in the README so the two-step
routing is visible to people who will not paste a key — most visitors won't.
