# Model backends

The agent runs on either Anthropic or Google Gemini. Both are given
**byte-identical** tool descriptions, read from `tools.TOOL_SCHEMAS`, so any
difference in how they route is a difference in the model, not in what it was
told. A test pins that (`test_gemini_declarations_match_tool_schemas_exactly`).

## Getting a free Gemini key

1. aistudio.google.com → "Get API key" → create one.
2. It starts with `AIza`. A string starting `AQ.` is an OAuth token, not an
   API key — wrong screen.
3. No billing setup needed.

Google's free tier states that content is used to improve their products.
Fine for DBAASP, which is public. Do not point the free tier at unpublished
sequences.

## Running

```bash
export GEMINI_API_KEY=...        # or ANTHROPIC_API_KEY
python chat.py                   # prints which backend it picked
python -m evals.run_evals        # pass rate, written to evals/results.json
```

Selection order:

| | |
|---|---|
| `PEPTIDE_BACKEND=gemini` | explicit override, wins over everything |
| `ANTHROPIC_API_KEY` set | → anthropic |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` set | → gemini |
| neither | `RuntimeError` naming both |

Models default to `claude-sonnet-4-5` and `gemini-3.8-flash`; override with
`ANTHROPIC_MODEL` / `GEMINI_MODEL`.

Model names get retired. If a call returns `404 ... no longer available`,
the error names the replacement — set `GEMINI_MODEL` to it and, if it keeps
working, change the default in `agent.py`. Nothing else in the loop depends
on the model name.

## Comparing the two

```bash
python -m evals.run_evals --backend gemini    --out evals/results-gemini.json
python -m evals.run_evals --backend anthropic --out evals/results-anthropic.json
```

The pass rate is only the headline. The interesting column is which tools each
model chose: a model that answers "which are strongly amphipathic?" without
calling `dataset_summary` first has guessed a threshold instead of deriving
one, and the eval marks that as a routing failure even when the final list
looks plausible.

## Adding another backend

`agent.py` is the only file that talks to a model. A new backend is one
function with the same shape as `_run_gemini`: send messages plus
`TOOL_SCHEMAS`, check whether the reply contains tool calls, run them through
`_dispatch`, append the results, repeat. Then add it to `resolve_backend`.
