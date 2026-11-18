"""The agent loop.

Deliberately hand-written rather than framework-driven: the whole mechanism
is a few dozen lines per backend, and it is the thing worth being able to
explain.

The loop, for either backend: send the conversation and the list of
available tools to the model -> if it asks for a tool, run it and append the
result -> repeat -> stop when it answers in prose instead.

Two backends are supported so the same tools can be benchmarked across
providers. Both read the SAME tool schemas from tools.TOOL_SCHEMAS, so a
routing difference between them is a difference in the model, not in what it
was told.

Backend selection, in order:
    PEPTIDE_BACKEND=anthropic|gemini   explicit override
    ANTHROPIC_API_KEY set              -> anthropic
    GEMINI_API_KEY / GOOGLE_API_KEY    -> gemini
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field

from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MAX_TURNS = 10

# Free tiers return 429 (rate limited) and 503 (busy) routinely. A 26-case
# eval run makes dozens of calls, so one transient failure must not kill it.
MAX_RETRIES = 5
RETRYABLE = ("429", "503", "500", "overloaded", "unavailable", "high demand",
             "rate limit", "resource_exhausted", "deadline")


def _with_retry(call, label: str = "", verbose: bool = False):
    """Run call(), retrying transient provider errors with exponential
    backoff and jitter. Anything that is not obviously transient -- a bad
    key, an unknown model, a malformed request -- is raised immediately,
    because retrying those just wastes time and hides the real problem.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except Exception as e:
            text = f"{type(e).__name__} {e}".lower()
            if not any(k in text for k in RETRYABLE) or attempt == MAX_RETRIES - 1:
                raise
            wait = min(2 ** attempt, 16) + random.uniform(0, 1)
            if verbose:
                print(f"  {label}transient error, retrying in {wait:.1f}s "
                      f"({attempt + 1}/{MAX_RETRIES - 1})")
            time.sleep(wait)

SYSTEM_PROMPT = """You are a peptide analysis assistant.

You have tools for computing physicochemical descriptors, querying a loaded
peptide dataset, and searching PubMed. Decide which to use and in what order.

Rules:
- Never state a descriptor value you did not get from a tool. Do not estimate.
- For questions about "my peptides" or "the dataset", query the dataset rather
  than computing sequences one at a time.
- If you are unsure what thresholds are reasonable, call dataset_summary first,
  then pick a cutoff and say why you picked it.
- net_charge counts ionizable SIDE CHAINS ONLY; the N- and C-terminal groups
  are excluded. Say so when you report a charge, and note it reads about
  1 unit lower than ExPASy or Biopython.
- isoelectric_point is null when the charge curve never crosses zero. Report
  that as "no pI under this convention", never as zero or as missing.
- Scope is linear peptides of the 20 standard amino acids, L or D (they give
  identical values here). If asked about cyclic peptides, non-standard
  residues or 3D structure, say plainly that it is out of scope.
- Cite only literature that search_pubmed actually returned.
- Report numbers with units and say which tool produced them.
- This is a research tool, not a clinical or safety assessment."""


@dataclass
class Trace:
    """What the agent did, for evaluation and for the demo."""
    question: str
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    turns: int = 0
    stopped_early: bool = False
    backend: str = ""

    @property
    def tool_sequence(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


def resolve_backend(name: str | None = None) -> str:
    """Pick a backend from an explicit name or the environment."""
    choice = (name or os.environ.get("PEPTIDE_BACKEND") or "").strip().lower()
    if choice in ("anthropic", "gemini"):
        return choice
    if choice:
        raise ValueError(f"unknown backend {choice!r}; use 'anthropic' or 'gemini'")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    raise RuntimeError(
        "No API key found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY "
        "(a Gemini key is free: aistudio.google.com)."
    )


def _dispatch(name: str, args: dict, trace: Trace, verbose: bool) -> dict:
    """Run one tool call and record it. Never raises: a tool failure is data
    the model can react to, not a reason to kill the loop."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        out = {"ok": False, "error": f"unknown tool {name!r}"}
    else:
        try:
            out = fn(**args)
        except Exception as e:                      # bad arguments from the model
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    trace.tool_calls.append({"name": name, "input": args,
                             "ok": out.get("ok", True)})
    if verbose:
        print(f"  [{trace.turns}] {name}({json.dumps(args, default=str)[:90]})")
    return out


# --- Anthropic --------------------------------------------------------------

def _run_anthropic(question: str, trace: Trace, max_turns: int,
                   verbose: bool, client=None) -> Trace:
    import anthropic

    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    for turn in range(max_turns):
        trace.turns = turn + 1
        response = _with_retry(
            lambda: client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=2048, system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS, messages=messages,
            ), verbose=verbose)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            trace.answer = "".join(
                b.text for b in response.content if b.type == "text"
            ).strip()
            return trace

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            out = _dispatch(block.name, block.input, trace, verbose)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})

    trace.stopped_early = True
    trace.answer = f"Stopped after {max_turns} turns without reaching an answer."
    return trace


# --- Gemini -----------------------------------------------------------------

def _gemini_tools():
    """Reuse TOOL_SCHEMAS unchanged via parameters_json_schema, so both
    backends are given byte-identical tool descriptions."""
    from google.genai import types

    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters_json_schema=t["input_schema"],
        ) for t in TOOL_SCHEMAS
    ])]


def _run_gemini(question: str, trace: Trace, max_turns: int,
                verbose: bool, client=None) -> Trace:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = client or genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, tools=_gemini_tools(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True       # we drive the loop ourselves, as with Anthropic
        ),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    for turn in range(max_turns):
        trace.turns = turn + 1
        response = _with_retry(
            lambda: client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=config,
            ), verbose=verbose)

        candidate = (response.candidates or [None])[0]
        parts = (candidate.content.parts if candidate and candidate.content
                 else []) or []
        calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

        if not calls:
            trace.answer = "".join(
                p.text for p in parts if getattr(p, "text", None)
            ).strip()
            return trace

        contents.append(candidate.content)
        replies = []
        for call in calls:
            out = _dispatch(call.name, dict(call.args or {}), trace, verbose)
            replies.append(types.Part.from_function_response(
                name=call.name, response={"result": out},
            ))
        contents.append(types.Content(role="user", parts=replies))

    trace.stopped_early = True
    trace.answer = f"Stopped after {max_turns} turns without reaching an answer."
    return trace


# --- public entry point -----------------------------------------------------

def run(question: str, client=None, max_turns: int = MAX_TURNS,
        verbose: bool = False, backend: str | None = None) -> Trace:
    """Answer one question, letting the model choose and sequence the tools."""
    name = resolve_backend(backend)
    trace = Trace(question=question, backend=name)
    runner = _run_anthropic if name == "anthropic" else _run_gemini
    return runner(question, trace, max_turns, verbose, client)
