"""Backend selection and loop mechanics.

The Gemini loop is exercised with a fake client, so these run offline and in
CI with no API key.
"""

import os
import pytest

from peptide_agent import agent
from peptide_agent.tools import TOOL_SCHEMAS


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "PEPTIDE_BACKEND"):
        monkeypatch.delenv(v, raising=False)


# --- backend selection ------------------------------------------------------

def test_anthropic_key_selects_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert agent.resolve_backend() == "anthropic"


def test_gemini_key_selects_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    assert agent.resolve_backend() == "gemini"


def test_google_api_key_also_selects_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "y")
    assert agent.resolve_backend() == "gemini"


def test_explicit_override_beats_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert agent.resolve_backend("gemini") == "gemini"


def test_env_override_beats_key_presence(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("PEPTIDE_BACKEND", "gemini")
    assert agent.resolve_backend() == "gemini"


def test_no_key_raises_with_actionable_message():
    with pytest.raises(RuntimeError) as e:
        agent.resolve_backend()
    assert "GEMINI_API_KEY" in str(e.value)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        agent.resolve_backend("llama")


# --- both backends see the same tools ---------------------------------------

def test_gemini_declarations_match_tool_schemas_exactly():
    """A routing difference between backends must be the model's doing, not a
    difference in what each was told."""
    decls = agent._gemini_tools()[0].function_declarations
    assert len(decls) == len(TOOL_SCHEMAS)
    for d, t in zip(decls, TOOL_SCHEMAS):
        assert d.name == t["name"]
        assert d.description == t["description"]


# --- the Gemini loop, against a fake client ---------------------------------

class _FakeCall:
    def __init__(self, name, args):
        self.name, self.args = name, args


class _FakePart:
    def __init__(self, text=None, function_call=None):
        self.text, self.function_call = text, function_call


class _FakeContent:
    def __init__(self, parts):
        self.parts, self.role = parts, "model"


class _FakeResponse:
    def __init__(self, parts):
        self.candidates = [type("C", (), {"content": _FakeContent(parts)})()]


class _FakeClient:
    """Replays a scripted sequence of model responses."""
    def __init__(self, script):
        self.script, self.calls = list(script), 0
        self.models = self

    def generate_content(self, **kw):
        r = self.script[self.calls]
        self.calls += 1
        return r


def test_gemini_loop_single_tool_then_answer():
    client = _FakeClient([
        _FakeResponse([_FakePart(function_call=_FakeCall(
            "compute_properties", {"sequence": "KRWWKWWRR"}))]),
        _FakeResponse([_FakePart(text="Net charge is +5.00 from side chains.")]),
    ])
    t = agent.run("charge of KRWWKWWRR?", client=client, backend="gemini")
    assert t.tool_sequence == ["compute_properties"]
    assert t.turns == 2
    assert "+5.00" in t.answer
    assert not t.stopped_early


def test_gemini_loop_chains_two_tools():
    client = _FakeClient([
        _FakeResponse([_FakePart(function_call=_FakeCall("dataset_summary", {}))]),
        _FakeResponse([_FakePart(function_call=_FakeCall(
            "query_dataset", {"sort_by": "hydrophobic_moment", "limit": 3}))]),
        _FakeResponse([_FakePart(text="Here are the top three.")]),
    ])
    t = agent.run("most amphipathic?", client=client, backend="gemini")
    assert t.tool_sequence == ["dataset_summary", "query_dataset"]
    assert t.turns == 3


def test_gemini_loop_answers_with_no_tools():
    client = _FakeClient([_FakeResponse([_FakePart(text="GRAVY means hydropathy.")])])
    t = agent.run("what does GRAVY stand for?", client=client, backend="gemini")
    assert t.tool_sequence == []
    assert t.turns == 1


def test_gemini_loop_respects_turn_limit():
    """A model that never stops calling tools must be cut off, not loop."""
    looping = _FakeResponse([_FakePart(function_call=_FakeCall(
        "dataset_summary", {}))])
    client = _FakeClient([looping] * 10)
    t = agent.run("loop forever", client=client, backend="gemini", max_turns=3)
    assert t.stopped_early
    assert t.turns == 3


def test_unknown_tool_is_reported_not_raised():
    """A hallucinated tool name must come back as data, so the model can
    recover, rather than killing the loop."""
    client = _FakeClient([
        _FakeResponse([_FakePart(function_call=_FakeCall("predict_activity", {}))]),
        _FakeResponse([_FakePart(text="That tool does not exist.")]),
    ])
    t = agent.run("predict activity", client=client, backend="gemini")
    assert t.tool_calls[0]["ok"] is False
    assert not t.stopped_early


def test_bad_arguments_are_reported_not_raised():
    client = _FakeClient([
        _FakeResponse([_FakePart(function_call=_FakeCall(
            "compute_properties", {"wrong_arg": "GG"}))]),
        _FakeResponse([_FakePart(text="I passed the wrong argument.")]),
    ])
    t = agent.run("x", client=client, backend="gemini")
    assert t.tool_calls[0]["ok"] is False
    assert not t.stopped_early


def test_trace_records_backend():
    client = _FakeClient([_FakeResponse([_FakePart(text="hi")])])
    assert agent.run("x", client=client, backend="gemini").backend == "gemini"


# --- transient-failure retry ------------------------------------------------

class _Boom(Exception):
    pass


def test_retry_recovers_from_transient_error(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Boom("503 UNAVAILABLE: model is experiencing high demand")
        return "ok"

    assert agent._with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_retry_does_not_mask_a_bad_key(monkeypatch):
    """A 401 is not transient. Retrying it wastes time and hides the cause."""
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def unauthorized():
        calls["n"] += 1
        raise _Boom("401 UNAUTHENTICATED: invalid api key")

    with pytest.raises(_Boom):
        agent._with_retry(unauthorized)
    assert calls["n"] == 1


def test_retry_gives_up_and_raises(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_busy():
        calls["n"] += 1
        raise _Boom("503 UNAVAILABLE")

    with pytest.raises(_Boom):
        agent._with_retry(always_busy)
    assert calls["n"] == agent.MAX_RETRIES


def test_unknown_model_404_is_not_retried(monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def gone():
        calls["n"] += 1
        raise _Boom("404 NOT_FOUND: model no longer available")

    with pytest.raises(_Boom):
        agent._with_retry(gone)
    assert calls["n"] == 1
