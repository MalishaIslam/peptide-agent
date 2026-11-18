"""Run the eval suite and report pass rate.

    python -m evals.run_evals --dataset data/peptides.csv

Each case checks three things independently: did the agent route to the right
tools, did it avoid the wrong ones, and is the answer factually right. A case
passes only if all of its stated checks pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from peptide_agent import agent, tools


def _numbers_in(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def check(case: dict, trace: agent.Trace) -> tuple[bool, list[str]]:
    failures = []
    used = set(trace.tool_sequence)
    answer = trace.answer.lower()

    if expected := case.get("expect_tools"):
        if case.get("match_any_tool"):
            if not (used & set(expected)):
                failures.append(f"used none of {expected}; used {sorted(used) or 'nothing'}")
        else:
            missing = [t for t in expected if t not in used]
            if missing:
                failures.append(f"did not call {missing}; used {sorted(used) or 'nothing'}")

    if forbidden := case.get("forbid_tools"):
        if hit := [t for t in forbidden if t in used]:
            failures.append(f"called forbidden {hit}")

    for field, want in (case.get("expect_values") or {}).items():
        tol = max(abs(want) * 0.02, 0.05)
        if not any(abs(n - want) <= tol for n in _numbers_in(trace.answer)):
            failures.append(f"{field}={want} not found in answer")

    for frag in (case.get("expect_text") or []):
        if frag.lower() not in answer:
            failures.append(f"missing text {frag!r}")

    for frag in (case.get("reject_text") or []):
        if frag.lower() in answer:
            failures.append(f"contains forbidden text {frag!r}")

    if trace.stopped_early:
        failures.append("hit turn limit without answering")

    return not failures, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/peptides.csv")
    ap.add_argument("--questions", default="evals/questions.yaml")
    ap.add_argument("--only", help="run one case by id")
    ap.add_argument("--out", default="evals/results.json")
    ap.add_argument("--backend", choices=["anthropic", "gemini"],
                    help="override the backend; default picks by API key")
    args = ap.parse_args()

    backend = agent.resolve_backend(args.backend)
    print(f"backend: {backend}")

    info = tools.load_dataset(args.dataset)
    print(f"dataset: {info['loaded']} peptides ({info['failed']} rejected)\n")

    cases = yaml.safe_load(Path(args.questions).read_text())
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]

    records, passed = [], 0
    for i, case in enumerate(cases, 1):
        trace = agent.run(case["question"], backend=backend)
        ok, failures = check(case, trace)
        passed += ok
        print(f"[{i:2d}/{len(cases)}] {'PASS' if ok else 'FAIL'}  {case['id']}")
        print(f"         tools: {' -> '.join(trace.tool_sequence) or '(none)'}")
        for f in failures:
            print(f"         !! {f}")
        records.append({"id": case["id"], "question": case["question"], "passed": ok,
                        "failures": failures, "tools": trace.tool_sequence,
                        "turns": trace.turns, "answer": trace.answer})

    rate = passed / len(cases) * 100 if cases else 0
    print(f"\n{'='*58}\n{passed}/{len(cases)} passed  ({rate:.0f}%)\n{'='*58}")

    Path(args.out).write_text(json.dumps(
        {"backend": backend, "passed": passed, "total": len(cases),
         "pass_rate": round(rate, 1), "cases": records}, indent=2))
    print(f"written to {args.out}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
