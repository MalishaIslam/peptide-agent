"""Scripted demo: runs a fixed set of questions and prints each trace.

    python demo.py                 # all questions
    python demo.py --only 2        # just the third one

Use this to record the README GIF -- one command, no typing in a browser.
The questions are chosen to show three different routes through the tools:
a single call, a plan-then-filter, and a cross-source chain.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from peptide_agent import agent, tools

QUESTIONS = [
    "What is the net charge and molecular weight of GLFDIVKKVVGALGSL?",
    "Which peptides in my dataset are strongly amphipathic? "
    "Pick a sensible threshold and say why you chose it.",
    "Find the most positively charged peptide in my dataset, "
    "then search PubMed for work on cationic antimicrobial peptides.",
    "What is the molecular weight of cyclic GLFDIVKK with a head-to-tail bond?",
]

RULE = "─" * 70


def show(i: int, question: str) -> None:
    print(f"\n{RULE}\n[{i + 1}] {question}\n{RULE}")
    trace = agent.run(question, verbose=True)
    print()
    for line in trace.answer.splitlines():
        print(textwrap.fill(line, 78) if line.strip() else "")
    print(f"\n   tools: {' -> '.join(trace.tool_sequence) or '(none)'}"
          f"   |   turns: {trace.turns}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/peptides.csv")
    ap.add_argument("--only", type=int, help="run a single question by index (0-based)")
    args = ap.parse_args()

    try:
        info = tools.load_dataset(args.dataset)
    except FileNotFoundError:
        print(f"no dataset at {args.dataset} -- run scripts/fetch_dataset.py first",
              file=sys.stderr)
        return 1
    print(f"dataset: {info['loaded']} peptides loaded, {info['failed']} rejected")

    items = (
        [(args.only, QUESTIONS[args.only])] if args.only is not None
        else list(enumerate(QUESTIONS))
    )
    for i, q in items:
        show(i, q)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
