"""Interactive terminal chat with the agent.

    python chat.py

Type a question, watch which tools it calls, read the answer. Ctrl-C or
"quit" to exit. Each question starts a fresh conversation -- this is a tool
for watching routing decisions, not a chatbot with memory.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from peptide_agent import agent, tools

EXAMPLES = [
    "What is the net charge of KRWWKWWRR?",
    "Which peptides in my dataset are strongly amphipathic?",
    "Find my most cationic peptide, then search PubMed for related work.",
    "Compare KRWWKWWRR and DDDDEEEE on charge and hydrophobicity.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/peptides.csv")
    args = ap.parse_args()

    try:
        print(f"Backend: {agent.resolve_backend()}")
    except RuntimeError as e:
        print(e)
        return 1

    try:
        info = tools.load_dataset(args.dataset)
        print(f"Loaded {info['loaded']} peptides from {args.dataset} "
              f"({info['failed']} rejected).")
    except FileNotFoundError:
        print(f"No dataset at {args.dataset}. Dataset questions will fail; "
              "single-sequence ones still work.\n"
              "Run: python scripts/fetch_dataset.py --limit 800")

    print("\nTry one of these, or ask your own:")
    for e in EXAMPLES:
        print(f"  • {e}")
    print("\nType 'quit' to exit.\n")

    while True:
        try:
            q = input("\033[1myou >\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            print("bye")
            return 0

        print()
        try:
            trace = agent.run(q, verbose=True)
        except Exception as e:
            print(f"  error: {type(e).__name__}: {e}\n")
            continue

        print()
        for line in trace.answer.splitlines():
            print(textwrap.fill(line, 78) if line.strip() else "")
        print(f"\n  \033[2mroute: {' → '.join(trace.tool_sequence) or '(no tools)'}"
              f"   |   {trace.turns} turn(s)\033[0m\n")


if __name__ == "__main__":
    sys.exit(main())
