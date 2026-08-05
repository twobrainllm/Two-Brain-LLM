"""Command-line entry point.

    python -m two_brain_router                       # 3 example queries
    python -m two_brain_router --tier mobile         # route as the 1B mobile tier
    python -m two_brain_router --query "..."         # route one query
    python -m two_brain_router --query "..." --json  # machine-readable decision
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Sequence

from two_brain_router.routing import RouteDecision, TwoBrainRouter

#: The demo set: easy/local, hard/cloud, PII+hard/cloud-with-masking.
DEMO_QUERIES: list[tuple[str, str]] = [
    ("What time zone is Tokyo in?", ""),
    (
        "Derive the time complexity of merge sort step by step and compare it "
        "to quicksort's worst case, then explain the trade-offs.",
        "",
    ),
    (
        "My email is jane.doe@example.com and my phone is 555-123-4567 -- "
        "can you draft a reply telling the sender their SSN 123-45-6789 was "
        "found in an old backup and needs to be rotated?",
        "",
    ),
]


def print_decision(query: str, decision: RouteDecision) -> None:
    print(f"\n> {query}")
    print(
        f"  routed to: {decision.tier_answered} | difficulty={decision.difficulty_score:.2f} "
        f"| est_latency_ms={decision.est_latency_ms:.0f} "
        f"| est_cost_usd={decision.est_cost_usd:.5f}"
    )
    for note in decision.notes:
        print(f"  - {note}")
    print(f"  answer: {decision.answer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="two_brain_router",
        description="Privacy- and power-aware routing between a local fast brain "
        "and a Cloud AI 100 deep brain.",
    )
    parser.add_argument(
        "--tier",
        choices=["pc", "mobile"],
        default="pc",
        help="which local tier this process is running as (default: pc)",
    )
    parser.add_argument("--query", help="route a single query instead of the demo set")
    parser.add_argument("--context", default="", help="context to attach to --query")
    parser.add_argument("--json", action="store_true", help="emit the decision as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    router = TwoBrainRouter(tier=args.tier)

    queries = [(args.query, args.context)] if args.query else DEMO_QUERIES
    for query, context in queries:
        decision = router.route(query, context)
        if args.json:
            print(json.dumps(dataclasses.asdict(decision), indent=2))
        else:
            print_decision(query, decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
