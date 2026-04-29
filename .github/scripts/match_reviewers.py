#!/usr/bin/env python3
"""Match changed files against merge_rules.yaml and output reviewers to request.

Reuses patterns_to_regex from gitutils.py to ensure matching semantics are
identical to trymerge.

Usage:
    python match_reviewers.py --merge-rules PATH --changed-files f1 f2 ... --pr-author LOGIN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gitutils import patterns_to_regex


def is_wildcard_only(patterns: list[str]) -> bool:
    return all(p in ("*", "**") for p in patterns)


KNOWN_BOTS = frozenset(
    {
        "pytorchbot",
        "pytorchmergebot",
        "facebook-github-bot",
    }
)


def is_bot(username: str) -> bool:
    return username.lower() in KNOWN_BOTS


def match_reviewers(
    rules: list[dict],
    changed_files: list[str],
    pr_author: str,
) -> tuple[list[str], list[str]]:
    """Return (reviewers, team_slugs) matched from merge rules."""
    reviewers: set[str] = set()
    teams: set[str] = set()
    pr_author_lower = pr_author.lower()

    for rule in rules:
        patterns = rule.get("patterns", [])
        approvers = rule.get("approved_by", [])

        if is_wildcard_only(patterns):
            continue

        regex = patterns_to_regex(patterns)
        matched = any(regex.match(f) for f in changed_files)

        if matched:
            for approver in approvers:
                if is_bot(approver):
                    continue
                if approver.startswith("pytorch/"):
                    teams.add(approver[len("pytorch/") :])
                elif approver.lower() != pr_author_lower:
                    reviewers.add(approver)

    return sorted(reviewers), sorted(teams)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-rules", required=True, type=Path)
    parser.add_argument("--changed-files", nargs="*", default=[])
    parser.add_argument(
        "--changed-files-stdin",
        action="store_true",
        help="Read changed files as JSON array from stdin",
    )
    parser.add_argument("--pr-author", required=True)
    args = parser.parse_args()

    import yaml

    with open(args.merge_rules) as f:
        rules = yaml.safe_load(f)

    if args.changed_files_stdin:
        changed_files = json.load(sys.stdin)
    else:
        changed_files = args.changed_files

    reviewers, teams = match_reviewers(rules, changed_files, args.pr_author)

    result = {"reviewers": reviewers, "teams": teams}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
