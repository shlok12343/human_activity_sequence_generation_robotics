"""
CLI entrypoint for the soda_can_problem prototype.

Run from the repository root (so ``state_graph_generator`` is importable)::

    export GOOGLE_API_KEY=your_key
    python -m soda_can_problem.cli

Use ``--help`` for flags (output dir, combination cap, cache refresh).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from soda_can_problem.defaults import (
    DEFAULT_GEMINI_TIMEOUT_SECONDS,
    DEFAULT_HAZARD_BATCH_SIZE,
    DEFAULT_HAZARD_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_AFFORDANCE_RULES,
    DEFAULT_ORIGINAL_OBJECTS,
)
from soda_can_problem.pipeline import run_all


def _parse_objects(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [x.strip() for x in arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Open-world hazard prototype: Gemini interaction discovery, "
            "reuse of run_state_graph_pipeline per interaction target, "
            "and batched hazard judgments."
        ),
        epilog=(
            "Requires GOOGLE_API_KEY and a cwd/repo root on PYTHONPATH.\n"
            "Example:\n"
            "  cd /path/to/alternate_sequences_project\n"
            "  export GOOGLE_API_KEY=...\n"
            "  python -m soda_can_problem.cli --max-combinations 50\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--objects",
        type=str,
        default=None,
        help=(
            "Comma-separated original object slugs (default: soda_can,plastic_container,"
            "metal_fork,knife,towel)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON artifacts (default: soda_can_problem/artifacts/).",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Cap combinations before hazard judging (demo / cost control).",
    )
    parser.add_argument(
        "--skip-graph-cache",
        action="store_true",
        help="Rebuild state graphs instead of reusing cached JSON under graphs/.",
    )
    parser.add_argument(
        "--no-gemini-fallback",
        action="store_true",
        help="Disable Gemini-proposed sequences for unknown interaction targets.",
    )
    parser.add_argument(
        "--hazard-batch-size",
        type=int,
        default=DEFAULT_HAZARD_BATCH_SIZE,
        help=f"Hazard judgments per Gemini call (default {DEFAULT_HAZARD_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num-affordance-rules",
        type=int,
        default=DEFAULT_NUM_AFFORDANCE_RULES,
        help=f"Forwarded to run_state_graph_pipeline (default {DEFAULT_NUM_AFFORDANCE_RULES}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            f"Gemini model for steps 2-3 (discovery + state graphs; "
            f"default {DEFAULT_MODEL_NAME})."
        ),
    )
    parser.add_argument(
        "--hazard-model",
        type=str,
        default=DEFAULT_HAZARD_MODEL_NAME,
        help=(
            f"Gemini model for step 5 hazard batches only "
            f"(default {DEFAULT_HAZARD_MODEL_NAME})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_GEMINI_TIMEOUT_SECONDS,
        help=(
            f"Per-call timeout in seconds for step 5 hazard API calls "
            f"(default {DEFAULT_GEMINI_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--fresh-hazard",
        action="store_true",
        help=(
            "Ignore step5_checkpoint.json and re-run all hazard batches from scratch."
        ),
    )
    args = parser.parse_args()

    originals = _parse_objects(args.objects)
    if originals is None:
        originals = list(DEFAULT_ORIGINAL_OBJECTS)

    run_all(
        original_objects=originals,
        output_dir=args.output_dir,
        max_combinations=args.max_combinations,
        skip_graph_cache=args.skip_graph_cache,
        use_gemini_fallback=not args.no_gemini_fallback,
        hazard_batch_size=args.hazard_batch_size,
        num_affordance_rules=args.num_affordance_rules,
        model_name=args.model,
        hazard_model_name=args.hazard_model,
        timeout_seconds=args.timeout,
        fresh_hazard=args.fresh_hazard,
    )


if __name__ == "__main__":
    main()
