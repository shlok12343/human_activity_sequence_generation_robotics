"""Orchestrate discovery -> state graphs (originals only) -> combinations -> hazard judging."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from soda_can_problem.combinations import build_combinations
from soda_can_problem.defaults import (
    DEFAULT_GEMINI_TIMEOUT_SECONDS,
    DEFAULT_HAZARD_BATCH_SIZE,
    DEFAULT_HAZARD_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_ORIGINAL_OBJECTS,
    DEFAULT_NUM_AFFORDANCE_RULES,
)
from soda_can_problem.graph_cache import ResolvedGraph, ensure_graph_for_interaction
from soda_can_problem.hazard_evaluation import evaluate_all_combinations
from soda_can_problem.interaction_discovery import discover_interactions
from soda_can_problem.schemas import GraphBuildRecord


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def run_all(
    original_objects: Optional[List[str]] = None,
    *,
    output_dir: Optional[Path] = None,
    max_combinations: Optional[int] = None,
    skip_graph_cache: bool = False,
    use_gemini_fallback: bool = True,
    hazard_batch_size: int = DEFAULT_HAZARD_BATCH_SIZE,
    num_affordance_rules: int = DEFAULT_NUM_AFFORDANCE_RULES,
    model_name: str = DEFAULT_MODEL_NAME,
    hazard_model_name: str = DEFAULT_HAZARD_MODEL_NAME,
    timeout_seconds: float = DEFAULT_GEMINI_TIMEOUT_SECONDS,
    fresh_hazard: bool = False,
) -> Path:
    """
    Run steps 2-5 and write JSON under ``output_dir``.

    State graphs are built only for the original objects. Interaction targets
    from step 2 remain plain strings used in the combination tuples.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export it or add it to .env before running.",
        )

    originals = list(original_objects or DEFAULT_ORIGINAL_OBJECTS)
    out = output_dir or (_package_dir() / "artifacts")
    out.mkdir(parents=True, exist_ok=True)

    # -- Step 2: discover interaction targets per original object ---------------
    print("Step 2: discovering interactions via Gemini...")
    interactions_by_object = discover_interactions(originals, model_name=model_name)
    step2_path = out / "step2_interactions.json"
    with open(step2_path, "w", encoding="utf-8") as f:
        json.dump(interactions_by_object, f, indent=2, sort_keys=True)
    print(f"  wrote {step2_path}")

    # -- Step 3: build state graphs for the originals only ---------------------
    print(
        f"Step 3: building/caching state graphs for {len(originals)} "
        "original objects..."
    )
    graphs_by_original: Dict[str, ResolvedGraph] = {}
    graph_report: List[Dict[str, Any]] = []
    for obj in originals:
        resolved = ensure_graph_for_interaction(
            obj,
            output_dir=out,
            skip_cache=skip_graph_cache,
            use_gemini_fallback=use_gemini_fallback,
            num_affordance_rules=num_affordance_rules,
            model_name=model_name,
        )
        graphs_by_original[obj] = resolved
        graph_report.append(
            GraphBuildRecord(
                interaction_raw=resolved.interaction_raw,
                normalized_key=resolved.normalized_key,
                slug=resolved.slug,
                status=resolved.status,
                reason=resolved.reason,
                target_object=resolved.target_object,
                graph_path=resolved.graph_path,
                validation_errors=resolved.validation_errors,
                node_count=resolved.node_count,
            ).model_dump()
        )
        print(f"  {obj!r}: {resolved.status} ({resolved.node_count} nodes)")

    report_path = out / "graph_build_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(graph_report, f, indent=2)
    print(f"  wrote {report_path}")

    # -- Step 4: expand combinations (original x state x interaction_target) ---
    print("Step 4: expanding combinations...")
    combinations = build_combinations(
        graphs_by_original,
        interactions_by_object,
        originals,
    )
    if max_combinations is not None and max_combinations >= 0:
        combinations = combinations[:max_combinations]
    step4_path = out / "step4_combinations.json"
    with open(step4_path, "w", encoding="utf-8") as f:
        json.dump(combinations, f, indent=2)
    print(f"  {len(combinations)} combinations -> {step4_path}")

    # -- Step 5: Gemini hazard batches -----------------------------------------
    n_combos = len(combinations)
    n_hazard_calls = (
        (n_combos + hazard_batch_size - 1) // hazard_batch_size if n_combos else 0
    )
    print(
        f"Step 5: Gemini hazard batches "
        f"({n_hazard_calls} API calls, batch_size={hazard_batch_size}, "
        f"model={hazard_model_name}, timeout={timeout_seconds}s, "
        f"{n_combos} combinations)..."
    )
    assessments = evaluate_all_combinations(
        combinations,
        batch_size=hazard_batch_size,
        hazard_model_name=hazard_model_name,
        timeout_seconds=timeout_seconds,
        output_dir=out,
        fresh_hazard=fresh_hazard,
    )
    step5_path = out / "step5_all_assessments.json"
    haz_path = out / "hazardous_combinations.json"
    n_hazardous = sum(1 for a in assessments if a.hazardous)
    print(f"  wrote {step5_path} ({len(assessments)} assessments)")
    print(f"  {n_hazardous} hazardous rows -> {haz_path}")

    return out
