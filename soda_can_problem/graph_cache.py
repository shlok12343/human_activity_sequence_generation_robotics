"""Build or load cached state graphs via ``run_state_graph_pipeline``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from soda_can_problem.defaults import DEFAULT_NUM_AFFORDANCE_RULES
from soda_can_problem.interaction_graph_registry import (
    propose_sequence_via_gemini,
    resolve_pipeline_inputs,
    slug_for_cache,
)


@dataclass
class ResolvedGraph:
    interaction_raw: str
    normalized_key: str
    slug: str
    status: str
    reason: str
    target_object: str
    graph_path: str
    validation_errors: List[str]
    node_count: int
    graph_dict: Optional[Dict[str, Any]]


def _graphs_dir(output_dir: Path) -> Path:
    d = output_dir / "graphs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_graph_for_interaction(
    interaction_raw: str,
    *,
    output_dir: Path,
    skip_cache: bool,
    use_gemini_fallback: bool,
    num_affordance_rules: int = DEFAULT_NUM_AFFORDANCE_RULES,
    model_name: str,
) -> ResolvedGraph:
    """
    Resolve interaction string to a state graph JSON file or record skip/failure.
    """
    key, status, target_object, base_sequence, reason = resolve_pipeline_inputs(
        interaction_raw,
        use_gemini_fallback=use_gemini_fallback,
    )
    slug = slug_for_cache(key or normalize_fallback_slug(interaction_raw))

    if status == "needs_fallback":
        try:
            proposed = propose_sequence_via_gemini(
                interaction_raw,
                model_name=model_name,
            )
            target_object = proposed.target_object.strip()
            base_sequence = [s.strip() for s in proposed.base_sequence if s.strip()]
            status = "ok"
            reason = "gemini_fallback_sequence"
        except Exception as exc:  # noqa: BLE001 — prototype resilience
            return ResolvedGraph(
                interaction_raw=interaction_raw,
                normalized_key=key,
                slug=slug,
                status="failed",
                reason=f"fallback_sequence_error:{exc}",
                target_object="",
                graph_path="",
                validation_errors=[],
                node_count=0,
                graph_dict=None,
            )

    if status in ("skipped_abstract", "skipped_no_mapping"):
        return ResolvedGraph(
            interaction_raw=interaction_raw,
            normalized_key=key,
            slug=slug,
            status=status,
            reason=reason,
            target_object=target_object,
            graph_path="",
            validation_errors=[],
            node_count=0,
            graph_dict=None,
        )

    graphs_dir = _graphs_dir(output_dir)
    cache_path = graphs_dir / f"{slug}.json"
    meta_path = graphs_dir / f"{slug}.meta.json"

    if not skip_cache and cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            graph_dict = json.load(f)
        nodes = graph_dict.get("nodes") or []
        val_errs: List[str] = []
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as mf:
                meta = json.load(mf)
                val_errs = list(meta.get("validation_errors", []))
        return ResolvedGraph(
            interaction_raw=interaction_raw,
            normalized_key=key,
            slug=slug,
            status="ok",
            reason="cache_hit",
            target_object=target_object,
            graph_path=str(cache_path),
            validation_errors=val_errs,
            node_count=len(nodes),
            graph_dict=graph_dict,
        )

    try:
        from state_graph_generator import (
            run_state_graph_pipeline,
            validate_state_graph_output,
        )

        pipeline_output = run_state_graph_pipeline(
            base_sequence=base_sequence,
            target_object=target_object,
            num_affordance_rules=num_affordance_rules,
        )
        sg = pipeline_output.state_graph
        val_errs = validate_state_graph_output(sg)
        graph_dict = sg.model_dump()
    except Exception as exc:  # noqa: BLE001
        return ResolvedGraph(
            interaction_raw=interaction_raw,
            normalized_key=key,
            slug=slug,
            status="failed",
            reason=f"pipeline_error:{exc}",
            target_object=target_object,
            graph_path="",
            validation_errors=[],
            node_count=0,
            graph_dict=None,
        )

    node_count = len(graph_dict.get("nodes") or [])
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "interaction_raw": interaction_raw,
                "normalized_key": key,
                "target_object": target_object,
                "validation_errors": val_errs,
            },
            f,
            indent=2,
        )

    return ResolvedGraph(
        interaction_raw=interaction_raw,
        normalized_key=key,
        slug=slug,
        status="ok",
        reason="built",
        target_object=target_object,
        graph_path=str(cache_path),
        validation_errors=val_errs,
        node_count=node_count,
        graph_dict=graph_dict,
    )


def normalize_fallback_slug(raw: str) -> str:
    r = raw.strip().lower().replace(" ", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in r)[:80]
