"""Step 4: Cartesian expansion — original_object x original_state x interaction_target."""

from __future__ import annotations

from typing import Dict, List, TypedDict

from soda_can_problem.graph_cache import ResolvedGraph


class CombinationDict(TypedDict):
    original_object: str
    graph_node_id: int
    original_object_state: Dict[str, str]
    original_object_state_label: str
    interaction_target: str


def build_combination_targets(
    interactions_by_object: Dict[str, List[str]],
    original_objects: List[str],
) -> List[str]:
    """
    Union of every interaction string from step 2 plus every original object slug,
    deduplicated case-insensitively, insertion order preserved (interactions first).
    """
    seen: set[str] = set()
    ordered: List[str] = []
    for xs in interactions_by_object.values():
        for x in xs:
            s = x.strip()
            if not s:
                continue
            key = s.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(s)
    for o in original_objects:
        s = o.strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(s)
    return ordered


def build_combinations(
    graphs_by_original: Dict[str, ResolvedGraph],
    interactions_by_object: Dict[str, List[str]],
    original_objects: List[str],
) -> List[CombinationDict]:
    """
    For each original that has a graph, iterate its state nodes and pair each with
    every target in ``union(step2 interactions) ∪ original_objects``, excluding
    self-pairs (original_object != interaction_target).
    """
    interaction_targets = build_combination_targets(
        interactions_by_object,
        original_objects,
    )
    if not interaction_targets:
        return []

    combos: List[CombinationDict] = []
    for original_object, resolved in graphs_by_original.items():
        if resolved is None or resolved.graph_dict is None:
            continue
        orig_key = original_object.strip().lower()
        nodes = resolved.graph_dict.get("nodes") or []
        for node in nodes:
            facts = node.get("facts") or {}
            label = node.get("label") or ""
            node_id = int(node.get("id", -1))
            for target in interaction_targets:
                if target.strip().lower() == orig_key:
                    continue
                combos.append(
                    {
                        "original_object": original_object,
                        "graph_node_id": node_id,
                        "original_object_state": dict(facts),
                        "original_object_state_label": label,
                        "interaction_target": target.strip(),
                    }
                )
    return combos
