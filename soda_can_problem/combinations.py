"""Step 4: Cartesian expansion — original_object x original_state x interaction_target."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from soda_can_problem.graph_cache import ResolvedGraph


class CombinationDict(TypedDict):
    original_object: str
    graph_node_id: int
    original_object_state: Dict[str, str]
    original_object_state_label: str
    interaction_target: str


def build_combinations(
    graphs_by_original: Dict[str, ResolvedGraph],
    interactions_by_object: Dict[str, List[str]],
) -> List[CombinationDict]:
    """
    For each original that has a graph, iterate its state nodes and pair each
    with every interaction target discovered for that original in step 2.
    """
    combos: List[CombinationDict] = []
    for original_object, resolved in graphs_by_original.items():
        if resolved is None or resolved.graph_dict is None:
            continue
        interaction_targets = interactions_by_object.get(original_object, [])
        if not interaction_targets:
            continue
        nodes = resolved.graph_dict.get("nodes") or []
        for node in nodes:
            facts = node.get("facts") or {}
            label = node.get("label") or ""
            node_id = int(node.get("id", -1))
            for target in interaction_targets:
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
