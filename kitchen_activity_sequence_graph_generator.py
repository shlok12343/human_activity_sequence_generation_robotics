from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from graphviz import Digraph

from affordance_generator import generate_affordance_rules
from kitchen_activity_sequence_split_generator import (
    SegmentedActivitySequences,
    generate_normal_and_hazardous_sequences,
    print_segmented_sequences,
)

load_dotenv()


@dataclass
class GraphNode:
    node_id: int
    label: str


@dataclass
class GraphEdge:
    source_id: int
    target_id: int
    color: str


def build_process_graph_with_hazard_branch_coloring(
    normal_sequences: List[List[str]],
    hazardous_sequences: List[List[str]],
) -> Tuple[Dict[int, GraphNode], List[GraphEdge]]:
    """
    Build a prefix graph where:
    - safe/shared path edges are black
    - hazardous branch edges are red from first divergence onward
    """
    next_node_id = 0
    prefix_to_node_id: Dict[Tuple[str, ...], int] = {(): next_node_id}
    nodes: Dict[int, GraphNode] = {next_node_id: GraphNode(node_id=next_node_id, label="START")}
    next_node_id += 1

    edge_colors: Dict[Tuple[int, int], str] = {}

    def get_or_create_node(prefix: Tuple[str, ...], label: str) -> int:
        nonlocal next_node_id
        existing = prefix_to_node_id.get(prefix)
        if existing is not None:
            return existing
        node_id = next_node_id
        next_node_id += 1
        prefix_to_node_id[prefix] = node_id
        nodes[node_id] = GraphNode(node_id=node_id, label=label)
        return node_id

    # Add normal sequences first; these establish baseline black edges.
    for sequence in normal_sequences:
        current_prefix: Tuple[str, ...] = ()
        current_node_id = prefix_to_node_id[current_prefix]
        for step in sequence:
            next_prefix = current_prefix + (step,)
            next_node_id_for_prefix = get_or_create_node(next_prefix, step)
            edge_colors[(current_node_id, next_node_id_for_prefix)] = "#000000"
            current_prefix = next_prefix
            current_node_id = next_node_id_for_prefix

    # Add hazardous sequences.
    # Keep edges black while matching existing safe path, then red after divergence.
    for sequence in hazardous_sequences:
        current_prefix = ()
        current_node_id = prefix_to_node_id[current_prefix]
        diverged = False

        for step in sequence:
            next_prefix = current_prefix + (step,)
            next_node_id_for_prefix = get_or_create_node(next_prefix, step)
            edge_key = (current_node_id, next_node_id_for_prefix)

            existing_color = edge_colors.get(edge_key)
            if not diverged:
                if existing_color is None:
                    diverged = True
                    edge_colors[edge_key] = "#E53E3E"
                else:
                    # Shared edge stays black.
                    edge_colors[edge_key] = "#000000"
            else:
                # Once branched from safe path, keep hazardous continuation red.
                edge_colors[edge_key] = "#E53E3E"

            current_prefix = next_prefix
            current_node_id = next_node_id_for_prefix

    edges = [
        GraphEdge(source_id=src, target_id=dst, color=color)
        for (src, dst), color in edge_colors.items()
    ]
    return nodes, edges


def render_process_graph(
    nodes: Dict[int, GraphNode],
    edges: List[GraphEdge],
    out_name: str = "process_graph_normal_vs_hazardous",
) -> str:
    """Render process graph PNG with colored edges."""
    dot = Digraph("process_graph", format="png", engine="dot")
    dot.attr(rankdir="TB")
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#4A5568",
        fillcolor="#F7FAFC",
        fontname="Helvetica",
    )
    dot.attr("edge", color="#000000")

    for node_id in sorted(nodes):
        dot.node(f"n{node_id}", nodes[node_id].label)

    for edge in edges:
        dot.edge(f"n{edge.source_id}", f"n{edge.target_id}", color=edge.color)

    return dot.render(filename=out_name, cleanup=True)


def build_and_render_process_graph_with_hazard_branch_coloring(
    normal_sequences: List[List[str]],
    hazardous_sequences: List[List[str]],
    out_name: str = "process_graph_normal_vs_hazardous",
) -> str:
    """Convenience wrapper to build and render the graph."""
    nodes, edges = build_process_graph_with_hazard_branch_coloring(
        normal_sequences=normal_sequences,
        hazardous_sequences=hazardous_sequences,
    )
    return render_process_graph(nodes=nodes, edges=edges, out_name=out_name)


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export it before running this script."
        )

    base_sequence_oven = [
        "Utensils on countertop",
        "Oven on",
        "Utensils in oven",
        "Oven Used",
        "Oven Off",
        "Utensils removed from Oven",
        "Utensils placed on countertop",
    ]
    target_object_oven = "oven"

    affordance_result = generate_affordance_rules(
        base_sequence=base_sequence_oven,
        target_object=target_object_oven,
        num_rules=8,
    )

    segmented_result: SegmentedActivitySequences = generate_normal_and_hazardous_sequences(
        base_sequence=base_sequence_oven,
        affordance_rules=affordance_result.affordance_rules,
        target_object=target_object_oven,
        num_normal_sequences=10,
        num_hazardous_sequences=10,
    )

    print_segmented_sequences(segmented_result)

    output_path = build_and_render_process_graph_with_hazard_branch_coloring(
        normal_sequences=segmented_result.normal_sequences,
        hazardous_sequences=segmented_result.hazardous_sequences,
        out_name="process_graph_normal_vs_hazardous",
    )
    print(f"Process graph saved to: {output_path}")


if __name__ == "__main__":
    main()
