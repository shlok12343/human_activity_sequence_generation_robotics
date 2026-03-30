from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from graphviz import Digraph


SEQUENCES: List[List[str]] = [
    [
        "Open Fridge",
        "Take out vegetables",
        "Close Fridge",
        "Cut Vegetables",
        "Cook vegetables",
        "Eat vegetables",
    ],
    [
        "Open Fridge",
        "Take out vegetables",
        "Close Fridge",
        "Cut Vegetables",
        "Eat vegetables",
    ],
    [
        "Open Fridge",
        "Take out vegetables",
        "Cut Vegetables",
        "Cook vegetables",
        "Eat vegetables",
    ],
    [
        "Open Fridge",
        "Take out vegetables",
        "Close Fridge",
        "Cut Vegetables",
        "Eat vegetables",
        "Open Fridge",
        "Put back vegetables",
        "Close Fridge",
    ],
    [
        "Open Fridge",
        "Take out vegetables",
        "Close Fridge",
        "Cook vegetables",
        "Eat vegetables",
    ],
    [
        "Open Fridge",
        "Take out vegetables",
        "Close Fridge",
        "Cut Vegetables",
        "Cook vegetables",
        "Open Fridge",
        "Put back vegetables",
    ],
]


@dataclass
class TrieNode:
    label: str
    children: Dict[str, "TrieNode"] = field(default_factory=dict)


@dataclass
class DagNode:
    node_id: int
    label: str
    # Each edge stores (step_label, child_node_id).
    edges: Tuple[Tuple[str, int], ...]


def build_prefix_trie(sequences: List[List[str]]) -> TrieNode:
    root = TrieNode(label="START")
    for sequence in sequences:
        current = root
        for step in sequence:
            if step not in current.children:
                current.children[step] = TrieNode(label=step)
            current = current.children[step]
    return root


def merge_equivalent_suffix_states(root: TrieNode) -> Tuple[int, Dict[int, DagNode]]:
    """
    Convert trie -> DAG by merging equivalent suffix states bottom-up.

    This preserves a DAG and naturally:
    - merges common prefixes (from trie construction),
    - converges identical endings (e.g., shared leaf steps),
    - keeps repeated labels distinct when their future differs.
    """
    signature_to_id: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], int] = {}
    dag_nodes: Dict[int, DagNode] = {}
    next_id = 0

    def canonicalize(node: TrieNode) -> int:
        nonlocal next_id
        child_sigs: List[Tuple[str, int]] = []
        for step_label, child in sorted(node.children.items(), key=lambda x: x[0]):
            child_id = canonicalize(child)
            child_sigs.append((step_label, child_id))

        signature = (node.label, tuple(child_sigs))
        existing = signature_to_id.get(signature)
        if existing is not None:
            return existing

        current_id = next_id
        next_id += 1
        signature_to_id[signature] = current_id
        dag_nodes[current_id] = DagNode(
            node_id=current_id, label=node.label, edges=tuple(child_sigs)
        )
        return current_id

    root_id = canonicalize(root)
    return root_id, dag_nodes


def render_process_dag(root_id: int, dag_nodes: Dict[int, DagNode], out_name: str) -> str:
    dot = Digraph("process_dag", format="png", engine="dot")
    dot.attr(rankdir="TB")
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#4A5568",
        fillcolor="#F7FAFC",
        fontname="Helvetica",
    )
    dot.attr("edge", color="#718096")

    reachable = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(child_id for _, child_id in dag_nodes[node_id].edges)

    for node_id in sorted(reachable):
        node = dag_nodes[node_id]
        dot.node(f"n{node_id}", node.label)

    for node_id in sorted(reachable):
        node = dag_nodes[node_id]
        for _, child_id in node.edges:
            dot.edge(f"n{node_id}", f"n{child_id}")

    # Produces process_dag.png in the current directory.
    return dot.render(filename=out_name, cleanup=True)


def build_and_render_process_dag(
    sequences: List[List[str]], out_name: str = "process_dag"
) -> str:
    """Build a merged DAG from sequences and render it to a PNG file."""
    trie_root = build_prefix_trie(sequences)
    root_id, dag_nodes = merge_equivalent_suffix_states(trie_root)
    return render_process_dag(root_id, dag_nodes, out_name=out_name)


def main() -> None:
    output_path = build_and_render_process_dag(SEQUENCES, out_name="process_dag")
    print(f"DAG saved to: {output_path}")


if __name__ == "__main__":
    main()
