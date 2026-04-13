from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from graphviz import Digraph
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from affordance_generator import generate_affordance_rules

load_dotenv()


class StateVariable(BaseModel):
    name: str = Field(description="Canonical variable name, e.g. oven_door.")
    allowed_values: List[str] = Field(
        description="Allowed values for this variable, e.g. ['open', 'closed']."
    )
    description: str = Field(description="What this variable tracks.")


class StateSchema(BaseModel):
    target_object: str
    variables: List[StateVariable]
    initial_state_facts: Dict[str, str] = Field(
        description="Starting state facts keyed by variable name."
    )


class TransitionCandidate(BaseModel):
    action: str
    from_conditions: Dict[str, str] = Field(
        description="Facts that must be true before this action."
    )
    to_effects: Dict[str, str] = Field(
        description="Facts changed by the action after execution."
    )


class TransitionCandidateSet(BaseModel):
    transitions: List[TransitionCandidate]


class GroundedTransition(BaseModel):
    action: str
    from_conditions: Dict[str, str]
    preconditions: List[str]
    invariants: List[str]
    effects: Dict[str, str]
    affordance_rule_refs: List[str]


class ForbiddenTransition(BaseModel):
    action: str
    from_conditions: Dict[str, str]
    reason: str


class GroundedTransitionSet(BaseModel):
    transitions: List[GroundedTransition]
    forbidden_transitions: List[ForbiddenTransition]


class ConsistencyReport(BaseModel):
    is_consistent: bool
    issues: List[str]
    recommended_fixes: List[str]


class StateGraphNode(BaseModel):
    id: int
    facts: Dict[str, str]
    label: str


class StateGraphEdge(BaseModel):
    source_id: int
    target_id: int
    action: str
    preconditions: List[str]
    invariants: List[str]
    effects: Dict[str, str]
    affordance_rule_refs: List[str]
    usage_count: int = 0


class StateGraph(BaseModel):
    nodes: List[StateGraphNode]
    edges: List[StateGraphEdge]
    initial_state_ids: List[int]
    forbidden_transitions: List[ForbiddenTransition]
    consistency_report: ConsistencyReport


@dataclass
class PipelineOutput:
    state_schema: StateSchema
    transition_candidates: TransitionCandidateSet
    grounded_transitions: GroundedTransitionSet
    consistency_report: ConsistencyReport
    state_graph: StateGraph


def _build_llm(model_name: str = "gemini-3.1-pro-preview", temperature: float = 0.1):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def stage1_extract_state_schema(
    base_sequence: List[str],
    target_object: str,
) -> StateSchema:
    parser = PydanticOutputParser(pydantic_object=StateSchema)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a robotics state-modeling expert.\n"
                    "Extract a canonical state schema for a kitchen object workflow.\n"
                    "Return ONLY JSON that matches the format instructions."
                ),
            ),
            (
                "human",
                (
                    "Target object:\n{target_object}\n\n"
                    "Base sequence:\n{base_sequence}\n\n"
                    "Define the minimum complete state variables and allowed values.\n"
                    "Provide a valid initial_state_facts assignment.\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    chain = prompt | _build_llm() | parser
    return chain.invoke(
        {
            "target_object": target_object,
            "base_sequence": base_sequence,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def stage2_generate_transition_candidates(
    base_sequence: List[str],
    target_object: str,
    state_schema: StateSchema,
) -> TransitionCandidateSet:
    parser = PydanticOutputParser(pydantic_object=TransitionCandidateSet)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a robotics planning expert.\n"
                    "Generate transition candidates over a provided state schema.\n"
                    "Return ONLY JSON that matches the format instructions."
                ),
            ),
            (
                "human",
                (
                    "Target object:\n{target_object}\n\n"
                    "State schema:\n{state_schema}\n\n"
                    "Base sequence:\n{base_sequence}\n\n"
                    "Produce action transitions where from_conditions and to_effects are explicit,\n"
                    "using only variables from the schema.\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    chain = prompt | _build_llm() | parser
    return chain.invoke(
        {
            "target_object": target_object,
            "state_schema": state_schema.model_dump(),
            "base_sequence": base_sequence,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def stage3_ground_affordance_guards(
    target_object: str,
    state_schema: StateSchema,
    transition_candidates: TransitionCandidateSet,
    affordance_rules: List[str],
) -> GroundedTransitionSet:
    parser = PydanticOutputParser(pydantic_object=GroundedTransitionSet)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a robotics safety and affordance expert.\n"
                    "Ground transition guards using the affordance rules and state schema.\n"
                    "Return ONLY JSON that matches the format instructions."
                ),
            ),
            (
                "human",
                (
                    "Target object:\n{target_object}\n\n"
                    "State schema:\n{state_schema}\n\n"
                    "Transition candidates:\n{transition_candidates}\n\n"
                    "Affordance rules:\n{affordance_rules}\n\n"
                    "For each legal transition, provide preconditions, invariants, effects,\n"
                    "and references to supporting affordance rules.\n"
                    "List disallowed transitions in forbidden_transitions.\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    chain = prompt | _build_llm() | parser
    return chain.invoke(
        {
            "target_object": target_object,
            "state_schema": state_schema.model_dump(),
            "transition_candidates": transition_candidates.model_dump(),
            "affordance_rules": affordance_rules,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def stage4_validate_consistency(
    target_object: str,
    state_schema: StateSchema,
    grounded_transitions: GroundedTransitionSet,
) -> ConsistencyReport:
    parser = PydanticOutputParser(pydantic_object=ConsistencyReport)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a formal verification assistant for state machines.\n"
                    "Check consistency and detect contradictions in transitions.\n"
                    "Return ONLY JSON that matches the format instructions."
                ),
            ),
            (
                "human",
                (
                    "Target object:\n{target_object}\n\n"
                    "State schema:\n{state_schema}\n\n"
                    "Grounded transitions:\n{grounded_transitions}\n\n"
                    "Examples of inconsistencies to catch:\n"
                    "- impossible effects outside allowed_values\n"
                    "- missing required precondition for an action\n"
                    "- contradictory from_conditions and effects\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    chain = prompt | _build_llm() | parser
    return chain.invoke(
        {
            "target_object": target_object,
            "state_schema": state_schema.model_dump(),
            "grounded_transitions": grounded_transitions.model_dump(),
            "format_instructions": parser.get_format_instructions(),
        }
    )


def _facts_key(facts: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(facts.items(), key=lambda item: item[0]))


def _state_label(facts: Dict[str, str]) -> str:
    return "\n".join(f"{k}={v}" for k, v in sorted(facts.items(), key=lambda item: item[0]))


def _conditions_satisfied(state_facts: Dict[str, str], conditions: Dict[str, str]) -> bool:
    return all(state_facts.get(key) == value for key, value in conditions.items())


def _apply_effects(state_facts: Dict[str, str], effects: Dict[str, str]) -> Dict[str, str]:
    updated = dict(state_facts)
    updated.update(effects)
    return updated


def _apply_usage_counts_from_sequence(edges: List[StateGraphEdge], base_sequence: List[str]) -> None:
    for action_step in base_sequence:
        action_step_norm = action_step.strip().lower()
        for edge in edges:
            if edge.action.strip().lower() == action_step_norm:
                edge.usage_count += 1


def build_state_graph(
    state_schema: StateSchema,
    grounded_transitions: GroundedTransitionSet,
    consistency_report: ConsistencyReport,
    base_sequence: List[str],
    max_expansions: int = 300,
) -> StateGraph:
    key_to_node_id: Dict[Tuple[Tuple[str, str], ...], int] = {}
    node_facts_by_id: Dict[int, Dict[str, str]] = {}
    edges_map: Dict[Tuple[int, int, str], StateGraphEdge] = {}
    queue: List[Dict[str, str]] = []

    initial_key = _facts_key(state_schema.initial_state_facts)
    key_to_node_id[initial_key] = 0
    node_facts_by_id[0] = dict(state_schema.initial_state_facts)
    queue.append(dict(state_schema.initial_state_facts))
    next_node_id = 1

    expansions = 0
    while queue and expansions < max_expansions:
        current_facts = queue.pop(0)
        current_key = _facts_key(current_facts)
        current_id = key_to_node_id[current_key]

        for transition in grounded_transitions.transitions:
            if not _conditions_satisfied(current_facts, transition.from_conditions):
                continue

            next_facts = _apply_effects(current_facts, transition.effects)
            next_key = _facts_key(next_facts)
            if next_key not in key_to_node_id:
                key_to_node_id[next_key] = next_node_id
                node_facts_by_id[next_node_id] = next_facts
                queue.append(next_facts)
                next_node_id += 1

            target_id = key_to_node_id[next_key]
            edge_key = (current_id, target_id, transition.action)
            if edge_key not in edges_map:
                edges_map[edge_key] = StateGraphEdge(
                    source_id=current_id,
                    target_id=target_id,
                    action=transition.action,
                    preconditions=transition.preconditions,
                    invariants=transition.invariants,
                    effects=transition.effects,
                    affordance_rule_refs=transition.affordance_rule_refs,
                )
        expansions += 1

    nodes = [
        StateGraphNode(id=node_id, facts=facts, label=_state_label(facts))
        for node_id, facts in sorted(node_facts_by_id.items(), key=lambda item: item[0])
    ]
    edges = list(edges_map.values())
    _apply_usage_counts_from_sequence(edges, base_sequence)

    return StateGraph(
        nodes=nodes,
        edges=edges,
        initial_state_ids=[0],
        forbidden_transitions=grounded_transitions.forbidden_transitions,
        consistency_report=consistency_report,
    )


def render_state_graph(state_graph: StateGraph, out_name: str) -> str:
    dot = Digraph("state_graph", format="png", engine="dot")
    dot.attr(rankdir="LR")
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#4A5568",
        fillcolor="#F7FAFC",
        fontname="Helvetica",
    )
    dot.attr("edge", color="#2D3748", fontname="Helvetica")

    for node in state_graph.nodes:
        dot.node(f"n{node.id}", node.label if node.label else f"state_{node.id}")

    for edge in state_graph.edges:
        pre = ", ".join(edge.preconditions) if edge.preconditions else "-"
        inv = ", ".join(edge.invariants) if edge.invariants else "-"
        label = f"{edge.action}\npre: {pre}\ninv: {inv}"
        dot.edge(f"n{edge.source_id}", f"n{edge.target_id}", label=label)

    return dot.render(filename=out_name, cleanup=True)


def export_state_graph_json(state_graph: StateGraph, out_name: str) -> str:
    out_path = f"{out_name}.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(state_graph.model_dump(), file, indent=2)
    return out_path


def run_state_graph_pipeline(
    base_sequence: List[str],
    target_object: str,
    num_affordance_rules: int = 10,
) -> PipelineOutput:
    affordance_result = generate_affordance_rules(
        base_sequence=base_sequence,
        target_object=target_object,
        num_rules=num_affordance_rules,
    )

    state_schema = stage1_extract_state_schema(
        base_sequence=base_sequence, target_object=target_object
    )
    transition_candidates = stage2_generate_transition_candidates(
        base_sequence=base_sequence,
        target_object=target_object,
        state_schema=state_schema,
    )
    grounded_transitions = stage3_ground_affordance_guards(
        target_object=target_object,
        state_schema=state_schema,
        transition_candidates=transition_candidates,
        affordance_rules=affordance_result.affordance_rules,
    )
    consistency_report = stage4_validate_consistency(
        target_object=target_object,
        state_schema=state_schema,
        grounded_transitions=grounded_transitions,
    )

    state_graph = build_state_graph(
        state_schema=state_schema,
        grounded_transitions=grounded_transitions,
        consistency_report=consistency_report,
        base_sequence=base_sequence,
    )

    return PipelineOutput(
        state_schema=state_schema,
        transition_candidates=transition_candidates,
        grounded_transitions=grounded_transitions,
        consistency_report=consistency_report,
        state_graph=state_graph,
    )


def validate_state_graph_output(state_graph: StateGraph) -> List[str]:
    errors: List[str] = []
    if not state_graph.nodes:
        errors.append("State graph has no nodes.")
    if not state_graph.edges:
        errors.append("State graph has no edges.")

    node_ids = {node.id for node in state_graph.nodes}
    for edge in state_graph.edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            errors.append(
                f"Edge {edge.action} references unknown node ({edge.source_id}->{edge.target_id})."
            )
        if not edge.preconditions:
            errors.append(f"Edge {edge.action} is missing preconditions.")
    return errors


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export it before running this script."
        )

    task_configs = [
        {
            "task_name": "oven",
            "target_object": "oven",
            "out_name": "state_graph_oven_affordance_rules",
            "base_sequence": [
                "Utensils on countertop",
                "Oven on",
                "Utensils in oven",
                "Oven Used",
                "Oven Off",
                "Utensils removed from Oven",
                "Utensils placed on countertop",
            ],
        },
        {
            "task_name": "vegetables",
            "target_object": "vegetables",
            "out_name": "state_graph_vegetables_affordance_rules",
            "base_sequence": [
                "Open Fridge",
                "Take out vegetables",
                "Close Fridge",
                "Cut Vegetables",
                "Open Fridge",
                "Put back vegetables",
                "Close Fridge",
                "Cook vegetables",
                "Eat vegetables",
            ],
        },
        {
            "task_name": "knives",
            "target_object": "knives",
            "out_name": "state_graph_knives_affordance_rules",
            "base_sequence": [
                "open drawer",
                "Remove knife from block",
                "Place vegetable on board",
                "Slice vegetable",
                "Wipe blade",
                "Place knife on countertop",
                "Wash knife",
                "Dry knife",
                "Return knife to block",
                "close drawer",
            ],
        },
        {
            "task_name": "proteins",
            "target_object": "protein",
            "out_name": "state_graph_proteins_affordance_rules",
            "base_sequence": [
                "open fridge",
                "take out protein",
                "close fridge",
                "Remove protein from packaging",
                "Pat protein dry",
                "Cut protein into pieces",
                "Place protein in hot pan",
                "Wash hands with soap and water",
                "cook protein",
                "Remove protein from heat",
                "Let protein rest",
                "eat protein",
            ],
        },
    ]

    for config in task_configs:
        print(f"\n=== Generating state graph for: {config['task_name']} ===")
        pipeline_output = run_state_graph_pipeline(
            base_sequence=config["base_sequence"],
            target_object=config["target_object"],
            num_affordance_rules=10,
        )

        validation_errors = validate_state_graph_output(pipeline_output.state_graph)
        if validation_errors:
            print("Validation issues:")
            for issue in validation_errors:
                print(f"- {issue}")
        else:
            print("Validation passed.")

        png_path = render_state_graph(
            pipeline_output.state_graph, out_name=config["out_name"]
        )
        json_path = export_state_graph_json(
            pipeline_output.state_graph, out_name=config["out_name"]
        )

        print(f"State graph image saved to: {png_path}")
        print(f"State graph JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
