"""Step 5: batched Gemini hazard judgments."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from soda_can_problem.defaults import (
    DEFAULT_HAZARD_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE_HAZARD,
)
from soda_can_problem.gemini_client import build_llm
from soda_can_problem.schemas import HazardAssessmentItem, HazardBatchOutput


def _combo_payload(row: Dict[str, Any]) -> str:
    payload = {
        "original_object": row["original_object"],
        "original_object_state": row["original_object_state"],
        "interaction_target": row["interaction_target"],
    }
    return json.dumps(payload, sort_keys=True)


def evaluate_hazard_batch(
    combinations_batch: List[Dict[str, Any]],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[HazardAssessmentItem]:
    """Judge one batch of combinations; returns assessments aligned to batch order."""
    if not combinations_batch:
        return []

    parser = PydanticOutputParser(pydantic_object=HazardBatchOutput)
    lines = []
    for i, row in enumerate(combinations_batch):
        lines.append(f"{i}. {_combo_payload(row)}")
    batch_block = "\n".join(lines)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a kitchen safety analyst. For each numbered scenario, "
                    "the original_object is currently in the state described by "
                    "original_object_state (variable=value facts). Decide whether "
                    "the original_object in that state being paired with, placed near, "
                    "or used together with the interaction_target could be hazardous.\n"
                    "Reply ONLY with JSON matching the format instructions. "
                    "Return exactly one assessment per input line, same order, "
                    "same indices 0..n-1.\n"
                    "original_object_state is the full state snapshot of the "
                    "original_object — use all relevant variables."
                ),
            ),
            (
                "human",
                (
                    "For EACH numbered combination below, answer:\n"
                    '"Is this combination hazardous in a kitchen environment?"\n\n'
                    "{batch_block}\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    llm = build_llm(model_name=model_name, temperature=DEFAULT_TEMPERATURE_HAZARD)
    chain = prompt | llm | parser
    parsed: HazardBatchOutput = chain.invoke(
        {
            "batch_block": batch_block,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    items = list(parsed.items)
    if len(items) != len(combinations_batch):
        raise ValueError(
            f"Hazard batch size mismatch: expected {len(combinations_batch)}, got {len(items)}"
        )
    for item, src in zip(items, combinations_batch, strict=True):
        if item.original_object != src["original_object"]:
            item.original_object = src["original_object"]
        if item.interaction_target != src["interaction_target"]:
            item.interaction_target = src["interaction_target"]
        if item.original_object_state != src["original_object_state"]:
            item.original_object_state = dict(src["original_object_state"])
    return items


def evaluate_all_combinations(
    combinations: List[Dict[str, Any]],
    *,
    batch_size: int = DEFAULT_HAZARD_BATCH_SIZE,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[HazardAssessmentItem]:
    all_items: List[HazardAssessmentItem] = []
    for start in range(0, len(combinations), batch_size):
        chunk = combinations[start : start + batch_size]
        all_items.extend(evaluate_hazard_batch(chunk, model_name=model_name))
    return all_items
