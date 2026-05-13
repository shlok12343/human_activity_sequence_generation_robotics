"""Step 2: Gemini discovers normal interaction partners per original object."""

from __future__ import annotations

from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from soda_can_problem.defaults import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE_DISCOVERY,
)
from soda_can_problem.gemini_client import build_llm
from soda_can_problem.schemas import InteractionDiscoveryOutput


def discover_interactions(
    original_objects: List[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, List[str]]:
    """
    Ask Gemini which kitchen entities each original object normally interacts with.

    Returns mapping ``original_slug -> [interaction strings]``.
    """
    parser = PydanticOutputParser(pydantic_object=InteractionDiscoveryOutput)
    objects_block = "\n".join(f"- {o}" for o in original_objects)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert on everyday kitchen tool use. "
                    "Answer ONLY with JSON that matches the format instructions."
                ),
            ),
            (
                "human",
                (
                    "For EACH of the following kitchen objects, answer:\n"
                    '"What objects, appliances, locations, or kitchen items does '
                    'this object normally interact with in a kitchen environment?"\n\n'
                    "Objects:\n{objects_block}\n\n"
                    "Rules:\n"
                    "- Use concise noun phrases (e.g. fridge, oven, cutting_board, sink).\n"
                    "- Include roughly 3–12 interaction targets per object.\n"
                    "- Prefer physically distinct entities (appliances, fixtures, surfaces).\n"
                    "- object_name must exactly match one of the input slugs.\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    llm = build_llm(model_name=model_name, temperature=DEFAULT_TEMPERATURE_DISCOVERY)
    chain = prompt | llm | parser
    result: InteractionDiscoveryOutput = chain.invoke(
        {
            "objects_block": objects_block,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    def norm_slug(s: str) -> str:
        return s.strip().lower().replace("-", "_")

    rows_by_norm = {norm_slug(row.object_name): row for row in result.objects}

    out: Dict[str, List[str]] = {}
    for canon in original_objects:
        row = rows_by_norm.get(norm_slug(canon))
        if row is None:
            out[canon] = []
            continue
        seen: set[str] = set()
        cleaned: List[str] = []
        for item in row.interacts_with:
            s = item.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                cleaned.append(s)
        out[canon] = cleaned
    return out
