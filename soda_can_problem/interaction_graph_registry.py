"""
Map Gemini-discovered interaction entities to (target_object, base_sequence)
for ``run_state_graph_pipeline``. Includes curated seeds from state_graph_generator
task configs and minimal sequences for common appliances.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from soda_can_problem.defaults import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE_DISCOVERY
from soda_can_problem.gemini_client import build_llm
from soda_can_problem.schemas import ProposedActivitySequence

CuratedEntry = Tuple[str, List[str]]

CURATED_REGISTRY: Dict[str, CuratedEntry] = {
    # From state_graph_generator.main task_configs
    "oven": (
        "oven",
        [
            "Utensils on countertop",
            "Oven on",
            "Utensils in oven",
            "Oven Used",
            "Oven Off",
            "Utensils removed from Oven",
            "Utensils placed on countertop",
        ],
    ),
    "vegetables": (
        "vegetables",
        [
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
    ),
    "fridge": (
        "vegetables",
        [
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
    ),
    "refrigerator": (
        "vegetables",
        [
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
    ),
    "knives": (
        "knives",
        [
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
    ),
    "knife": (
        "knives",
        [
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
    ),
    "knife_block": (
        "knives",
        [
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
    ),
    "drawer": (
        "knives",
        [
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
    ),
    "cutting_board": (
        "knives",
        [
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
    ),
    "protein": (
        "protein",
        [
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
    ),
    "proteins": (
        "protein",
        [
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
    ),
    "microwave": (
        "microwave",
        [
            "Open microwave door",
            "Place food in microwave",
            "Close microwave door",
            "Microwave on",
            "Microwave finished",
            "Open microwave door",
            "Remove food",
            "Close microwave door",
        ],
    ),
    "stove": (
        "stove",
        [
            "Place pan on stove",
            "Turn stove on",
            "Cook food",
            "Turn stove off",
            "Remove pan from stove",
        ],
    ),
    "cooktop": (
        "stove",
        [
            "Place pan on stove",
            "Turn stove on",
            "Cook food",
            "Turn stove off",
            "Remove pan from stove",
        ],
    ),
    "sink": (
        "sink",
        [
            "Turn on faucet",
            "Rinse dishes",
            "Turn off faucet",
            "Drain sink",
        ],
    ),
    "faucet": (
        "sink",
        [
            "Turn on faucet",
            "Rinse dishes",
            "Turn off faucet",
            "Drain sink",
        ],
    ),
    "dishwasher": (
        "dishwasher",
        [
            "Open dishwasher",
            "Load dishes",
            "Close dishwasher",
            "Start dishwasher",
            "Open dishwasher",
            "Unload dishes",
        ],
    ),
}

# Non-appliance / too-abstract — skip state-graph pipeline (plan v1).
SKIP_ABSTRACT_KEYS: frozenset[str] = frozenset(
    {
        "hand",
        "hands",
        "table",
        "counter",
        "countertop",
        "floor",
        "human",
        "person",
        "user",
    }
)

SYNONYM_NORMALIZATION: Dict[str, str] = {
    "freezer": "fridge",
    "kitchen_sink": "sink",
    "kitchen_sink_area": "sink",
    "cuttingboard": "cutting_board",
    "cutting board": "cutting_board",
    "knife drawer": "drawer",
    "oven_rack": "oven",
    "gas_stove": "stove",
    "electric_stove": "stove",
}


def normalize_interaction_key(raw: str) -> str:
    s = raw.strip().lower()
    s = s.replace("'", "").replace('"', "")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return SYNONYM_NORMALIZATION.get(s, s)


def slug_for_cache(normalized_key: str) -> str:
    return normalized_key[:80] if normalized_key else "unknown"


def lookup_curated(normalized_key: str) -> Optional[CuratedEntry]:
    if normalized_key in CURATED_REGISTRY:
        return CURATED_REGISTRY[normalized_key]
    return None


def propose_sequence_via_gemini(
    entity_description: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> ProposedActivitySequence:
    """Ask Gemini for a minimal activity sequence when no curated mapping exists."""
    parser = PydanticOutputParser(pydantic_object=ProposedActivitySequence)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You help build structured kitchen activity traces for "
                    "robot affordance modeling. Reply ONLY with JSON matching "
                    "the format instructions."
                ),
            ),
            (
                "human",
                (
                    "Kitchen entity to model:\n{entity}\n\n"
                    "Propose a short ordered list of activity steps (verb phrases) "
                    "a person typically performs with this entity in a home kitchen. "
                    "Include at least one step that changes operational state "
                    "(on/off, open/closed, running/stopped) if applicable.\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    llm = build_llm(model_name=model_name, temperature=DEFAULT_TEMPERATURE_DISCOVERY)
    chain = prompt | llm | parser
    return chain.invoke(
        {
            "entity": entity_description,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def resolve_pipeline_inputs(
    interaction_raw: str,
    *,
    use_gemini_fallback: bool,
) -> Tuple[str, str, str, List[str], str]:
    """
    Returns ``(normalized_key, status, target_object, base_sequence, reason)``.
    Status is ``ok``, ``skipped_abstract``, ``skipped_no_mapping``, or ``needs_fallback``.
    """
    key = normalize_interaction_key(interaction_raw)
    if not key:
        return "", "skipped_no_mapping", "", [], "empty interaction string"

    if key in SKIP_ABSTRACT_KEYS:
        return key, "skipped_abstract", "", [], "abstract location/agent — no appliance graph"

    curated = lookup_curated(key)
    if curated:
        target_object, seq = curated
        return key, "ok", target_object, seq, ""

    if use_gemini_fallback:
        return key, "needs_fallback", "", [], ""

    return key, "skipped_no_mapping", "", [], "no curated mapping and fallback disabled"
