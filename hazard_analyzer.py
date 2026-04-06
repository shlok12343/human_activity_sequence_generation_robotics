from __future__ import annotations

import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tabulate import tabulate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


KITCHEN_TASKS: Dict[str, List[str]] = {
    "knife_prep": [
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
    "protein_prep": [
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
    "oven_workflow": [
        "Utensils on countertop",
        "Oven on",
        "Utensils in oven",
        "Oven Used",
        "Oven Off",
        "Utensils removed from Oven",
        "Utensils placed on countertop",
    ],
    "dishwasher_cycle": [
        "Add Dishes to Dishwasher",
        "Dishwasher is on",
        "Add Dishwasher Detergent",
        "Run the dishwasher",
        "remove dishes",
    ],
    "liquid_handling": [
        "open fridge",
        "Get Liquid",
        "Remove Liquids",
        "close fridge",
        "Pour Liquid",
        "open fridge",
        "Put back Liquid",
        "close fridge",
    ],
    "dairy_ingredients": [
        "Open Fridge",
        "Remove Ingredients",
        "Use Ingredients",
        "Put back the ingredients",
        "Close Fridge",
    ],
    "kettle_use": [
        "Fill Kettle",
        "Heat On",
        "Pour water",
        "Heat Off",
        "Turn off the kettle",
    ],
    "toaster_use": [
        "Insert Bread",
        "Toaster On",
        "Remove Bread",
        "Power Off Toaster",
        "Turn off Toaster",
    ],
    "air_fryer_use": [
        "Air fryer on",
        "Air fryer used",
        "Air fryer is off",
        "Turn off the air fryer",
    ],
    "stove_use": [
        "Utensils on Stove",
        "Stove On",
        "Stove Used",
        "Stove Off",
        "Utensils removed from Stove",
        "Turn off the stove",
    ],
}


class HazardState(BaseModel):
    """One hazardous state/subsequence and its timing + reason."""

    hazardous_subsequence: List[str] = Field(
        description="Contiguous subsequence that is hazardous."
    )
    max_duration_seconds: int = Field(
        description="Maximum duration this hazardous state can persist before emergency risk."
    )
    reasoning: str = Field(
        description="Brief explanation of hazard mechanism."
    )


class HazardRules(BaseModel):
    """Structured output schema for hazard analysis of one task."""

    forbidden_transitions: List[HazardState] = Field(
        description="All hazardous subsequences with time thresholds and reasons."
    )


def build_hazard_rule_chain(
    model_name: str = "gemini-3.1-pro-preview", temperature: float = 0.2
):
    """Build prompt -> LLM -> parser chain for hazard rule generation."""
    parser = PydanticOutputParser(pydantic_object=HazardRules)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a robotics safety expert. Given a kitchen task state, define the safety constraints:\n\n"
                    "What transitions are hazardous?\n\n"
                    "How long can this state persist unattended before it becomes an emergency?\n\n"
                    "Why is it a risk?\n"
                    "Output in JSON format."
                ),
            ),
            (
                "human",
                (
                    "Task name:\n{task_name}\n\n"
                    "Base sequence:\n{base_sequence}\n\n"
                    "Analyze this sequence and return all hazardous contiguous subsequences.\n"
                    "Each subsequence must include:\n"
                    "- hazardous_subsequence: list of steps\n"
                    "- max_duration_seconds: integer threshold\n"
                    "- reasoning: short explanation\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    return prompt | llm | parser, parser


def generate_hazard_rules(task_name: str) -> HazardRules:
    """
    Query the LLM for hazardous subsequences, time thresholds, and reasons.

    The function looks up task_name in KITCHEN_TASKS for its base sequence.
    """
    if task_name not in KITCHEN_TASKS:
        raise ValueError(f"Unknown task_name '{task_name}'. Available: {list(KITCHEN_TASKS)}")

    chain, parser = build_hazard_rule_chain()
    return chain.invoke(
        {
            "task_name": task_name,
            "base_sequence": KITCHEN_TASKS[task_name],
            "format_instructions": parser.get_format_instructions(),
        }
    )


def print_hazard_table(master_results: Dict[str, HazardRules]) -> None:
    """Render the full hazard dictionary as a clean table."""
    rows: List[List[str]] = []
    for task_name, hazard_rules in master_results.items():
        for hazard in hazard_rules.forbidden_transitions:
            rows.append(
                [
                    task_name,
                    " -> ".join(hazard.hazardous_subsequence),
                    str(hazard.max_duration_seconds),
                    hazard.reasoning,
                ]
            )

    print("\nHazard Analysis Table:\n")
    print(
        tabulate(
            rows,
            headers=[
                "Task",
                "Unsafe Subsequence (Forbidden Transition)",
                "Max Unsafe Duration (s)",
                "Why Hazardous",
            ],
            tablefmt="grid",
        )
    )
def save_master_hazard_dictionary(
    master_results: Dict[str, HazardRules],
    out_path: str = "master_hazard_dictionary.txt",
) -> None:
    """Save full hazard results as pretty JSON text."""
    serializable = {
        task_name: rules.model_dump()
        for task_name, rules in master_results.items()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(serializable, indent=2))

def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export it before running this script."
        )

    master_hazard_dictionary: Dict[str, HazardRules] = {}

    for task_name in KITCHEN_TASKS:
        master_hazard_dictionary[task_name] = generate_hazard_rules(task_name)

    save_master_hazard_dictionary(master_hazard_dictionary)
    print_hazard_table(master_hazard_dictionary)
    print("Saved: master_hazard_dictionary.txt")


if __name__ == "__main__":
    main()
