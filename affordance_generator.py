from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class AffordanceRules(BaseModel):
    """Structured output schema for generated affordance rules."""

    affordance_rules: List[str] = Field(
        description="A list of kitchen affordance rules inferred from the activity sequence."
    )


def build_affordance_chain(
    model_name: str = "gemini-3.1-pro-preview", temperature: float = 0.3
):
    """Build a LangChain pipeline for affordance-rule generation."""
    parser = PydanticOutputParser(pydantic_object=AffordanceRules)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert in human activity understanding and kitchen affordances.\n"
                    "Infer concise, logically valid affordance rules from a sequence of actions.\n"
                    "Return ONLY JSON that matches the format instructions."
                ),
            ),
            (
                "human",
                (
                    "Target object:\n{target_object}\n\n"
                    "Observed action sequence:\n{base_sequence}\n\n"
                    "Generate exactly {num_rules} affordance rules inferred from this sequence.\n"
                    "Rules should capture preconditions and ordering constraints, e.g.:\n"
                    "- something must happen before another action\n"
                    "- an object must be in a valid state to perform an action\n"
                    "Keep rules specific to the provided target object and kitchen context.\n\n"
                    
                    "An Affordance Rule is NOT a safety recommendation or a 'best practice' (e.g., Dont leave the stove on).\n"
                    "Instead, an Affordance Rule defines a PHYSICAL IMPOSSIBILITY based on the state of the environment.\n"
                    "Example: You cannot take an item from a closed container before opening it or A fridge being closed before opening. or You cannot removed dishes before adding them to the dishwasher\n"


                    "{format_instructions}"
                ),
            ),
        ]
    )
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    return prompt | llm | parser, parser


def generate_affordance_rules(
    base_sequence: List[str],
    target_object: str,
    num_rules: int = 5,
) -> AffordanceRules:
    """Generate affordance rules from a given sequence."""
    chain, parser = build_affordance_chain()
    return chain.invoke(
        {
            "target_object": target_object,
            "base_sequence": base_sequence,
            "num_rules": num_rules,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def print_affordance_rules(result: AffordanceRules) -> None:
    """Print generated affordance rules in a readable format."""
    print("\nGenerated Affordance Rules:\n")
    for i, rule in enumerate(result.affordance_rules, start=1):
        print(f"{i}. {rule}")


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to .env before running.")

    base_sequence_liquids = [
        "open fridge",
        "Get Liquid",
        "Remove Liquids",
        "close fridge",
        "Pour Liquid",
        "open fridge",
        "Put back Liquid",
        "close fridge",
    ]

    target_object_liquids = "liquids"

    result = generate_affordance_rules(
        base_sequence=base_sequence_liquids,
        target_object=target_object_liquids,
        num_rules=5,
    )
    print_affordance_rules(result)


if __name__ == "__main__":
    main()
