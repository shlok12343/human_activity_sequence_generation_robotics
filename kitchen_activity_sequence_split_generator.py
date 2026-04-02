from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from affordance_generator import generate_affordance_rules
from build_process_dag import build_and_render_process_dag_with_colored_edges

load_dotenv()


class SegmentedActivitySequences(BaseModel):
    """Structured output with separate normal and hazardous sequence groups."""

    normal_sequences: List[List[str]] = Field(
        description="List of safe/normal activity sequences."
    )
    hazardous_sequences: List[List[str]] = Field(
        description="List of intentionally hazardous activity sequences."
    )


def build_segmented_chain(
    model_name: str = "gemini-3.1-pro-preview", temperature: float = 0.7
):
    """Build a LangChain pipeline for split normal/hazardous generation."""
    parser = PydanticOutputParser(pydantic_object=SegmentedActivitySequences)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert in kitchen activity modeling and safety analysis.\n"
                    "Return ONLY JSON that matches the format instructions."
                ),
            ),
            (
                "human",
                (
                    "Normative sequence:\n{base_sequence}\n\n"
                    "Target object:\n{target_object}\n\n"
                    "World knowledge / affordance rules:\n{affordance_rules}\n\n"
                    "Generate exactly {num_normal_sequences} normal_sequences.\n"
                    "Generate exactly {num_hazardous_sequences} hazardous_sequences.\n\n"
                    "Requirements:\n"
                    "- normal_sequences must be realistic and safe.\n"
                    "- hazardous_sequences must include clear safety violations.\n"
                    "- Keep all steps focused on the target object in a kitchen.\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    return prompt | llm | parser, parser


def generate_normal_and_hazardous_sequences(
    base_sequence: List[str],
    affordance_rules: List[str],
    target_object: str,
    num_normal_sequences: int = 6,
    num_hazardous_sequences: int = 6,
) -> SegmentedActivitySequences:
    """Generate two separate lists: normal sequences and hazardous sequences."""
    chain, parser = build_segmented_chain()
    return chain.invoke(
        {
            "base_sequence": base_sequence,
            "affordance_rules": affordance_rules,
            "target_object": target_object,
            "num_normal_sequences": num_normal_sequences,
            "num_hazardous_sequences": num_hazardous_sequences,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def print_segmented_sequences(result: SegmentedActivitySequences) -> None:
    """Print normal and hazardous sequence groups in a readable format."""
    print("\nNormal Sequences:\n")
    for i, sequence in enumerate(result.normal_sequences, start=1):
        print(f"Normal Sequence {i}:")
        for j, step in enumerate(sequence, start=1):
            print(f"  {j}. {step}")
        print("-" * 50)

    print("\nHazardous Sequences:\n")
    for i, sequence in enumerate(result.hazardous_sequences, start=1):
        print(f"Hazardous Sequence {i}:")
        for j, step in enumerate(sequence, start=1):
            print(f"  {j}. {step}")
        print("-" * 50)


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

    result = generate_normal_and_hazardous_sequences(
        base_sequence=base_sequence_oven,
        affordance_rules=affordance_result.affordance_rules,
        target_object=target_object_oven,
        num_normal_sequences=10,
        num_hazardous_sequences=10,
    )

    print_segmented_sequences(result)

    output_path = build_and_render_process_dag_with_colored_edges(
        normal_sequences=result.normal_sequences,
        hazardous_sequences=result.hazardous_sequences,
        out_name="process_dag_normal_vs_hazardous",
    )
    print(f"Colored DAG saved to: {output_path}")


if __name__ == "__main__":
    main()
