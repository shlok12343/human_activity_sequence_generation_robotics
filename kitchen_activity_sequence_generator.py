from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()



class ActivitySequences(BaseModel):
    """Structured output schema for generated activity sequences."""

    sequences: List[List[str]] = Field(
        description="A list of alternative human activity sequences; each sequence is a list of steps."
    )


def build_chain(model_name: str = "gemini-3.1-pro-preview", temperature: float = 0.7):
    """Build a LangChain pipeline: prompt -> LLM -> Pydantic parser."""
    parser = PydanticOutputParser(pydantic_object=ActivitySequences)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert in kitchen activity modeling and safety analysis.\n"
                    "Generate realistic alternative activity sequences for a single object workflow.\n"
                    "Return ONLY JSON that matches the format instructions."
                ),
            ),
            (
                "human",
                (
                    "Normative sequence:\n{base_sequence}\n\n"
                    "World knowledge / affordance rules:\n{affordance_rules}\n\n"
                    "Generate exactly {num_sequences} alternative sequences.\n"
                    "Include a mix of:\n"
                    "- Optimal/Efficient versions (fewer steps)\n"
                    "- Hazardous versions (e.g., leaving fridge open, leaving stove on)\n"
                    "- Alternate-order versions (e.g., eating raw vs cooked)\n\n"
                    "Keep all steps focused on manipulations around one object flow in a kitchen.\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    chain = prompt | llm | parser
    return chain, parser


def generate_sequences(
    base_sequence: List[str],
    affordance_rules: List[str],
    num_sequences: int = 6,
) -> ActivitySequences:
    """Run the chain and return parsed activity sequences."""
    chain, parser = build_chain()

    return chain.invoke(
        {
            "base_sequence": base_sequence,
            "affordance_rules": affordance_rules,
            "num_sequences": num_sequences,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def print_sequences(result: ActivitySequences) -> None:
    """Print generated sequences in a clear, readable format."""
    print("\nGenerated Activity Sequence Variations:\n")
    for i, sequence in enumerate(result.sequences, start=1):
        print(f"Sequence {i}:")
        for j, step in enumerate(sequence, start=1):
            print(f"  {j}. {step}")
        print("-" * 50)


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Export it before running this script."
        )

    base_sequence = [
        "Open Fridge",
        "Take out vegetables",
        "Close Fridge",
        "Cut Vegetables",
        "Open Fridge",
        "Put back vegetables",
        "Close Fridge",
        "Cook vegetables",
        "Eat vegetables",
    ]

    affordance_rules = [
        "Fridge must be open to interact with contents",
        "Vegetables can be eaten raw or cooked",
        "Fridge must be opened before closed",
        "Humans can cook all vegetables",
        "Humans can eat all vegetables",
    ]

    object = "vegetables"




    result = generate_sequences(
        base_sequence=base_sequence,
        affordance_rules=affordance_rules,
        num_sequences=6,
    )
    print_sequences(result)


if __name__ == "__main__":
    main()
