from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from build_process_dag import build_and_render_process_dag

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
                    "Target object:\n{target_object}\n\n"
                    "World knowledge / affordance rules:\n{affordance_rules}\n\n"
                    "Generate exactly {num_sequences} alternative sequences.\n"
                    "Include a mix of:\n"
                    "- fewer steps versions\n"
                    "- Hazardous versions\n"
                    "- Alternate-order versions\n\n"
                    "- edge cases versions\n"
                    "All steps must stay focused on manipulations around the target object in a kitchen.\n\n"
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
    target_object: str,
    num_sequences: int = 6,
) -> ActivitySequences:
    """Run the chain and return parsed activity sequences."""
    chain, parser = build_chain()

    return chain.invoke(
        {
            "base_sequence": base_sequence,
            "affordance_rules": affordance_rules,
            "target_object": target_object,
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

    base_sequence_vegetables = [
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

    affordance_rules_vegetables = [
        "Fridge must be open to interact with contents",
        "Vegetables can be eaten raw or cooked",
        "Fridge must be opened before closed",
        "Humans can cook all vegetables",
        "Humans can eat all vegetables",
    ]

    target_object_vegetables = "vegetables"


    base_sequence_knives = [
    "open drawer",
    "Remove knife from block",
    "Place vegetable on board",
    "Slice vegetable",
    "Wipe blade",
    "Wash knife",
    "Dry knife",
    "Return knife to block" 
    "close drawer",
    ]

    affordance_rules_knives = [
        "Drawer must be open to reach knives",
        "Drawer must be open before it can be closed",
        "Knife must be removed from storage before slicing",
        "You can not dry kife before washing it"
    ]

    target_object_knives = "knives"

    base_sequence_proteins = [
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
    "eat protein"
    ]

    affordance_rules_proteins = [
        "Protien can not be cut before being removed from packaging"
        "Protien can not be removed from packaging before being removed from fridge"
    ]
    target_object_proteins = "protein"




    # result = generate_sequences(
    #     base_sequence=base_sequence_proteins,
    #     affordance_rules=affordance_rules_proteins,
    #     target_object=target_object_proteins,
    #     num_sequences=15,
    # )

    result = generate_sequences(
        base_sequence=base_sequence_vegetables,
        affordance_rules=affordance_rules_vegetables,
        target_object=target_object_vegetables,
        num_sequences=15,)
    
    # result = generate_sequences(
    #     base_sequence=base_sequence_knives,
    #     affordance_rules=affordance_rules_knives,
    #     target_object=target_object_knives,
    #     num_sequences=15,)




    print_sequences(result)
    output_path = build_and_render_process_dag(result.sequences, out_name="process_dag")
    print(f"DAG saved to: {output_path}")


if __name__ == "__main__":
    main()
