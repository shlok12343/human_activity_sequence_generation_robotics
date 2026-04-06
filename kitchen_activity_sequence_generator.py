from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from build_process_dag import build_and_render_process_dag
from affordance_generator import generate_affordance_rules

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


def generate_sequences_object(
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


def generate_hazardous_sequences(
    base_sequence: List[str],
    affordance_rules: List[str],
    target_object: str,
    num_sequences: int = 6,
) -> ActivitySequences:
    """
    Generate only hazardous variants of the activity sequence.

    Every generated sequence is intentionally unsafe and should include at least
    one clear safety violation for the target object workflow.
    """
    parser = PydanticOutputParser(pydantic_object=ActivitySequences)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert in kitchen safety risk modeling.\n"
                    "Generate hazardous human activity sequences only.\n"
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
                    "CRITICAL: every sequence must be hazardous in nature.\n"
                    "All steps must stay focused on manipulations around the target object in a kitchen.\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.7)
    chain = prompt | llm | parser
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
    "Place knife on countertop"
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



    base_sequence_oven= [
        "Utensils on countertop",
        "Oven on",
        "Utensils in oven",
        "Oven Used",
        "Oven Off",
        "Utensils removed from Oven",
        "Utensils placed on countertop",
    ]

    affordance_rules_oven = [
        "Oven must be on to reach Oven Used state",
        "Utensils must be in oven before they can be removed from oven",
    ]

    target_object_oven = "oven"


    base_sequence_dishwasher = [
        "Add Dishes to Dishwasher",
        "Dishwasher is on",
        "Add Dishwasher Detergent",
        "Run the dishwasher",
        "remove dishes",
    ]

    affordance_rules_dishwasher = [
        "remove dishes can only be done after the dishes are added to the dishwasher"
    ]

    target_object_dishwasher = "dishwasher"


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

    affordance_rules_liquids = [
        "Liquid can only be removed from fridge or cabinet before it can be poured",
        "Liquid can only be poured after it is removed from fridge or cabinet",
    ]

    target_object_liquids = "liquids"

  
    base_sequence_dairy_ingredients = [
        "Open Fridge",
        "Remove Ingredients",
        "Use Ingredients",
        "Put back the ingredients",
        "Close Fridge",
    ]

    affordance_rules_dairy_ingredients = [
        "Items must be removed before the put back",
        "Fridge cannot be closed until all items are put back",
        "Dairy products cannot be used if they are still inside the fridge",
    ]

    target_object_dairy_ingredients = "dairy_ingredients"


    base_sequence_kettle = [
        "Fill Kettle",
        "Heat On",
        "Pour water",
        "Heat Off",
        "Turn off the kettle",
    ]

    affordance_rules_kettle = [
        "Kettle must be filled before Heat is turned On",
        "Heat must be On to reach Pour water state",
        "Water cannot be poured if the kettle is currently at Heat On state",
        "Sequence cannot terminate until Turn off the kettle is reached",
    ]

    target_object_kettle = "kettle"

    base_sequence_toaster = [
        "Insert Bread",
        "Toaster On",
        "Remove Bread",
        "Power Off Toaster",
        "Turn off Toaster",
    ]

    affordance_rules_toaster = [
        "Bread must be inserted before Toaster is turned On",
        "Toaster must be On to reach Remove Bread state",
        "Toaster cannot be Power Off while Bread is still being toasted",
        "Sequence must reach Turn off Toaster for safety completion",
    ]

    target_object_toaster = "toaster"

    base_sequence_air_fryer = [
        "Air fryer on",
        "Air fryer used",
        "Air fryer is off",
        "Turn off the air fryer",
    ]

    affordance_rules_air_fryer = [
        "Air fryer must be on before it can be used",
        "Air fryer cannot be off while food is still being cooked (used state)",
        "Air fryer must reach the 'is off' state before the final safety turn off",
    ]

    target_object_air_fryer = "air_fryer"

    base_sequence_stove = [
        "Utensils on Stove",
        "Stove On",
        "Stove Used",
        "Stove Off",
        "Utensils removed from Stove",
        "Turn off the stove",
    ]

    affordance_rules_stove = [
        "Utensils must be on Stove before Stove is turned On",
        "Stove must be On to reach Stove Used state",
        "Utensils cannot be removed from Stove while Stove is still On",
        "Sequence must reach Turn off the stove to be physically complete",
    ]

    target_object_stove = "stove"


    # result = generate_sequences_object(
    #     base_sequence=base_sequence_proteins,
    #     affordance_rules=affordance_rules_proteins,
    #     target_object=target_object_proteins,
    #     num_sequences=15,
    # )

    # result = generate_sequences_object(
    #     base_sequence=base_sequence_vegetables,
    #     affordance_rules=affordance_rules_vegetables,
    #     target_object=target_object_vegetables,
    #     num_sequences=15,)
    
    # result = generate_sequences_object(
    #     base_sequence=base_sequence_oven,
    #     affordance_rules=affordance_rules_oven,
    #     target_object=target_object_oven,
    #     num_sequences=15)

    # result = generate_sequences_object(
    #     base_sequence=base_sequence_dishwasher
    #     affordance_rules=affordance_rules_dishwasher,
    #     target_object=target_object_dishwasher,
    #     num_sequences=50)

    # result = generate_hazardous_sequences(
    #     base_sequence=base_sequence_oven,
    #     affordance_rules=affordance_rules_oven,
    #     target_object=target_object_oven,
    #     num_sequences=15)
    
    # result = generate_sequences_object(
    #     base_sequence=base_sequence_knives,
    #     affordance_rules=affordance_rules_knives,
    #     target_object=target_object_knives,
    #     num_sequences=15,)

    liquids_affordance_rules = generate_affordance_rules(
        base_sequence=base_sequence_oven,
        target_object=target_object_oven,
        num_rules=10,
    )


    result = generate_sequences_object(
        base_sequence=base_sequence_liquids,
        affordance_rules=liquids_affordance_rules.affordance_rules,
        target_object=target_object_liquids,
        num_sequences=50,
    )


    print_sequences(result)
    output_path = build_and_render_process_dag(result.sequences, out_name="process_dag")
    print(f"DAG saved to: {output_path}")


if __name__ == "__main__":
    main()
