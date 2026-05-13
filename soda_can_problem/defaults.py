"""Default constants for the soda_can_problem prototype."""

DEFAULT_ORIGINAL_OBJECTS: tuple[str, ...] = (
    "soda_can",
    "metal_fork",
    "towel",
    "oven",
    "microwave",
    )

DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"
DEFAULT_TEMPERATURE_DISCOVERY = 0.2
DEFAULT_TEMPERATURE_HAZARD = 0.1
DEFAULT_NUM_AFFORDANCE_RULES = 10
DEFAULT_HAZARD_BATCH_SIZE = 15
