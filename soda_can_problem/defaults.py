"""Default constants for the soda_can_problem prototype."""

DEFAULT_ORIGINAL_OBJECTS: tuple[str, ...] = (
    "soda_can",
    "metal_fork",
    "oven",
    "microwave",
    "metal_spoon",
    "plastic_container",
    )

DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"
DEFAULT_HAZARD_MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_TEMPERATURE_DISCOVERY = 0.2
DEFAULT_TEMPERATURE_HAZARD = 0.1
DEFAULT_NUM_AFFORDANCE_RULES = 10
DEFAULT_HAZARD_BATCH_SIZE = 15
DEFAULT_GEMINI_TIMEOUT_SECONDS = 180
DEFAULT_HAZARD_MAX_ATTEMPTS = 5
DEFAULT_HAZARD_BACKOFF_SECONDS: tuple[int, ...] = (15, 45, 90, 120, 120)
