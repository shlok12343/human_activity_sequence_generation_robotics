"""Pydantic schemas for Gemini structured I/O."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class PerObjectInteractions(BaseModel):
    object_name: str = Field(description="Canonical object slug, e.g. soda_can.")
    interacts_with: List[str] = Field(
        description=(
            "Kitchen objects, appliances, locations, or items this object "
            "normally interacts with."
        ),
    )


class InteractionDiscoveryOutput(BaseModel):
    objects: List[PerObjectInteractions]


class ProposedActivitySequence(BaseModel):
    """Gemini fallback when no curated pipeline mapping exists."""

    target_object: str = Field(
        description="Short noun for state-graph target, e.g. microwave.",
    )
    base_sequence: List[str] = Field(
        description="Ordered activity phrases suitable for affordance extraction.",
        min_length=3,
    )


class HazardAssessmentItem(BaseModel):
    """One combination assessment (matches prompt batch row)."""

    original_object: str
    original_object_state: Dict[str, str]
    interaction_target: str
    hazardous: bool
    hazard_type: str
    explanation: str
    recommended_action: str


class HazardBatchOutput(BaseModel):
    items: List[HazardAssessmentItem]


class GraphBuildRecord(BaseModel):
    """Serialized outcome for one interaction entity."""

    interaction_raw: str
    normalized_key: str
    slug: str
    status: str
    reason: str = ""
    target_object: str = ""
    graph_path: str = ""
    validation_errors: List[str] = Field(default_factory=list)
    node_count: int = 0
