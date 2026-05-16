"""Step 5: batched Gemini hazard judgments (Flash Lite, timeout, retries, checkpoint)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from soda_can_problem.defaults import (
    DEFAULT_GEMINI_TIMEOUT_SECONDS,
    DEFAULT_HAZARD_BACKOFF_SECONDS,
    DEFAULT_HAZARD_BATCH_SIZE,
    DEFAULT_HAZARD_MAX_ATTEMPTS,
    DEFAULT_HAZARD_MODEL_NAME,
    DEFAULT_TEMPERATURE_HAZARD,
)
from soda_can_problem.gemini_client import build_llm
from soda_can_problem.schemas import HazardAssessmentItem, HazardBatchOutput

CHECKPOINT_FILENAME = "step5_checkpoint.json"
ALL_ASSESSMENTS_FILENAME = "step5_all_assessments.json"
HAZARDOUS_FILENAME = "hazardous_combinations.json"


def _combo_payload(row: Dict[str, Any]) -> str:
    payload = {
        "original_object": row["original_object"],
        "original_object_state": row["original_object_state"],
        "interaction_target": row["interaction_target"],
    }
    return json.dumps(payload, sort_keys=True)


def _backoff_seconds(attempt: int) -> int:
    """Seconds to wait after attempt ``attempt`` fails (1-based)."""
    idx = min(attempt - 1, len(DEFAULT_HAZARD_BACKOFF_SECONDS) - 1)
    return DEFAULT_HAZARD_BACKOFF_SECONDS[idx]


def _invoke_hazard_chain_with_retries(
    chain,
    invoke_args: Dict[str, str],
    *,
    batch_label: str,
) -> HazardBatchOutput:
    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_HAZARD_MAX_ATTEMPTS + 1):
        try:
            return chain.invoke(invoke_args)
        except Exception as exc:  # noqa: BLE001 — retry transient API / parse failures
            last_error = exc
            if attempt >= DEFAULT_HAZARD_MAX_ATTEMPTS:
                break
            wait = _backoff_seconds(attempt)
            print(
                f"  [Gemini hazard] {batch_label} attempt {attempt}/"
                f"{DEFAULT_HAZARD_MAX_ATTEMPTS} failed: {exc!r}. "
                f"Retrying in {wait}s...",
                flush=True,
            )
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def _write_hazard_artifacts(
    output_dir: Path,
    assessments: List[HazardAssessmentItem],
) -> None:
    """Persist full assessments and hazardous-only subset after each batch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dump = [a.model_dump() for a in assessments]
    with open(output_dir / ALL_ASSESSMENTS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)
    hazardous = [a.model_dump() for a in assessments if a.hazardous]
    with open(output_dir / HAZARDOUS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(hazardous, f, indent=2)


def _save_checkpoint(
    output_dir: Path,
    *,
    total_combinations: int,
    batch_size: int,
    hazard_model_name: str,
    completed_batches: int,
    assessments: List[HazardAssessmentItem],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_combinations": total_combinations,
        "batch_size": batch_size,
        "hazard_model_name": hazard_model_name,
        "completed_batches": completed_batches,
        "assessments": [a.model_dump() for a in assessments],
    }
    path = output_dir / CHECKPOINT_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_checkpoint(
    output_dir: Path,
    *,
    total_combinations: int,
    batch_size: int,
    hazard_model_name: str,
) -> tuple[int, List[HazardAssessmentItem]]:
    """
    Return ``(completed_batches, assessments)`` to resume from, or ``(0, [])`` if none.
    """
    path = output_dir / CHECKPOINT_FILENAME
    if not path.is_file():
        return 0, []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("total_combinations") != total_combinations:
        print(
            "  [Gemini hazard] checkpoint ignored (combination count changed).",
            flush=True,
        )
        return 0, []
    if data.get("batch_size") != batch_size:
        print(
            "  [Gemini hazard] checkpoint ignored (batch_size changed).",
            flush=True,
        )
        return 0, []
    if data.get("hazard_model_name") != hazard_model_name:
        print(
            "  [Gemini hazard] checkpoint ignored (hazard model changed).",
            flush=True,
        )
        return 0, []

    completed = int(data.get("completed_batches", 0))
    raw = data.get("assessments") or []
    items = [HazardAssessmentItem.model_validate(row) for row in raw]
    expected = min(completed * batch_size, total_combinations)
    if len(items) != expected:
        print(
            f"  [Gemini hazard] checkpoint ignored "
            f"(expected {expected} assessments, found {len(items)}).",
            flush=True,
        )
        return 0, []

    return completed, items


def evaluate_hazard_batch(
    combinations_batch: List[Dict[str, Any]],
    *,
    model_name: str = DEFAULT_HAZARD_MODEL_NAME,
    timeout_seconds: float = DEFAULT_GEMINI_TIMEOUT_SECONDS,
) -> List[HazardAssessmentItem]:
    """Judge one batch of combinations; returns assessments aligned to batch order."""
    if not combinations_batch:
        return []

    parser = PydanticOutputParser(pydantic_object=HazardBatchOutput)
    lines = []
    for i, row in enumerate(combinations_batch):
        lines.append(f"{i}. {_combo_payload(row)}")
    batch_block = "\n".join(lines)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a kitchen safety analyst. For each numbered scenario, "
                    "the original_object is currently in the state described by "
                    "original_object_state (variable=value facts). Decide whether "
                    "the original_object in that state being paired with, placed near, "
                    "or used together with the interaction_target could be hazardous.\n"
                    "Reply ONLY with JSON matching the format instructions. "
                    "Return exactly one assessment per input line, same order, "
                    "same indices 0..n-1.\n"
                    "original_object_state is the full state snapshot of the "
                    "original_object — use all relevant variables."
                ),
            ),
            (
                "human",
                (
                    "For EACH numbered combination below, answer:\n"
                    '"Is this combination hazardous in a kitchen environment?"\n\n'
                    "{batch_block}\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )
    llm = build_llm(
        model_name=model_name,
        temperature=DEFAULT_TEMPERATURE_HAZARD,
        timeout=timeout_seconds,
        max_retries=0,
    )
    chain = prompt | llm | parser
    invoke_args = {
        "batch_block": batch_block,
        "format_instructions": parser.get_format_instructions(),
    }
    parsed = _invoke_hazard_chain_with_retries(
        chain,
        invoke_args,
        batch_label=f"batch ({len(combinations_batch)} combos)",
    )
    items = list(parsed.items)
    if len(items) != len(combinations_batch):
        raise ValueError(
            f"Hazard batch size mismatch: expected {len(combinations_batch)}, got {len(items)}"
        )
    for item, src in zip(items, combinations_batch, strict=True):
        if item.original_object != src["original_object"]:
            item.original_object = src["original_object"]
        if item.interaction_target != src["interaction_target"]:
            item.interaction_target = src["interaction_target"]
        if item.original_object_state != src["original_object_state"]:
            item.original_object_state = dict(src["original_object_state"])
    return items


def evaluate_all_combinations(
    combinations: List[Dict[str, Any]],
    *,
    batch_size: int = DEFAULT_HAZARD_BATCH_SIZE,
    hazard_model_name: str = DEFAULT_HAZARD_MODEL_NAME,
    timeout_seconds: float = DEFAULT_GEMINI_TIMEOUT_SECONDS,
    output_dir: Optional[Path] = None,
    fresh_hazard: bool = False,
) -> List[HazardAssessmentItem]:
    total = len(combinations)
    num_batches = (total + batch_size - 1) // batch_size if total else 0
    save_dir = output_dir

    completed_batches = 0
    all_items: List[HazardAssessmentItem] = []

    if save_dir and not fresh_hazard:
        completed_batches, all_items = _load_checkpoint(
            save_dir,
            total_combinations=total,
            batch_size=batch_size,
            hazard_model_name=hazard_model_name,
        )
        if completed_batches > 0:
            hazardous_so_far = sum(1 for item in all_items if item.hazardous)
            print(
                f"  [Gemini hazard] resuming from batch {completed_batches + 1}/"
                f"{num_batches} ({len(all_items)} assessments loaded, "
                f"{hazardous_so_far} hazardous so far)",
                flush=True,
            )

    hazardous_so_far = sum(1 for item in all_items if item.hazardous)

    for batch_idx in range(completed_batches + 1, num_batches + 1):
        start = (batch_idx - 1) * batch_size
        end = min(start + batch_size, total)
        print(
            f"  [Gemini hazard] call {batch_idx}/{num_batches} "
            f"(judging combinations {start + 1}-{end} of {total}, "
            f"model={hazard_model_name})...",
            flush=True,
        )
        chunk = combinations[start:end]
        items = evaluate_hazard_batch(
            chunk,
            model_name=hazard_model_name,
            timeout_seconds=timeout_seconds,
        )
        batch_hazardous = sum(1 for item in items if item.hazardous)
        hazardous_so_far += batch_hazardous
        all_items.extend(items)

        if save_dir:
            _save_checkpoint(
                save_dir,
                total_combinations=total,
                batch_size=batch_size,
                hazard_model_name=hazard_model_name,
                completed_batches=batch_idx,
                assessments=all_items,
            )
            _write_hazard_artifacts(save_dir, all_items)
            print(
                f"  [Gemini hazard] checkpoint saved "
                f"({CHECKPOINT_FILENAME}, {len(all_items)} assessments)",
                flush=True,
            )

        print(
            f"  [Gemini hazard] call {batch_idx}/{num_batches} done "
            f"({len(items)} judged, {batch_hazardous} hazardous this batch, "
            f"{hazardous_so_far} hazardous total so far)",
            flush=True,
        )

    if num_batches:
        print(
            f"  [Gemini hazard] finished {num_batches} calls, "
            f"{len(all_items)} assessments, {hazardous_so_far} hazardous",
            flush=True,
        )
    return all_items
