"""Validation and serialization for Sigrok's private RPG relationship state.

The persisted schema is deliberately numeric/enum-only. Discord transcript text
may be sent to the reflection model, but no free-form model output is accepted.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

DISPOSITIONS = frozenset({"ally", "neutral", "rival", "ignore"})
LAST_VIBES = frozenset(
    {"chill", "friendly", "spicy", "hostile", "correction", "dismissive"}
)

DEFAULT_RELATIONSHIP_STATE: dict[str, Any] = {
    "affinity": 100,
    "trust": 50,
    "disposition": "neutral",
    "roast_level": 1,
    "engagement_weight": 100,
    "last_vibe": None,
    "updated_at": None,
}

_DELTA_LIMITS = {
    "affinity_delta": 10,
    "trust_delta": 5,
    "roast_level_delta": 1,
    "engagement_weight_delta": 10,
}
_MIN_CONFIDENCE = 0.6


def relationship_state_dict(row: Optional[Any]) -> dict[str, Any]:
    if row is None:
        return dict(DEFAULT_RELATIONSHIP_STATE)
    updated_at = getattr(row, "updated_at", None)
    return {
        "affinity": int(getattr(row, "affinity", 100)),
        "trust": int(getattr(row, "trust", 50)),
        "disposition": str(getattr(row, "disposition", "neutral")),
        "roast_level": int(getattr(row, "roast_level", 1)),
        "engagement_weight": int(getattr(row, "engagement_weight", 100)),
        "last_vibe": getattr(row, "last_vibe", None),
        "updated_at": updated_at.isoformat() if updated_at is not None else None,
    }


def _extract_json_object(response: str) -> Optional[dict[str, Any]]:
    text = response.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _plain_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def validate_reflection_response(
    response: str,
    *,
    allowed_user_ids: set[int],
    message_author_by_id: dict[int, int],
) -> Optional[list[dict[str, Any]]]:
    """Parse and constrain a relationship reflection response.

    Returns ``None`` for an invalid top-level response. A valid response with no
    justified changes returns an empty list.
    """

    payload = _extract_json_object(response)
    if payload is None:
        return None
    raw_updates = payload.get("updates")
    if not isinstance(raw_updates, list):
        return None

    best_by_user: dict[int, tuple[float, dict[str, Any]]] = {}
    for raw in raw_updates:
        if not isinstance(raw, dict):
            continue
        user_id = _plain_int(raw.get("user_id"))
        if user_id is None or user_id not in allowed_user_ids:
            continue

        confidence_raw = raw.get("confidence")
        if (
            isinstance(confidence_raw, bool)
            or not isinstance(confidence_raw, (int, float))
        ):
            continue
        confidence = max(0.0, min(1.0, float(confidence_raw)))
        if confidence < _MIN_CONFIDENCE:
            continue

        evidence_raw = raw.get("evidence_message_ids")
        if not isinstance(evidence_raw, list):
            continue
        valid_evidence: set[int] = set()
        for raw_message_id in evidence_raw:
            message_id = _plain_int(raw_message_id)
            if (
                message_id is not None
                and message_author_by_id.get(message_id) == user_id
            ):
                valid_evidence.add(message_id)
        if not valid_evidence:
            continue

        normalized: dict[str, Any] = {"user_id": user_id}
        for field, limit in _DELTA_LIMITS.items():
            value = _plain_int(raw.get(field))
            if value is not None:
                normalized[field] = max(-limit, min(limit, value))

        disposition = raw.get("disposition")
        if isinstance(disposition, str) and disposition in DISPOSITIONS:
            normalized["disposition"] = disposition

        last_vibe = raw.get("last_vibe")
        if isinstance(last_vibe, str) and last_vibe in LAST_VIBES:
            normalized["last_vibe"] = last_vibe

        meaningful = any(
            value != 0
            for key, value in normalized.items()
            if key.endswith("_delta")
        ) or any(key in normalized for key in ("disposition", "last_vibe"))
        if not meaningful:
            continue

        previous = best_by_user.get(user_id)
        if previous is None or confidence > previous[0]:
            best_by_user[user_id] = (confidence, normalized)

    return [best_by_user[user_id][1] for user_id in sorted(best_by_user)]
