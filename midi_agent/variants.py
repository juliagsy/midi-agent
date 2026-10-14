"""Registered agent variants for Paper 4 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VariantName = Literal["oracle", "g1", "g2", "g3", "g4"]

VARIANT_DESCRIPTIONS: dict[VariantName, str] = {
    "oracle": "Execute gold plan from manifest (tool ceiling)",
    "g1": "Plan-once: NL → JSON plan → deterministic execute",
    "g2": "G1 + pre-execute validation + one replan on schema/track errors",
    "g3": "G1 + MCP knowledge context (music21-mcp) before planning",
    "g4": "G2 + predicate verify loop (keep best joint, up to K rounds)",
}


@dataclass(frozen=True)
class VariantConfig:
    name: VariantName
    max_validation_replans: int = 0
    max_verify_rounds: int = 1
    joint_threshold: float = 0.9


def resolve_variant(name: str, max_rounds: int = 3, joint_threshold: float = 0.9) -> VariantConfig:
    normalized = name.lower().strip()
    if normalized not in VARIANT_DESCRIPTIONS:
        supported = ", ".join(VARIANT_DESCRIPTIONS)
        raise ValueError(f"unknown variant {name!r}; supported: {supported}")
    if normalized == "oracle":
        return VariantConfig(name="oracle", max_verify_rounds=1, joint_threshold=joint_threshold)
    if normalized == "g1":
        return VariantConfig(name="g1", max_verify_rounds=1, joint_threshold=joint_threshold)
    if normalized == "g2":
        return VariantConfig(
            name="g2",
            max_validation_replans=1,
            max_verify_rounds=1,
            joint_threshold=joint_threshold,
        )
    if normalized == "g3":
        return VariantConfig(name="g3", max_verify_rounds=1, joint_threshold=joint_threshold)
    return VariantConfig(
        name="g4",
        max_validation_replans=1,
        max_verify_rounds=max(1, max_rounds),
        joint_threshold=joint_threshold,
    )
