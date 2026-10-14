"""Client protocol for plan generation."""

from __future__ import annotations

from typing import Protocol

from musicinstruct.schema import BenchmarkItem, Plan


class PlanGenerationClient(Protocol):
    def generate_plan(
        self,
        item: BenchmarkItem,
        feedback: str | None = None,
    ) -> tuple[Plan | None, str, dict]: ...
