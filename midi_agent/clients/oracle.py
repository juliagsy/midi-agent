"""Oracle client that returns gold plans from the benchmark manifest."""

from __future__ import annotations

from dataclasses import dataclass

from musicinstruct.schema import BenchmarkItem, Plan


@dataclass
class OraclePlanClient:
    def generate_plan(
        self,
        item: BenchmarkItem,
        feedback: str | None = None,
    ) -> tuple[Plan | None, str, dict]:
        del feedback
        plan = item.plan
        raw = plan.model_dump_json() if plan else ""
        return plan, raw, {"client": "oracle"}
