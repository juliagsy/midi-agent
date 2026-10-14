"""Track-summary and composable knowledge providers."""

from __future__ import annotations

from typing import Protocol

from musicinstruct.plan_executor import build_track_context
from musicinstruct.schema import BenchmarkItem


class KnowledgeProvider(Protocol):
    def gather(self, item: BenchmarkItem) -> dict: ...

    def close(self) -> None: ...


class TrackSummaryKnowledge:
    """Default perception: musicinstruct track summary (no MCP)."""

    def gather(self, item: BenchmarkItem) -> dict:
        return {"track_summary": build_track_context(item.midi_in)}

    def close(self) -> None:
        return None
