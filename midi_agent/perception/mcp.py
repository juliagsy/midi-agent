"""MCP-backed knowledge (e.g. music21-mcp) for richer MIDI context."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from musicinstruct.plan_executor import build_track_context
from musicinstruct.schema import BenchmarkItem

from midi_agent.perception.base import TrackSummaryKnowledge

_MIDI_PATH = "{midi_path}"


@dataclass(frozen=True)
class McpToolSpec:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpServerConfig:
    """Stdio MCP server launch configuration."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    tools: tuple[McpToolSpec, ...] = (
        McpToolSpec("parse_midi_file", {"midi_path": _MIDI_PATH}),
        McpToolSpec("analyze_key", {}),
    )


def default_music21_mcp_config(command: str = "python", *extra_args: str) -> McpServerConfig:
    """Preset aligned with cclawton/music21-mcp (MIDI parse + key analysis)."""
    args = ("-m", "music21_mcp.server", *extra_args)
    return McpServerConfig(command=command, args=args)


def _substitute_arguments(template: dict[str, Any], midi_path: str) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, str):
            resolved[key] = value.replace(_MIDI_PATH, midi_path)
        else:
            resolved[key] = value
    return resolved


def _tool_result_payload(result: Any) -> Any:
    chunks: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    combined = "\n".join(chunks).strip()
    if not combined:
        return None
    try:
        return json.loads(combined)
    except json.JSONDecodeError:
        return {"text": combined}


def _normalize_tool_name(name: str) -> str:
    return re.sub(r"^mcp_", "", name)


class McpKnowledgeProvider:
    """Call configured MCP tools once per item; always includes track_summary."""

    def __init__(self, config: McpServerConfig, *, fallback_on_error: bool = True) -> None:
        self._config = config
        self._fallback_on_error = fallback_on_error
        self._summary = TrackSummaryKnowledge()
        self._session: Any = None
        self._stdio_ctx: Any = None
        self._read_write: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    async def _connect(self) -> None:
        if self._session is not None:
            return
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ImportError(
                "MCP knowledge requires the optional `mcp` package. "
                "Install with: uv pip install -e '.[mcp]'"
            ) from exc

        params = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            env=self._config.env,
        )
        self._stdio_ctx = stdio_client(params)
        self._read_write = await self._stdio_ctx.__aenter__()
        read, write = self._read_write
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

    async def _disconnect(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._stdio_ctx is not None:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None
        self._read_write = None

    async def _call_tools_async(self, midi_path: str) -> dict[str, Any]:
        await self._connect()
        assert self._session is not None
        resolved_path = str(Path(midi_path).resolve())
        tool_results: dict[str, Any] = {}
        errors: list[str] = []

        listed = await self._session.list_tools()
        available = {_normalize_tool_name(t.name): t.name for t in listed.tools}

        for spec in self._config.tools:
            canonical = _normalize_tool_name(spec.name)
            tool_name = available.get(canonical) or available.get(spec.name) or spec.name
            if canonical not in available and spec.name not in available.values():
                errors.append(f"tool_not_found:{spec.name}")
                continue
            arguments = _substitute_arguments(spec.arguments, resolved_path)
            try:
                result = await self._session.call_tool(tool_name, arguments)
                payload = _tool_result_payload(result)
                tool_results[canonical] = payload if payload is not None else {"ok": True}
            except Exception as exc:  # noqa: BLE001 — collect per-tool failures
                errors.append(f"{spec.name}:{exc}")
                if not self._fallback_on_error:
                    raise

        if errors:
            tool_results["_errors"] = errors
        return tool_results

    def gather(self, item: BenchmarkItem) -> dict:
        loop = self._ensure_loop()
        perception: dict[str, Any] = {"track_summary": self._summary.gather(item)["track_summary"]}
        try:
            perception["mcp"] = loop.run_until_complete(self._call_tools_async(item.midi_in))
        except Exception as exc:  # noqa: BLE001
            if not self._fallback_on_error:
                raise
            perception["mcp"] = {"_errors": [str(exc)]}
        return perception

    def close(self) -> None:
        if self._loop is None:
            return
        if self._session is not None or self._stdio_ctx is not None:
            self._loop.run_until_complete(self._disconnect())
        self._loop.close()
        self._loop = None
