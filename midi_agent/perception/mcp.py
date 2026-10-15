"""MCP-backed knowledge (e.g. music21-mcp) for richer MIDI context."""

from __future__ import annotations

import asyncio
import json
import re
import threading
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
        McpToolSpec("parse_midi_file", {"file_path": _MIDI_PATH}),
        McpToolSpec("analyze_key", {"file_path": _MIDI_PATH}),
    )


def default_music21_mcp_config(command: str = "python", *extra_args: str) -> McpServerConfig:
    """Preset aligned with cclawton/music21-mcp (file_path + key analysis)."""
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
    """Call configured MCP tools once per item; always includes track_summary.

    MCP/anyio requires connect and disconnect on the same asyncio task. A dedicated
    background thread owns the event loop and stdio session for the benchmark run.
    """

    def __init__(self, config: McpServerConfig, *, fallback_on_error: bool = True) -> None:
        self._config = config
        self._fallback_on_error = fallback_on_error
        self._summary = TrackSummaryKnowledge()
        self._session: Any = None
        self._stdio_ctx: Any = None
        self._read_write: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._closed = False
        self._lock = threading.Lock()

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
        if self._session is None:
            raise RuntimeError("MCP session is not connected")
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

    def _worker_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            try:
                loop.run_until_complete(self._connect())
            except BaseException as exc:
                self._start_error = exc
                return
            finally:
                self._ready.set()
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(self._disconnect())
            except Exception:
                pass
            loop.close()
            self._loop = None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP knowledge provider is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._worker_main,
                    name="mcp-knowledge",
                    daemon=True,
                )
                self._thread.start()
        if not self._ready.wait(timeout=120):
            raise TimeoutError("MCP server did not become ready within 120s")
        if self._start_error is not None:
            raise RuntimeError(f"MCP server failed to start: {self._start_error}") from self._start_error

    def gather(self, item: BenchmarkItem) -> dict:
        perception: dict[str, Any] = {"track_summary": self._summary.gather(item)["track_summary"]}
        try:
            self._ensure_worker()
            assert self._loop is not None
            future = asyncio.run_coroutine_threadsafe(
                self._call_tools_async(item.midi_in),
                self._loop,
            )
            perception["mcp"] = future.result(timeout=300)
        except Exception as exc:  # noqa: BLE001
            if not self._fallback_on_error:
                raise
            perception["mcp"] = {"_errors": [str(exc)]}
        return perception

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=60)
