"""Prompt templates for plan generation and replanning."""

from __future__ import annotations

import json

from musicinstruct.plan_executor import build_track_context

from musicinstruct.schema import BenchmarkItem

PLAN_PROMPT = """You are a MIDI editing assistant. Given a source MIDI file summary and a natural-language edit instruction, output exactly one JSON object describing the edit to apply.

Supported operations:
- transpose: params {{"semitones": int, "tracks": [int, ...] optional}}
  Use positive semitones for up/raise/higher, negative for down/lower.
  If tracks omitted, apply to all non-drum tracks.
- tempo_scale: params {{"factor": float}}
  Values >1 mean faster, <1 mean slower.
- velocity_scale: params {{"factor": float, "tracks": [int, ...] optional}}
- mute_tracks: params {{"tracks": [int, ...]}}
- program_change: params {{"track": int, "program": int}}
  program is General MIDI program number 0-127.

Important:
- Track indices are 0-based. Only use track indices listed in the source summary.
- Choose the operation that matches the instruction (do not transpose when asked to change instrument program).

Respond with JSON only, no markdown or explanation:
{{"op": "<operation>", "params": {{...}}}}

Source MIDI context (track summary and optional MCP analysis):
{context}

Edit instruction:
{instruction}"""

REPLAN_PROMPT = """You are a MIDI editing assistant. Your previous plan failed validation or did not satisfy the edit requirements. Output a corrected JSON plan.

Supported operations:
- transpose: params {{"semitones": int, "tracks": [int, ...] optional}}
- tempo_scale: params {{"factor": float}}
- velocity_scale: params {{"factor": float, "tracks": [int, ...] optional}}
- mute_tracks: params {{"tracks": [int, ...]}}
- program_change: params {{"track": int, "program": int}}

Important:
- Track indices are 0-based. Only use track indices listed in the source summary.
- Choose the operation that matches the instruction.

Respond with JSON only:
{{"op": "<operation>", "params": {{...}}}}

Source MIDI context (track summary and optional MCP analysis):
{context}

Edit instruction:
{instruction}

Feedback from the last attempt:
{feedback}"""


def _default_perception(item: BenchmarkItem) -> dict:
    return {"track_summary": build_track_context(item.midi_in)}


def build_plan_prompt(item: BenchmarkItem, perception: dict | None = None) -> str:
    context = json.dumps(perception or _default_perception(item), indent=2)
    return PLAN_PROMPT.format(context=context, instruction=item.instruction)


def build_replan_prompt(item: BenchmarkItem, feedback: str, perception: dict | None = None) -> str:
    context = json.dumps(perception or _default_perception(item), indent=2)
    return REPLAN_PROMPT.format(
        context=context,
        instruction=item.instruction,
        feedback=feedback.strip(),
    )
