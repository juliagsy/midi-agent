"""Command-line interface for midi-agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from musicinstruct.evaluation import score_records

from midi_agent.clients.llama import DEFAULT_LLAMA_MODEL, LlamaAgentClient
from midi_agent.clients.oracle import OraclePlanClient
from midi_agent.perception.base import TrackSummaryKnowledge
from midi_agent.perception.mcp import McpKnowledgeProvider, McpServerConfig, McpToolSpec
from midi_agent.runners.agent import run_agent_suite
from midi_agent.variants import VARIANT_DESCRIPTIONS, resolve_variant


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2))


def _build_mcp_config(args: argparse.Namespace) -> McpServerConfig:
    mcp_args = tuple(args.mcp_arg) if args.mcp_arg else ("-m", "music21_mcp.server")
    tools: tuple[McpToolSpec, ...] | None = None
    if args.mcp_tool:
        parsed: list[McpToolSpec] = []
        for raw in args.mcp_tool:
            payload = json.loads(raw)
            parsed.append(McpToolSpec(name=payload["name"], arguments=payload.get("arguments", {})))
        tools = tuple(parsed)
    default_tools = (
        McpToolSpec("parse_midi_file", {"midi_path": "{midi_path}"}),
        McpToolSpec("analyze_key", {}),
    )
    return McpServerConfig(
        command=args.mcp_command,
        args=mcp_args,
        tools=tools or default_tools,
    )


def _build_knowledge(args: argparse.Namespace):
    use_mcp = args.variant == "g3" or args.mcp_command
    if not use_mcp:
        return TrackSummaryKnowledge()
    if not args.mcp_command:
        print(
            "Variant g3 requires an MCP server. Pass --mcp-command, e.g.\n"
            "  --mcp-command python --mcp-arg -m --mcp-arg music21_mcp.server\n"
            "(after installing cclawton/music21-mcp or your fork)",
            file=sys.stderr,
        )
        sys.exit(2)
    return McpKnowledgeProvider(_build_mcp_config(args))


def _build_client(args: argparse.Namespace):
    if args.variant == "oracle":
        return OraclePlanClient()
    return LlamaAgentClient(
        model_id=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        hf_token=args.hf_token,
        knowledge=_build_knowledge(args),
    )


def cmd_run(args: argparse.Namespace) -> int:
    config = resolve_variant(
        args.variant,
        max_rounds=args.max_rounds,
        joint_threshold=args.joint_threshold,
    )
    client = _build_client(args)
    runner_name = args.runner_name
    if runner_name is None and args.variant != "oracle":
        runner_name = f"midi_agent_{config.name}_{Path(args.model).name}"

    summary = run_agent_suite(
        args.dataset,
        client,
        args.output_dir,
        config,
        split=args.split,
        runner_name=runner_name,
        max_items=args.max_items,
        offset=args.offset,
        resume=args.resume,
    )
    _print_json(summary)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    results = score_records(
        args.gold,
        args.predictions,
        joint_threshold=args.joint_threshold,
        split=args.split,
        strict=args.strict,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    _print_json(results)
    return 0


def cmd_variants(_args: argparse.Namespace) -> int:
    _print_json({"variants": VARIANT_DESCRIPTIONS})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="midi-agent", description="Tool-augmented MIDI editing agents")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an agent variant on a MIDI-Instruct manifest")
    run.add_argument("dataset", help="Path to pilot.jsonl or manifest")
    run.add_argument("--variant", choices=list(VARIANT_DESCRIPTIONS), default="oracle")
    run.add_argument("--output-dir", required=True, help="Directory for predictions and report")
    run.add_argument("--split", default="test")
    run.add_argument("--joint-threshold", type=float, default=0.9)
    run.add_argument("--max-rounds", type=int, default=3, help="Verify rounds for g4")
    run.add_argument("--max-items", type=int, default=None)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--runner-name", default=None)
    run.add_argument("--model", default=DEFAULT_LLAMA_MODEL)
    run.add_argument("--device", default="auto")
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--hf-token", default=None)
    run.add_argument(
        "--mcp-command",
        default=None,
        help="Stdio MCP server executable (required for g3; optional enrich g1/g2/g4)",
    )
    run.add_argument(
        "--mcp-arg",
        action="append",
        default=None,
        help="Extra argv for MCP server (default: -m music21_mcp.server)",
    )
    run.add_argument(
        "--mcp-tool",
        action="append",
        default=None,
        help='JSON MCP tool call; default parse_midi_file + analyze_key with {midi_path}',
    )
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="Score predictions with musicinstruct")
    score.add_argument("gold", help="Gold manifest path")
    score.add_argument("predictions", help="Predictions JSONL path")
    score.add_argument("--split", default=None)
    score.add_argument("--joint-threshold", type=float, default=0.9)
    score.add_argument("--strict", action="store_true")
    score.add_argument("--output", default=None)
    score.set_defaults(func=cmd_score)

    variants = sub.add_parser("variants", help="List supported agent variants")
    variants.set_defaults(func=cmd_variants)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
