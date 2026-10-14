# midi-agent

Tool-augmented MIDI editing agents for general-purpose LLMs (Paper 4).

Builds on [musicinstruct](https://github.com/juliagsy/musicinstruct) for the MIDI-Instruct benchmark, deterministic edit tools, and dual-metric scoring.

## Variants

| Variant | Description |
|---------|-------------|
| `oracle` | Execute gold plan from manifest (tool ceiling) |
| `g1` | Plan-once: NL → JSON plan → deterministic execute |
| `g2` | G1 + pre-execute validation + one replan on schema/track errors |
| `g3` | G1 + **MCP knowledge** (music21-mcp) prepended to plan context |
| `g4` | G2 + predicate verify loop (up to K replans, keep best joint) |

## Install

```bash
cd /path/to/midi-agent
uv pip install -e ../musicinstruct
uv pip install -e ".[llm,dev]"
```

## Quick start

Oracle ceiling on the pilot test split:

```bash
midi-agent run data/pilot/pilot.jsonl \
  --variant oracle \
  --output-dir results/oracle
```

Plan-once with Llama 3.2 8B:

```bash
midi-agent run ../musicinstruct/data/pilot/pilot.jsonl \
  --variant g1 \
  --model meta-llama/Llama-3.2-8B-Instruct \
  --output-dir results/g1_llama8b \
  --split test
```

Verify loop (G4, max 3 rounds):

```bash
midi-agent run ../musicinstruct/data/pilot/pilot.jsonl \
  --variant g4 \
  --model meta-llama/Llama-3.2-8B-Instruct \
  --max-rounds 3 \
  --output-dir results/g4_llama8b
```

Score existing predictions:

```bash
midi-agent score data/pilot/pilot.jsonl results/g4_llama8b/predictions.jsonl
```

### MCP knowledge (g3)

Install an MCP server (e.g. [cclawton/music21-mcp](https://github.com/cclawton/music21-mcp)) and the client extra:

```bash
uv pip install -e ".[llm,mcp]"
```

Run **g3** (plan-once with MCP analysis in the prompt). Action and scoring still use `musicinstruct` `execute_plan` + grader — MCP is perception only.

```bash
midi-agent run ../musicinstruct/data/pilot/pilot.jsonl \
  --variant g3 \
  --mcp-command python \
  --mcp-arg -m \
  --mcp-arg music21_mcp.server \
  --model meta-llama/Llama-3.2-8B-Instruct \
  --output-dir results/g3_mcp \
  --split test \
  --max-items 5
```

You can pass `--mcp-command` on `g1`/`g2`/`g4` as well. Default tools: `parse_midi_file` and `analyze_key` with `{midi_path}` substituted per item. Override with `--mcp-tool '{"name":"analyze_key","arguments":{}}'`.

## Google Colab

Self-contained notebooks under [`colab/`](colab/README.md): setup + data restore, gold audit, T4 agent runs, and scoring. Dataset on Drive is shared with [musicinstruct](https://github.com/juliagsy/musicinstruct); artifacts go to `MyDrive/midi-agent/runs/<RUN_ID>/`.

## Layout

```
midi_agent/
  cli.py           # midi-agent CLI
  prompts.py       # plan + replan prompts
  feedback.py      # validation / predicate feedback for replanning
  variants.py      # variant registry
  clients/         # LLM + oracle clients
  runners/         # per-variant predict + suite runners
```
