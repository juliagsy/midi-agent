# Google Colab — midi-agent (Paper 4)

Notebooks to run **tool-augmented MIDI editing agents** on the MIDI-Instruct test split.

Dataset files live on the shared Drive tree (`MyDrive/musicinstruct/data/…`). Agent **predictions and reports** go under `MyDrive/midi-agent/runs/<RUN_ID>/`.

## One runtime per notebook

| Notebook | GPU | Purpose |
|----------|-----|---------|
| `01_setup_and_data` | CPU OK | Clone repos, restore data, oracle smoke |
| `02_audit_gold` | CPU OK | Oracle-gold vs oracle-exec audit (investigation #6) |
| `03_run_agent` | **T4 required** | `midi-agent run` (`oracle`, `g1`, `g2`, `g3`, `g4`) |
| `04_score_agent` | CPU OK | `midi-agent score` |

Set **`DATASET_MODE`** and **`RUN_ID`** consistently:

| Mode | `DATASET_MODE` | `RUN_ID` | Manifest |
|------|----------------|----------|----------|
| Pilot | `"pilot"` | `pilot_v1` | `data/pilot/pilot.jsonl` |
| Real v0.2 | `"real_v0.2"` | `real_v0.2_v1` | `data/v0.2/manifest.jsonl` |

## Drive layout

| Path | Contents |
|------|----------|
| `MyDrive/musicinstruct/data/pilot/` or `…/data/v0.2/` | Shared dataset ([musicinstruct colab 01](https://github.com/juliagsy/musicinstruct/tree/main/colab)) |
| `MyDrive/midi-agent/runs/<RUN_ID>/audit_gold/` | Gold audit (notebook **02**) |
| `MyDrive/midi-agent/runs/<RUN_ID>/agent_<variant>/` | `predictions.jsonl`, `agent_report.json`, `score_report.json` |

## Prerequisites

1. Complete musicinstruct colab **01** once so Drive has `data/pilot` or `data/v0.2`.
2. [Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) + HF read token for **03**.
3. **03**: Runtime → T4 GPU.

## Recommended workflow (real v0.2)

1. **01** (CPU): restore data, oracle smoke on a few items.
2. **02** (CPU): audit; inspect `bucket_counts` before long runs.
3. **03** (T4): smoke `MAX_ITEMS=5`, then full test; `RESUME=True` after disconnect.
4. **04** (CPU): score; compare to plan-executor baseline in `MyDrive/musicinstruct/runs/<RUN_ID>/plan_executor_llama/score_report.json`.

Timing (T4, **1B**, ~180 test items): roughly 2–3 h per variant (`g4` slower). **Do not use 8B on T4** — weights exceed VRAM; use 1B (notebook default) or A100+.

**MCP (notebook 03):** set `VARIANT="g3"` or `USE_MCP=True`. The notebook installs [music21-mcp](https://github.com/cclawton/music21-mcp), `midi-agent[llm,mcp]`, and `--mcp-command python -m music21_mcp.server`. Results sync to `agent_<variant>_mcp/` on Drive. MCP is perception-only; benchmark edits still use `execute_plan`.
