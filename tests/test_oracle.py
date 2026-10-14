from pathlib import Path

from musicinstruct.evaluation import score_records
from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset

from midi_agent.clients.oracle import OraclePlanClient
from midi_agent.runners.agent import run_agent_suite
from midi_agent.variants import resolve_variant


def test_oracle_achieves_perfect_joint_on_test_split(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    manifest = out / "pilot.jsonl"
    config = resolve_variant("oracle")
    summary = run_agent_suite(
        manifest,
        OraclePlanClient(),
        tmp_path / "results",
        config,
        split="test",
    )
    assert summary["overall"]["joint"] == 1.0
    assert summary["overall"]["plan_match"] == 1.0
    assert summary["prediction_coverage"] == 1.0

    scores = score_records(manifest, summary["predictions"], split="test")
    assert scores["overall"]["joint"] == 1.0
