from dataclasses import dataclass, field
from pathlib import Path

from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset
from musicinstruct.schema import BenchmarkItem, Plan
from musicinstruct.transforms import make_seed_midi

from midi_agent.runners.agent import predict_agent
from midi_agent.variants import resolve_variant


@dataclass
class SequencePlanClient:
    plans: list[Plan | None]
    calls: int = 0
    feedback_seen: list[str | None] = field(default_factory=list)

    def generate_plan(
        self,
        item: BenchmarkItem,
        feedback: str | None = None,
    ) -> tuple[Plan | None, str, dict]:
        del item
        self.feedback_seen.append(feedback)
        index = min(self.calls, len(self.plans) - 1)
        self.calls += 1
        plan = self.plans[index]
        raw = plan.model_dump_json() if plan else ""
        return plan, raw, {"client": "sequence"}


def test_g2_replans_after_invalid_track_index(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "source.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="bad_tracks",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="program_change",
        instruction="Change track 0 to GM program 25.",
        midi_in=str(source),
        plan=Plan(op="program_change", params={"track": 0, "program": 25}),
    )
    client = SequencePlanClient(
        plans=[
            Plan(op="transpose", params={"semitones": 3, "tracks": [0, 2, 3]}),
            Plan(op="program_change", params={"track": 0, "program": 25}),
        ]
    )
    config = resolve_variant("g2")
    pred = predict_agent(item, client, tmp_path / "midi", config)
    assert pred.midi_path is not None
    assert pred.plan is not None
    assert pred.plan.op == "program_change"
    assert client.calls == 2
    assert client.feedback_seen[1] is not None
    assert "track 3 out of range" in client.feedback_seen[1]


def test_g1_does_not_replan_on_validation_error(tmp_path: Path) -> None:
    source = make_seed_midi(tmp_path / "source.mid", seed_index=0)
    item = BenchmarkItem(
        item_id="bad_tracks",
        composition_id="seed_00",
        gold_mode="unique",
        op_family="program_change",
        instruction="Change track 0 to GM program 25.",
        midi_in=str(source),
    )
    client = SequencePlanClient(
        plans=[Plan(op="transpose", params={"semitones": 3, "tracks": [0, 2, 3]})]
    )
    config = resolve_variant("g1")
    pred = predict_agent(item, client, tmp_path / "midi", config)
    assert pred.midi_path is None
    assert client.calls == 1


def test_g2_suite_runs_on_pilot_subset(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    manifest = out / "pilot.jsonl"
    from musicinstruct.dataset import load_jsonl, resolve_item_paths

    items = [
        resolve_item_paths(item, out)
        for item in load_jsonl(manifest)
        if item.split == "test" and item.plan is not None
    ][:3]
    client = SequencePlanClient(plans=[item.plan for item in items])
    config = resolve_variant("g2")
    preds = []
    for item in items:
        from midi_agent.runners.agent import predict_agent

        preds.append(predict_agent(item, client, tmp_path / "midi", config))
    assert all(pred.midi_path is not None for pred in preds)
