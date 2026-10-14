from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset
from musicinstruct.schema import BenchmarkItem

from midi_agent.perception.base import TrackSummaryKnowledge
from midi_agent.prompts import build_plan_prompt


class FakeMcpKnowledge:
    def gather(self, item: BenchmarkItem) -> dict:
        return {
            "track_summary": {"note_count": 1},
            "mcp": {"analyze_key": {"tonic": "C", "mode": "major"}},
        }

    def close(self) -> None:
        return None


def test_build_plan_prompt_includes_mcp_block(tmp_path):
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    manifest = out / "pilot.jsonl"
    line = manifest.read_text(encoding="utf-8").splitlines()[0]
    item = BenchmarkItem.model_validate_json(line)
    item = item.model_copy(update={"midi_in": str(out / item.midi_in)})

    prompt = build_plan_prompt(item, perception=FakeMcpKnowledge().gather(item))
    assert "analyze_key" in prompt
    assert "track_summary" in prompt
    assert '"tonic": "C"' in prompt


def test_track_summary_knowledge_has_tracks(tmp_path):
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    line = (out / "pilot.jsonl").read_text(encoding="utf-8").splitlines()[0]
    item = BenchmarkItem.model_validate_json(line)
    item = item.model_copy(update={"midi_in": str(out / item.midi_in)})

    perception = TrackSummaryKnowledge().gather(item)
    assert "tracks" in perception["track_summary"]
