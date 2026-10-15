from pathlib import Path

from musicinstruct.evaluation import load_predictions
from musicinstruct.generate import TARGET_ITEMS, generate_pilot_dataset
from musicinstruct.schema import Prediction

from midi_agent.runners.reparse import reparse_predictions


def test_reparse_fixes_truncated_transpose_json(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    generate_pilot_dataset(out, target=TARGET_ITEMS)
    manifest = out / "pilot.jsonl"

    lines = manifest.read_text(encoding="utf-8").splitlines()
    item_id = None
    for line in lines:
        if "transpose" in line:
            import json

            item_id = json.loads(line)["item_id"]
            break
    assert item_id

    pred_path = tmp_path / "predictions.jsonl"
    pred_path.write_text(
        Prediction(
            item_id=item_id,
            midi_path=None,
            plan=None,
            metadata={
                "error": "no_plan_returned",
                "raw_response": '{"op": "transpose", "params": {"semitones": 2, "tracks": [0]}}',
            },
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    summary = reparse_predictions(
        manifest,
        pred_path,
        tmp_path / "repaired",
        split=None,
        only_failed=True,
    )
    assert summary["repaired_count"] >= 1
    fixed = load_predictions(tmp_path / "repaired" / "predictions.jsonl")[item_id]
    assert fixed.midi_path is not None
    assert fixed.plan is not None
    assert fixed.plan.op == "transpose"
