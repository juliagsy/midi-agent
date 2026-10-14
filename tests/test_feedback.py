from musicinstruct.metrics import ItemScores

from midi_agent.feedback import format_predicate_feedback, format_validation_error


def test_format_validation_error_includes_plan_dump() -> None:
    text = format_validation_error("track 3 out of range 0..2", plan_dump={"op": "transpose"})
    assert "track 3 out of range" in text
    assert "transpose" in text


def test_format_predicate_feedback_lists_failed_predicates() -> None:
    scores = ItemScores(
        item_id="x",
        validity=1.0,
        edit_success=0.5,
        preserve=1.0,
        joint=0.0,
        over_edit=0.0,
        gold_note_f1=0.5,
        plan_match=0.0,
        edit_details=[
            {"name": "pitch_shifted_by", "passed": False, "score": 0.0, "detail": "expected 3"},
        ],
        preserve_details=[],
    )
    text = format_predicate_feedback(scores)
    assert "pitch_shifted_by" in text
    assert "expected 3" in text
