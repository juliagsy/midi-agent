"""Format validation and predicate failures for agent replanning."""

from __future__ import annotations

from musicinstruct.metrics import ItemScores


def format_validation_error(error: str, plan_dump: dict | None = None) -> str:
    lines = [f"Plan validation failed: {error}"]
    if plan_dump:
        lines.append(f"Rejected plan: {plan_dump}")
    lines.append("Fix track indices, operation choice, or parameter types and try again.")
    return "\n".join(lines)


def format_predicate_feedback(scores: ItemScores, joint_threshold: float = 0.9) -> str:
    lines = [
        f"Edit result did not pass verification (joint threshold {joint_threshold}).",
        f"edit_success={scores.edit_success:.3f}, preserve={scores.preserve:.3f}, "
        f"over_edit={scores.over_edit:.3f}",
    ]
    failed_edit = [d for d in scores.edit_details if not d.get("passed")]
    failed_preserve = [d for d in scores.preserve_details if not d.get("passed")]
    if failed_edit:
        lines.append("Must-change predicates that failed:")
        for detail in failed_edit:
            lines.append(f"  - {detail['name']}: {detail.get('detail', '')}")
    if failed_preserve:
        lines.append("Must-preserve predicates that failed:")
        for detail in failed_preserve:
            lines.append(f"  - {detail['name']}: {detail.get('detail', '')}")
    lines.append("Propose a corrected plan that satisfies the instruction without breaking preservation.")
    return "\n".join(lines)


def format_round_summary(round_index: int, error: str | None, scores: ItemScores | None) -> str:
    if error:
        return f"Round {round_index}: validation/execution error — {error}"
    if scores is None:
        return f"Round {round_index}: no output produced"
    return (
        f"Round {round_index}: edit_success={scores.edit_success:.3f}, "
        f"preserve={scores.preserve:.3f}, joint={scores.joint:.0f}"
    )
