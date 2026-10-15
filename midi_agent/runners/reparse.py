"""Re-parse and re-execute predictions from stored raw LLM responses."""

from __future__ import annotations

import json
from pathlib import Path

from musicinstruct.dataset import load_jsonl, resolve_item_paths
from musicinstruct.evaluation import filter_items, load_predictions, merge_predictions, score_records, write_predictions
from musicinstruct.metrics import score_item
from musicinstruct.plan_executor import execute_plan, normalize_plan, parse_plan_text
from musicinstruct.schema import BenchmarkItem, Prediction

from midi_agent.utils import make_json_safe


def _raw_response_from_prediction(pred: Prediction) -> str | None:
    meta = pred.metadata or {}
    raw = meta.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        return raw
    rounds = meta.get("rounds")
    if isinstance(rounds, list):
        for record in reversed(rounds):
            if isinstance(record, dict):
                raw = record.get("raw_response")
                if isinstance(raw, str) and raw.strip():
                    return raw
    attempts = meta.get("validation_attempts")
    if isinstance(attempts, list):
        for record in reversed(attempts):
            if isinstance(record, dict):
                raw = record.get("raw_response")
                if isinstance(raw, str) and raw.strip():
                    return raw
    return None


def reparse_predictions(
    manifest_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = "test",
    joint_threshold: float = 0.9,
    only_failed: bool = True,
) -> dict:
    """Parse plans from ``raw_response``, execute, and rewrite predictions + report."""
    manifest_path = Path(manifest_path)
    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)

    gold_root = manifest_path.parent
    items_by_id = {
        item.item_id: resolve_item_paths(item, gold_root)
        for item in filter_items(load_jsonl(manifest_path), split=split)
    }
    predictions = load_predictions(predictions_path)
    out_path = output_dir / "predictions.jsonl"
    repaired: list[str] = []
    still_failed: list[str] = []
    skipped = 0

    updated: dict[str, Prediction] = {}
    for item_id, item in items_by_id.items():
        pred = predictions.get(item_id)
        if pred is None:
            continue
        if only_failed and pred.midi_path and not (pred.metadata or {}).get("error"):
            updated[item_id] = pred
            skipped += 1
            continue

        raw = _raw_response_from_prediction(pred)
        if not raw:
            updated[item_id] = pred
            still_failed.append(item_id)
            continue

        try:
            plan = normalize_plan(parse_plan_text(raw), item)
            destination = midi_dir / f"{item_id}_repaired.mid"
            execute_plan(item.midi_in, destination, plan)
            scores = score_item(item, str(destination), predicted_plan=plan, joint_threshold=joint_threshold)
            meta = dict(pred.metadata or {})
            meta.pop("error", None)
            meta["repaired_from_raw"] = True
            meta["reparse_scores"] = make_json_safe(scores.as_dict())
            updated[item_id] = Prediction(
                item_id=item_id,
                midi_path=str(destination.relative_to(output_dir)),
                plan=plan,
                metadata=make_json_safe(meta),
            )
            repaired.append(item_id)
        except (ValueError, OSError, TypeError) as exc:
            meta = dict(pred.metadata or {})
            meta["reparse_error"] = str(exc)
            updated[item_id] = Prediction(
                item_id=pred.item_id,
                midi_path=pred.midi_path,
                plan=pred.plan,
                metadata=make_json_safe(meta),
            )
            still_failed.append(item_id)

    for item_id, pred in predictions.items():
        if item_id not in updated:
            updated[item_id] = pred

    write_predictions(out_path, list(updated.values()))
    scores = score_records(manifest_path, out_path, joint_threshold=joint_threshold, split=split)
    summary = {
        "manifest": str(manifest_path),
        "source_predictions": str(predictions_path),
        "predictions": str(out_path),
        "split": split,
        "repaired_count": len(repaired),
        "repaired_item_ids": repaired[:30],
        "still_failed_count": len(still_failed),
        "skipped_unchanged": skipped,
        "overall": scores["overall"],
        "composition_macro": scores.get("composition_macro"),
        "by_op_family": scores["by_op_family"],
    }
    report_path = output_dir / "reparse_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)
    return summary
