"""Agent runners for oracle, G1, G2, and G4 variants."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from musicinstruct.dataset import load_jsonl, resolve_item_paths
from musicinstruct.evaluation import filter_items, load_predictions, merge_predictions, score_records, write_predictions
from musicinstruct.metrics import ItemScores, score_item
from musicinstruct.plan_executor import execute_plan, normalize_plan, validate_plan, validate_plan_for_source
from musicinstruct.schema import BenchmarkItem, Plan, Prediction

from midi_agent.clients.base import PlanGenerationClient
from midi_agent.feedback import format_predicate_feedback, format_validation_error
from midi_agent.utils import make_json_safe
from midi_agent.variants import VariantConfig


def _validate_plan_for_item(plan: Plan, item: BenchmarkItem) -> None:
    validate_plan(plan)
    validate_plan_for_source(plan, item.midi_in)


def _try_execute(item: BenchmarkItem, plan: Plan, destination: Path) -> str | None:
    try:
        execute_plan(item.midi_in, destination, plan)
    except Exception as exc:  # noqa: BLE001 — per-round execution errors are collected
        return str(exc) or f"{type(exc).__name__}"
    return None


def _score_attempt(
    item: BenchmarkItem,
    midi_path: str,
    plan: Plan,
    joint_threshold: float,
) -> ItemScores:
    return score_item(item, midi_path, predicted_plan=plan, joint_threshold=joint_threshold)


def _round_record(
    round_index: int,
    *,
    plan: Plan | None = None,
    raw_response: str = "",
    client_meta: dict | None = None,
    error: str | None = None,
    scores: ItemScores | None = None,
    validation_replan: int | None = None,
) -> dict:
    record: dict = {"round": round_index}
    if client_meta:
        record.update(client_meta)
    if raw_response:
        record["raw_response"] = raw_response
    if plan is not None:
        record["parsed_plan"] = plan.model_dump()
    if error:
        record["error"] = error
    if validation_replan is not None:
        record["validation_replan"] = validation_replan
    if scores is not None:
        record["scores"] = make_json_safe(scores.as_dict())
    return record


def _propose_validated_plan(
    item: BenchmarkItem,
    client: PlanGenerationClient,
    feedback: str | None,
    max_validation_replans: int,
) -> tuple[Plan | None, str, dict, list[dict], str | None]:
    """Return plan, last raw response, client meta, attempt records, and terminal error."""
    attempts: list[dict] = []
    last_raw = ""
    last_meta: dict = {}
    current_feedback = feedback

    for attempt_index in range(max_validation_replans + 1):
        plan, raw_response, client_meta = client.generate_plan(item, feedback=current_feedback)
        last_raw = raw_response
        last_meta = client_meta
        if plan is None:
            error = "no_plan_returned"
            attempts.append({"validation_attempt": attempt_index, "error": error, "raw_response": raw_response})
            return None, last_raw, last_meta, attempts, error

        plan = normalize_plan(plan, item)
        try:
            _validate_plan_for_item(plan, item)
            attempts.append(
                {
                    "validation_attempt": attempt_index,
                    "parsed_plan": plan.model_dump(),
                    "raw_response": raw_response,
                }
            )
            return plan, last_raw, last_meta, attempts, None
        except ValueError as exc:
            error = str(exc) or f"{type(exc).__name__}"
            attempts.append(
                {
                    "validation_attempt": attempt_index,
                    "error": error,
                    "parsed_plan": plan.model_dump(),
                    "raw_response": raw_response,
                }
            )
            if attempt_index >= max_validation_replans:
                return plan, last_raw, last_meta, attempts, error
            current_feedback = format_validation_error(error, plan_dump=plan.model_dump())

    return None, last_raw, last_meta, attempts, "validation_exhausted"


def predict_agent(
    item: BenchmarkItem,
    client: PlanGenerationClient,
    output_dir: str | Path,
    config: VariantConfig,
) -> Prediction:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "runner": f"midi_agent_{config.name}",
        "variant": config.name,
        "max_validation_replans": config.max_validation_replans,
        "max_verify_rounds": config.max_verify_rounds,
        "rounds": [],
    }

    feedback: str | None = None
    best_prediction: Prediction | None = None
    best_joint = -1.0

    for round_index in range(1, config.max_verify_rounds + 1):
        plan, raw_response, client_meta, validation_attempts, validation_error = _propose_validated_plan(
            item,
            client,
            feedback,
            config.max_validation_replans,
        )
        metadata["rounds"].append(
            _round_record(
                round_index,
                plan=plan,
                raw_response=raw_response,
                client_meta=client_meta,
                error=validation_error,
                validation_replan=len(validation_attempts) - 1 if validation_attempts else None,
            )
        )
        metadata["validation_attempts"] = make_json_safe(validation_attempts)

        if plan is None or validation_error:
            if config.name == "g4" and round_index < config.max_verify_rounds:
                feedback = format_validation_error(
                    validation_error or "no_plan_returned",
                    plan_dump=plan.model_dump() if plan else None,
                )
                continue
            return Prediction(
                item_id=item.item_id,
                midi_path=None,
                plan=plan,
                metadata=make_json_safe(
                    {**metadata, "error": validation_error or "no_plan_returned", "raw_response": raw_response}
                ),
            )

        destination = output_dir / f"{item.item_id}_r{round_index}.mid"
        exec_error = _try_execute(item, plan, destination)
        if exec_error:
            metadata["rounds"][-1]["execution_error"] = exec_error
            if config.name == "g4" and round_index < config.max_verify_rounds:
                feedback = format_validation_error(exec_error, plan_dump=plan.model_dump())
                continue
            return Prediction(
                item_id=item.item_id,
                midi_path=None,
                plan=plan,
                metadata=make_json_safe({**metadata, "error": exec_error, "raw_response": raw_response}),
            )

        scores = _score_attempt(item, str(destination), plan, config.joint_threshold)
        metadata["rounds"][-1]["scores"] = make_json_safe(scores.as_dict())

        prediction = Prediction(
            item_id=item.item_id,
            midi_path=str(destination),
            plan=plan,
            metadata=make_json_safe(metadata),
        )
        if scores.joint >= best_joint:
            best_joint = scores.joint
            best_prediction = prediction

        if config.name != "g4":
            prediction.metadata["stopped_reason"] = "single_pass"
            return prediction

        if scores.joint >= config.joint_threshold:
            prediction.metadata["stopped_reason"] = "joint_threshold_met"
            return prediction

        if round_index < config.max_verify_rounds:
            feedback = format_predicate_feedback(scores, joint_threshold=config.joint_threshold)
            continue

        prediction.metadata["stopped_reason"] = "max_rounds_reached"
        return prediction

    if best_prediction is not None:
        best_prediction.metadata["stopped_reason"] = "best_of_rounds"
        return best_prediction

    return Prediction(
        item_id=item.item_id,
        midi_path=None,
        plan=None,
        metadata=make_json_safe({**metadata, "error": "no_successful_attempt"}),
    )


def run_agent_suite(
    manifest_path: str | Path,
    client: PlanGenerationClient,
    output_dir: str | Path,
    config: VariantConfig,
    split: str | None = "test",
    runner_name: str | None = None,
    max_items: int | None = None,
    offset: int = 0,
    resume: bool = False,
) -> dict:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = manifest_path.parent
    split_items = filter_items(
        [resolve_item_paths(item, root) for item in load_jsonl(manifest_path)],
        split=split,
    )
    pred_path = output_dir / "predictions.jsonl"
    existing_predictions: dict[str, Prediction] = {}
    if resume and pred_path.is_file():
        existing_predictions = load_predictions(pred_path)

    items = list(split_items)
    if resume:
        items = [item for item in items if item.item_id not in existing_predictions]
    if offset:
        items = items[offset:]
    if max_items is not None:
        items = items[:max_items]

    runner = runner_name or f"midi_agent_{config.name}"

    if not items:
        scores = (
            score_records(
                manifest_path,
                pred_path,
                joint_threshold=config.joint_threshold,
                split=split,
            )
            if pred_path.is_file()
            else {"overall": {}, "prediction_coverage": 0.0}
        )
        return {
            "manifest": str(manifest_path),
            "split": split,
            "runner": runner,
            "variant": config.name,
            "batch_count": 0,
            "total_predictions": len(existing_predictions),
            "prediction_coverage": scores.get("prediction_coverage", 0.0),
            "message": "no new items to process",
            "predictions": str(pred_path) if pred_path.is_file() else None,
        }

    midi_dir = output_dir / "midi"
    batch_predictions: list[Prediction] = []
    merged_predictions = dict(existing_predictions)
    try:
        for item in items:
            pred = predict_agent(item, client, midi_dir, config)
            batch_predictions.append(pred)
            merged_predictions = merge_predictions(merged_predictions, [pred])
            write_predictions(pred_path, list(merged_predictions.values()))
    finally:
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()

    scores = score_records(
        manifest_path,
        pred_path,
        joint_threshold=config.joint_threshold,
        split=split,
    )
    failures = [
        {
            "item_id": pred.item_id,
            "error": pred.metadata.get("error"),
            "stopped_reason": pred.metadata.get("stopped_reason"),
            "rounds": len(pred.metadata.get("rounds", [])),
        }
        for pred in batch_predictions
        if pred.metadata.get("error") or pred.midi_path is None
    ]

    summary = {
        "manifest": str(manifest_path),
        "split": split,
        "runner": runner,
        "variant": config.name,
        "variant_config": asdict(config),
        "batch_count": len(items),
        "total_predictions": len(merged_predictions),
        "split_count": len(split_items),
        "prediction_coverage": scores["prediction_coverage"],
        "missing_prediction_ids": scores["missing_prediction_ids"],
        "joint_threshold": config.joint_threshold,
        "predictions": str(pred_path),
        "overall": scores["overall"],
        "by_op_family": scores["by_op_family"],
        "batch_failure_count": len(failures),
        "batch_failures": failures[:20],
    }
    report_path = output_dir / "agent_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)
    return summary
