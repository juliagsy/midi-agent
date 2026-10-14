"""Hugging Face Llama client with optional replan feedback."""

from __future__ import annotations

from dataclasses import dataclass, field

from musicinstruct.llm_client import resolve_device, resolve_hf_token
from musicinstruct.plan_executor import parse_plan_text
from musicinstruct.schema import BenchmarkItem, Plan

from midi_agent.perception.base import KnowledgeProvider, TrackSummaryKnowledge
from midi_agent.prompts import build_plan_prompt, build_replan_prompt

DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-8B-Instruct"


@dataclass
class LlamaAgentClient:
    model_id: str = DEFAULT_LLAMA_MODEL
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    hf_token: str | None = None
    knowledge: KnowledgeProvider = field(default_factory=TrackSummaryKnowledge)

    def __post_init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._resolved_device: str | None = None
        self._hf_token = resolve_hf_token(self.hf_token)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._resolved_device = resolve_device(self.device)
        dtype = torch.float16 if self._resolved_device == "cuda" else torch.float32
        load_kwargs = {"token": self._hf_token} if self._hf_token else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **load_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            **load_kwargs,
        )
        self._model.to(self._resolved_device)
        self._model.eval()

    def _generate(self, user_prompt: str) -> str:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._resolved_device is not None

        import torch

        messages = [
            {
                "role": "system",
                "content": "You convert MIDI edit instructions into strict JSON plans.",
            },
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self._resolved_device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._resolved_device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **generate_kwargs)

        new_tokens = output_ids[0, input_ids.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def shutdown(self) -> None:
        self.knowledge.close()

    def generate_plan(
        self,
        item: BenchmarkItem,
        feedback: str | None = None,
    ) -> tuple[Plan | None, str, dict]:
        perception = self.knowledge.gather(item)
        prompt = (
            build_replan_prompt(item, feedback, perception=perception)
            if feedback
            else build_plan_prompt(item, perception=perception)
        )
        raw = self._generate(prompt)
        meta = {
            "client": "llama",
            "model_id": self.model_id,
            "device": self._resolved_device or self.device,
            "had_feedback": feedback is not None,
            "perception_keys": sorted(perception.keys()),
        }
        if "mcp" in perception:
            mcp_block = perception["mcp"]
            if isinstance(mcp_block, dict) and mcp_block.get("_errors"):
                meta["mcp_errors"] = mcp_block["_errors"]
        try:
            plan = parse_plan_text(raw)
        except ValueError:
            return None, raw, meta
        return plan, raw, meta
