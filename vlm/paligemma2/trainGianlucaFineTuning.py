#!/usr/bin/env python3
"""
Fine-tuning PaliGemma 2 mix-224 su FER+ per Group Emotion Recognition.
Adapted from Orazio's finetune_minicpmv_emotion_qlora.py — same structure,
same hyperparameters that produced strong results (F1 0.92 happiness).

Key differences vs previous trainGianluca.py:
  - CLI argument parser (same style as Orazio's script)
  - LR: 1e-5 (was 1e-4)
  - LoRA r=4, alpha=8 (was r=16, alpha=32)
  - max_grad_norm: 0.3 (was 1.0)
  - warmup_ratio: 0.05 (was fixed warmup_steps)
  - paged_adamw_8bit optimizer (more stable on QLoRA)
  - sample_fraction: 1/3 stratified (matches Orazio)
  - Group output schema: {"subjects":[{"id":1,"emotion":"<e>"}],"group_emotion":"<e>"}
  - Prompt includes explicit emotion list (matches evaluate script)
  - Loss guardrails (NaN/Inf stop, LR reduction on explosion)
  - Training summary JSON saved at end

Usage:
  python trainGianluca.py \
    --train_json /workspace/datasets/jsonl/train.jsonl \
    --eval_json  /workspace/datasets/jsonl/val.jsonl \
    --output_dir /workspace/gianluca/paligemma2_fer_lora \
    --num_train_epochs 1 \
    --sample_fraction 0.3333333333 \
    --learning_rate 1e-5 \
    --lora_r 4 --lora_alpha 8

S3 upload after training:
  aws s3 sync /workspace/gianluca/paligemma2_fer_lora \
    s3://dimesvlm-data/models/paligemma2_fer_lora --region eu-west-1
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "google/paligemma2-3b-mix-224"
DEFAULT_EMOTIONS = (
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (mirrors Orazio's argument structure)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune PaliGemma 2 mix-224 with QLoRA on FERPlus emotion data."
    )

    # Paths
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--eval_json", default=None)
    parser.add_argument("--output_dir", default="/workspace/gianluca/paligemma2_fer_lora")
    parser.add_argument("--training_summary_json", default=None,
                        help="Where to save training summary. Defaults to <output_dir>/training_summary.json.")
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Trainer checkpoint directory to resume from.")

    # Model
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)

    # Data
    parser.add_argument("--sample_fraction", type=float, default=1.0 / 3.0,
                        help="Fraction sampled independently from each emotion class.")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="Val ratio carved from train when --eval_json is not provided.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emotion_labels", default=",".join(DEFAULT_EMOTIONS))
    parser.add_argument("--target_schema", choices=("group", "primary"), default="group",
                        help="group: subjects+group_emotion. primary: primary_emotion only.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--target_answer_max_length", type=int, default=64)

    # Training hyperparameters — defaults match Orazio's successful run
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)

    # LoRA — Orazio used r=4, alpha=8 successfully
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Guardrails (mirrors Orazio's script)
    parser.add_argument("--stop_on_nan_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss_guard_min_logs", type=int, default=3)
    parser.add_argument("--loss_explosion_factor", type=float, default=3.0)
    parser.add_argument("--loss_explosion_abs_threshold", type=float, default=10.0)
    parser.add_argument("--lr_reduction_factor", type=float, default=0.5)
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_reduction_cooldown_steps", type=int, default=50)
    parser.add_argument("--max_lr_reductions", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=1)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--load_best_model_at_end",
                        action=argparse.BooleanOptionalAction, default=True)

    # Infra
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--s3_bucket", default="s3://dimesvlm-data/models/paligemma2_fer_lora")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_emotion(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty.")
    if text[0] == "[":
        return json.loads(text)
    records = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
    return records


def extract_emotion(record: Dict[str, Any]) -> str:
    raw = record.get("output", {})
    if isinstance(raw, str):
        raw = json.loads(raw)
    emotion = (
        raw.get("primary_emotion")
        or raw.get("group_emotion")
        or raw.get("emotion")
        or raw.get("label")
    )
    if emotion is None:
        emotion = Path(str(record.get("image", ""))).parent.name
    return normalize_emotion(str(emotion))


def validate_and_annotate(
    records: Sequence[Dict[str, Any]],
    allowed: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_set = {normalize_emotion(e) for e in allowed}
    out, skipped_img, skipped_label = [], 0, Counter()
    for rec in records:
        if not Path(str(rec.get("image", ""))).exists():
            skipped_img += 1
            continue
        emo = extract_emotion(rec)
        if emo not in allowed_set:
            skipped_label[emo] += 1
            continue
        item = dict(rec)
        item["_emotion"] = emo
        out.append(item)
    if skipped_img:
        print(f"  Skipped {skipped_img} records with missing images.")
    if skipped_label:
        print(f"  Skipped labels: {dict(skipped_label)}")
    if not out:
        raise ValueError("No usable records after filtering.")
    return out


def stratified_sample(
    records: Sequence[Dict[str, Any]],
    fraction: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Keep fraction from each class independently — same as Orazio's approach."""
    if not 0 < fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1].")
    rng = random.Random(seed)
    groups: Dict[str, List] = defaultdict(list)
    for rec in records:
        groups[rec["_emotion"]].append(rec)
    sampled = []
    for emo, group in sorted(groups.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        keep = max(1, math.floor(len(shuffled) * fraction))
        sampled.extend(shuffled[:keep])
        print(f"  {emo:12s}: kept {keep}/{len(shuffled)}")
    rng.shuffle(sampled)
    return sampled


def stratified_train_eval_split(
    records: Sequence[Dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> Tuple[List, List]:
    if eval_ratio <= 0:
        return list(records), []
    rng = random.Random(seed)
    groups: Dict[str, List] = defaultdict(list)
    for rec in records:
        groups[rec["_emotion"]].append(rec)
    train_recs, eval_recs = [], []
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_eval = max(1, math.floor(len(shuffled) * eval_ratio)) if len(shuffled) > 1 else 0
        eval_recs.extend(shuffled[:n_eval])
        train_recs.extend(shuffled[n_eval:])
    rng.shuffle(train_recs)
    rng.shuffle(eval_recs)
    return train_recs, eval_recs


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT & TARGET  (identical to evaluate_paligemma2.py for consistency)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(emotion_labels: Sequence[str], schema: str) -> str:
    labels = ", ".join(emotion_labels)
    if schema == "primary":
        return (
            "<image> Analyze the visible face in the image. "
            f"Choose exactly one emotion from this list: {labels}. "
            'Return only valid JSON in this schema: {"primary_emotion":"<emotion>"}'
        )
    return (
        "<image> Analyze the image and identify each visible person separately. "
        f"For every subject, choose exactly one emotion from this list: {labels}. "
        "Set group_emotion to the most representative emotion; when several subjects "
        "are present, use the majority emotion, and if there is a tie choose the most "
        "central or salient subject. "
        'Return only valid JSON in this schema: '
        '{"subjects":[{"id":1,"emotion":"<emotion>"}],"group_emotion":"<emotion>"}'
    )


def build_target(emotion: str, schema: str) -> str:
    if schema == "primary":
        payload: Dict[str, Any] = {"primary_emotion": emotion}
    else:
        payload = {
            "subjects": [{"id": 1, "emotion": emotion}],
            "group_emotion": emotion,
        }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class FERPlusDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        prompt: str,
        target_schema: str,
    ) -> None:
        self.records = list(records)
        self.prompt = prompt
        self.target_schema = target_schema

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image = Image.open(rec["image"]).convert("RGB").resize((224, 224), Image.LANCZOS)
        answer = build_target(rec["_emotion"], self.target_schema)
        return {"image": image, "prompt": self.prompt, "answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────────────────────────────────────

class PaliGemmaCollator:
    """
    Tokenize FER+ examples into PaliGemma 2 training tensors.

    PaliGemma 2 mix uses VQA-style format:
      - Input:  <image tokens (256 for 224px)> + prompt text
      - Labels: -100 for prompt prefix, real ids for answer tokens only
    """

    def __init__(
        self,
        processor: Any,
        max_length: int,
        answer_max_length: int,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.answer_max_length = answer_max_length

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images   = [b["image"]  for b in batch]
        prompts  = [b["prompt"] for b in batch]
        answers  = [b["answer"] for b in batch]

        # Tokenize prompt + image
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        )

        # Tokenize answer tokens only (no special tokens)
        target_enc = self.processor.tokenizer(
            answers,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.answer_max_length,
            add_special_tokens=False,
        )

        input_len  = inputs["input_ids"].shape[1]
        target_ids = target_enc["input_ids"]
        batch_size = inputs["input_ids"].shape[0]

        # Mask prompt tokens from loss
        labels = torch.cat([
            torch.full((batch_size, input_len), -100, dtype=torch.long),
            target_ids,
        ], dim=1)

        full_input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)
        full_attention_mask = torch.cat([
            inputs["attention_mask"],
            torch.ones_like(target_ids),
        ], dim=1)

        return {
            "input_ids":      full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values":   inputs["pixel_values"],
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL CALLBACK  (adapted from Orazio's LossDivergenceGuardrailCallback)
# ─────────────────────────────────────────────────────────────────────────────

class GuardrailTrainingStop(RuntimeError):
    pass


class LossDivergenceGuardrailCallback(TrainerCallback):
    """Stop on NaN/Inf loss; reduce LR on loss explosion."""

    def __init__(
        self,
        stop_on_nan_loss: bool,
        loss_guard_min_logs: int,
        loss_explosion_factor: float,
        loss_explosion_abs_threshold: float,
        lr_reduction_factor: float,
        min_learning_rate: float,
        lr_reduction_cooldown_steps: int,
        max_lr_reductions: int,
    ) -> None:
        self.stop_on_nan_loss = stop_on_nan_loss
        self.loss_guard_min_logs = max(0, loss_guard_min_logs)
        self.loss_explosion_factor = loss_explosion_factor
        self.loss_explosion_abs_threshold = loss_explosion_abs_threshold
        self.lr_reduction_factor = lr_reduction_factor
        self.min_learning_rate = min_learning_rate
        self.lr_reduction_cooldown_steps = max(0, lr_reduction_cooldown_steps)
        self.max_lr_reductions = max_lr_reductions

        self.best_logged_loss: Optional[float] = None
        self.latest_logged_loss: Optional[float] = None
        self.latest_eval_loss: Optional[float] = None
        self.latest_learning_rate: Optional[float] = None
        self.logged_train_loss_count = 0
        self.last_lr_reduction_step = -(10 ** 12)
        self.lr_reduction_events: List[Dict[str, Any]] = []
        self.stop_reason: Optional[str] = None
        self.fatal_stop = False

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item()) if value.numel() == 1 else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_explosion(self, loss: float) -> bool:
        if self.best_logged_loss is None or self.logged_train_loss_count < self.loss_guard_min_logs:
            return False
        threshold = max(
            self.best_logged_loss * self.loss_explosion_factor,
            self.loss_explosion_abs_threshold,
        )
        return loss >= threshold

    def _reduce_lr(self, optimizer: Any, lr_scheduler: Any, step: int, loss: float) -> bool:
        if optimizer is None:
            return False
        old_lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
        new_lrs = [max(lr * self.lr_reduction_factor, self.min_learning_rate) for lr in old_lrs]
        if old_lrs == new_lrs:
            return False
        for group, new_lr in zip(optimizer.param_groups, new_lrs):
            group["lr"] = new_lr
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = [
                max(float(b) * self.lr_reduction_factor, self.min_learning_rate)
                for b in lr_scheduler.base_lrs
            ]
        self.last_lr_reduction_step = step
        self.latest_learning_rate = new_lrs[0] if new_lrs else None
        self.lr_reduction_events.append({
            "step": step, "loss": loss,
            "best_logged_loss": self.best_logged_loss,
            "old_lrs": old_lrs, "new_lrs": new_lrs,
        })
        return True

    def on_log(
        self, args: Any, state: Any, control: Any,
        logs: Optional[Dict[str, Any]] = None, **kwargs: Any,
    ) -> Any:
        if not logs:
            return control

        if "learning_rate" in logs:
            self.latest_learning_rate = self._to_float(logs["learning_rate"])

        # Check eval_loss NaN
        if "eval_loss" in logs:
            eval_loss = self._to_float(logs["eval_loss"])
            self.latest_eval_loss = eval_loss
            if self.stop_on_nan_loss and (eval_loss is None or not math.isfinite(eval_loss)):
                self.fatal_stop = True
                self.stop_reason = f"non_finite_eval_loss_step_{state.global_step}"
                control.should_training_stop = True
                raise GuardrailTrainingStop(
                    f"eval_loss became non-finite at step {state.global_step}: {eval_loss}"
                )

        if "loss" not in logs:
            return control

        loss = self._to_float(logs["loss"])
        self.latest_logged_loss = loss
        if loss is None:
            return control

        # Stop on NaN train loss
        if self.stop_on_nan_loss and not math.isfinite(loss):
            self.fatal_stop = True
            self.stop_reason = f"non_finite_train_loss_step_{state.global_step}"
            control.should_training_stop = True
            raise GuardrailTrainingStop(
                f"Train loss became non-finite at step {state.global_step}: {loss}"
            )

        self.logged_train_loss_count += 1
        if self.best_logged_loss is None or loss < self.best_logged_loss:
            self.best_logged_loss = loss
            return control

        if not self._is_explosion(loss):
            return control

        steps_since_last = state.global_step - self.last_lr_reduction_step
        if steps_since_last < self.lr_reduction_cooldown_steps:
            return control

        if self.max_lr_reductions > 0 and len(self.lr_reduction_events) >= self.max_lr_reductions:
            self.stop_reason = f"max_lr_reductions_reached_step_{state.global_step}"
            control.should_training_stop = True
            print("Guardrail stop: loss keeps exploding after max LR reductions.")
            return control

        reduced = self._reduce_lr(
            kwargs.get("optimizer"), kwargs.get("lr_scheduler"),
            state.global_step, loss,
        )
        if not reduced:
            self.stop_reason = f"loss_explosion_lr_not_reducible_step_{state.global_step}"
            control.should_training_stop = True
        else:
            print(
                f"Guardrail: loss spike at step {state.global_step} "
                f"(loss={loss:.4f}, best={self.best_logged_loss:.4f}) — LR reduced."
            )
        return control


class GuardedTrainer(Trainer):
    def __init__(self, *args: Any, guardrail_callback: Optional[LossDivergenceGuardrailCallback] = None, **kwargs: Any) -> None:
        self.guardrail_callback = guardrail_callback
        super().__init__(*args, **kwargs)

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        loss = super().training_step(model, inputs, *args, **kwargs)
        if self.guardrail_callback is not None:
            loss_val = self.guardrail_callback._to_float(loss)
            if (
                self.guardrail_callback.stop_on_nan_loss
                and loss_val is not None
                and not math.isfinite(loss_val)
            ):
                self.guardrail_callback.fatal_stop = True
                self.guardrail_callback.stop_reason = f"non_finite_raw_loss_step_{self.state.global_step}"
                raise GuardrailTrainingStop(
                    f"Raw training loss non-finite at step {self.state.global_step}: {loss_val}"
                )
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any]:
    print(f"\nCarico modello: {args.model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = PaliGemmaProcessor.from_pretrained(args.model_id)

    model.config.use_cache = False

    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    vram = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"Modello caricato — VRAM: {vram:.1f} GB")
    return model, processor


def setup_lora(model: Any, args: argparse.Namespace) -> Any:
    print(f"\nLoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def estimate_total_steps(args: argparse.Namespace, n_train: int) -> int:
    if args.max_steps > 0:
        return args.max_steps
    micro_per_epoch = math.ceil(n_train / max(1, args.per_device_train_batch_size))
    updates_per_epoch = math.ceil(micro_per_epoch / max(1, args.gradient_accumulation_steps))
    return max(1, math.ceil(updates_per_epoch * args.num_train_epochs))


def resolve_eval_save_steps(
    args: argparse.Namespace, n_train: int, load_best: bool
) -> Tuple[int, int, int]:
    total = estimate_total_steps(args, n_train)
    eff_eval = max(1, min(args.eval_steps, total))
    eff_save = eff_eval if load_best else args.save_steps
    return eff_eval, eff_save, total


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    emotion_labels = [normalize_emotion(e) for e in args.emotion_labels.split(",") if e.strip()]
    prompt = build_prompt(emotion_labels, args.target_schema)

    # ── Dataset ──
    print("\nCarico training set...")
    raw_train = read_json_or_jsonl(args.train_json)
    train_recs = validate_and_annotate(raw_train, emotion_labels)
    train_recs = stratified_sample(train_recs, args.sample_fraction, args.seed)
    print(f"Train: {len(train_recs)} esempi dopo stratified sampling")

    eval_recs: Optional[List] = None
    if args.eval_json:
        print("\nCarico validation set...")
        raw_eval = read_json_or_jsonl(args.eval_json)
        eval_recs = validate_and_annotate(raw_eval, emotion_labels)
        print(f"Val: {len(eval_recs)} esempi")
    elif args.eval_ratio > 0:
        train_recs, eval_recs = stratified_train_eval_split(train_recs, args.eval_ratio, args.seed)
        print(f"Auto split — train: {len(train_recs)}, val: {len(eval_recs)}")

    train_dataset = FERPlusDataset(train_recs, prompt, args.target_schema)
    eval_dataset  = FERPlusDataset(eval_recs, prompt, args.target_schema) if eval_recs else None

    # ── Model ──
    model, processor = load_model_and_processor(args)
    model = setup_lora(model, args)

    collator = PaliGemmaCollator(
        processor=processor,
        max_length=args.max_length,
        answer_max_length=args.target_answer_max_length,
    )

    # ── Training args ──
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    load_best = has_eval and args.load_best_model_at_end
    eff_eval, eff_save, total_steps = resolve_eval_save_steps(args, len(train_dataset), load_best)
    if has_eval and eff_eval != args.eval_steps:
        print(f"Adjusted eval_steps {args.eval_steps} → {eff_eval}")
    if load_best and eff_save != args.save_steps:
        print(f"Adjusted save_steps {args.save_steps} → {eff_save}")

    training_kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=eff_save,
        save_total_limit=2,
        eval_steps=eff_eval if has_eval else None,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        logging_nan_inf_filter=False,
        logging_first_step=True,
    )

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = (
        "steps" if has_eval else "no"
    )
    if "save_strategy" in ta_params:
        training_kwargs["save_strategy"] = "steps"
    if "load_best_model_at_end" in ta_params:
        training_kwargs["load_best_model_at_end"] = load_best
    if load_best:
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False

    training_args = TrainingArguments(**training_kwargs)

    # ── Guardrail callback ──
    guardrail_cb = LossDivergenceGuardrailCallback(
        stop_on_nan_loss=args.stop_on_nan_loss,
        loss_guard_min_logs=args.loss_guard_min_logs,
        loss_explosion_factor=args.loss_explosion_factor,
        loss_explosion_abs_threshold=args.loss_explosion_abs_threshold,
        lr_reduction_factor=args.lr_reduction_factor,
        min_learning_rate=args.min_learning_rate,
        lr_reduction_cooldown_steps=args.lr_reduction_cooldown_steps,
        max_lr_reductions=args.max_lr_reductions,
    )
    callbacks: List[TrainerCallback] = [guardrail_cb]
    if has_eval and args.early_stopping_patience > 0 and load_best:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    t_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in t_params:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in t_params:
        trainer_kwargs["tokenizer"] = getattr(processor, "tokenizer", None)

    trainer = GuardedTrainer(**trainer_kwargs, guardrail_callback=guardrail_cb)

    print(f"\n{'='*60}")
    print(f"Avvio training")
    print(f"  Train samples   : {len(train_dataset)}")
    print(f"  Val samples     : {len(eval_dataset) if eval_dataset else 0}")
    print(f"  Epochs          : {args.num_train_epochs}")
    print(f"  LR              : {args.learning_rate}")
    print(f"  LoRA r/alpha    : {args.lora_r}/{args.lora_alpha}")
    print(f"  max_grad_norm   : {args.max_grad_norm}")
    print(f"  Schema          : {args.target_schema}")
    print(f"  Est. steps      : {total_steps}")
    print(f"{'='*60}\n")

    # ── Train ──
    train_metrics: Dict[str, Any] = {}
    train_interrupted = False
    try:
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        train_metrics = result.metrics
    except GuardrailTrainingStop as exc:
        train_interrupted = True
        train_metrics = {"guardrail_stop": str(exc)}
        print(f"Training stopped by guardrail: {exc}")

    # ── Save ──
    should_save = not guardrail_cb.fatal_stop
    if should_save:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"\nModello salvato in: {args.output_dir}")
    else:
        print("\nSalvataggio saltato (fatal loss).")

    # ── Training summary JSON ──
    summary_path = Path(
        args.training_summary_json or
        str(Path(args.output_dir) / "training_summary.json")
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": args.output_dir,
        "training_status": "interrupted" if train_interrupted else "completed",
        "hyperparameters": {
            "model_id": args.model_id,
            "target_schema": args.target_schema,
            "emotion_labels": emotion_labels,
            "sample_fraction": args.sample_fraction,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_grad_norm": args.max_grad_norm,
            "warmup_ratio": args.warmup_ratio,
            "optim": "paged_adamw_8bit",
            "bf16": True,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset) if eval_dataset else 0,
        },
        "guardrails": {
            "stop_reason": guardrail_cb.stop_reason,
            "best_logged_loss": guardrail_cb.best_logged_loss,
            "latest_eval_loss": guardrail_cb.latest_eval_loss,
            "lr_reduction_events": guardrail_cb.lr_reduction_events,
        },
        "trainer_state": {
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        },
        "train_metrics": train_metrics,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(make_json_safe(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Training summary: {summary_path}")

    # ── S3 upload ──
    print(f"\nUpload su S3: {args.s3_bucket}")
    os.system(f"aws s3 sync {args.output_dir} {args.s3_bucket} --region eu-west-1")
    print("Upload completato.")


if __name__ == "__main__":
    main()
