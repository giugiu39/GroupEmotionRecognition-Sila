#!/usr/bin/env python3
"""
Evaluate PaliGemma 2 BASE model (no LoRA adapter) on the FERPlus test set.
Same structure as evaluate_paligemma2_FineTuning.py but loads the base model
directly without any adapter — useful when the base model is already strong
enough on the task (as observed with PaliGemma 2 on FER+).

Usage:
  python3 evaluate_paligemma2_base.py \
    --test_json  /workspace/datasets/jsonl/test.jsonl \
    --metrics_json /workspace/gianluca/paligemma2_base/test_metrics.json \
    --predictions_jsonl /workspace/gianluca/paligemma2_base/test_predictions.jsonl \
    --checkpoint_jsonl /workspace/gianluca/paligemma2_base/test_predictions.checkpoint.jsonl \
    --partial_metrics_json /workspace/gianluca/paligemma2_base/test_metrics.partial.json \
    --checkpoint_every 25 \
    --target_schema group
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "google/paligemma2-3b-mix-224"
DEFAULT_EMOTIONS = (
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PaliGemma 2 BASE model (no adapter) on FERPlus test data."
    )
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="Base PaliGemma 2 model id.")
    parser.add_argument("--metrics_json", default="./paligemma2_base_test_metrics.json")
    parser.add_argument("--predictions_jsonl", default=None)
    parser.add_argument("--checkpoint_jsonl", default=None)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--resume_from_checkpoint",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partial_metrics_json", default=None)
    parser.add_argument("--target_schema", choices=("group", "primary"), default="group")
    parser.add_argument("--emotion_labels", default=",".join(DEFAULT_EMOTIONS))
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
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


def extract_true_emotion(record: Dict[str, Any]) -> str:
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


def validate_test_records(
    records: Sequence[Dict[str, Any]],
    allowed: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_set = {normalize_emotion(e) for e in allowed}
    out, skipped_img, skipped_label = [], 0, Counter()
    for rec in records:
        if not Path(str(rec.get("image", ""))).exists():
            skipped_img += 1
            continue
        emo = extract_true_emotion(rec)
        if emo not in allowed_set:
            skipped_label[emo] += 1
            continue
        item = dict(rec)
        item["_true_emotion"] = emo
        out.append(item)
    if skipped_img:
        print(f"Skipped {skipped_img} records with missing images.")
    if skipped_label:
        print(f"Skipped labels outside --emotion_labels: {dict(skipped_label)}")
    if not out:
        raise ValueError("No valid test records after filtering.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT  (identical to training scripts for consistency)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(emotion_labels: Sequence[str], schema: str) -> str:
    """
    Simple single-word prompt for PaliGemma 2 base.
    The base model responds much more reliably to simple one-word instructions
    than to complex JSON schema prompts. The script converts the single-word
    response to the structured group schema in post-processing.
    """
    labels = ", ".join(emotion_labels)
    return (
        f"<image> What is the emotion of the person in this image? "
        f"Answer with exactly one word from this list: {labels}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (base model, no adapter)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any]:
    """Load PaliGemma 2 base in 4-bit — no LoRA adapter."""
    print(f"\nCarico modello base: {args.model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processor = PaliGemmaProcessor.from_pretrained(args.model_id)

    vram = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"Modello caricato — VRAM: {vram:.1f} GB")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict_one(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    image = Image.open(image_path).convert("RGB").resize((224, 224), Image.LANCZOS)

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
    else:
        gen_kwargs["do_sample"] = False

    pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    generated_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    answer_ids = generated_ids[:, prompt_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    decoded = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


# ─────────────────────────────────────────────────────────────────────────────
# PARSING MODEL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def majority_vote(labels: Sequence[str]) -> Optional[str]:
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def parse_predicted_emotion(
    raw_text: str,
    allowed: Sequence[str],
    schema: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Parse model output into a single emotion label.

    PaliGemma 2 base often responds with a plain word (e.g. 'happiness')
    instead of structured JSON. This function handles both cases:
    1. Proper JSON response -> extract group_emotion or primary_emotion
    2. Plain word response -> check if it matches an allowed emotion directly
    3. Fallback -> scan raw text for known emotion words
    """
    allowed_set = {normalize_emotion(e) for e in allowed}

    # ── Step 1: try JSON parsing ──
    parsed = extract_json_object(raw_text)
    info: Dict[str, Any] = {
        "parsed_json": parsed,
        "parse_status": "json_ok" if parsed is not None else "json_failed",
    }

    prediction: Optional[str] = None
    if parsed is not None:
        if schema == "primary":
            prediction = (
                parsed.get("primary_emotion")
                or parsed.get("emotion")
                or parsed.get("label")
            )
        else:
            prediction = parsed.get("group_emotion") or parsed.get("primary_emotion")
            if prediction is None and isinstance(parsed.get("subjects"), list):
                emos = [
                    normalize_emotion(str(s["emotion"]))
                    for s in parsed["subjects"]
                    if isinstance(s, dict) and s.get("emotion") is not None
                ]
                prediction = majority_vote(emos)

        if prediction is not None:
            norm = normalize_emotion(str(prediction))
            if norm in allowed_set:
                info["parse_status"] = "valid_label"
                return norm, info
            info["parse_status"] = "label_not_allowed"
            info["raw_label"] = norm

    # ── Step 2: plain word response (e.g. PaliGemma 2 base returns "happiness") ──
    # Check if the entire stripped response is a known emotion word
    stripped = normalize_emotion(raw_text.strip())
    if stripped in allowed_set:
        info["parse_status"] = "plain_word_label"
        info["parsed_json"] = {
            "subjects": [{"id": 1, "emotion": stripped}],
            "group_emotion": stripped,
        }
        return stripped, info

    # ── Step 3: scan raw text for exactly one known emotion word ──
    raw_lower = normalize_emotion(raw_text)
    found = [e for e in allowed_set if re.search(rf"\b{re.escape(e)}\b", raw_lower)]
    if len(found) == 1:
        info["parse_status"] = "text_label_fallback"
        info["parsed_json"] = {
            "subjects": [{"id": 1, "emotion": found[0]}],
            "group_emotion": found[0],
        }
        return found[0], info

    return None, info


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(
    true_labels: Sequence[str],
    pred_labels: Sequence[Optional[str]],
    emotion_labels: Sequence[str],
) -> Dict[str, Any]:
    labels = [normalize_emotion(e) for e in emotion_labels]
    idx = {e: i for i, e in enumerate(labels)}
    n = len(labels)
    confusion = [[0] * n for _ in range(n)]

    total, correct, invalid = len(true_labels), 0, 0
    for true, pred in zip(true_labels, pred_labels):
        true = normalize_emotion(true)
        pred = normalize_emotion(pred) if pred else None
        if pred is None or pred not in idx:
            invalid += 1
            continue
        if true == pred:
            correct += 1
        if true in idx:
            confusion[idx[true]][idx[pred]] += 1

    per_class: Dict[str, Any] = {}
    macro_p, macro_r, macro_f1 = [], [], []
    wp_sum, wr_sum, wf1_sum, w_total = 0.0, 0.0, 0.0, 0

    for i, label in enumerate(labels):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(n) if r != i)
        fn = sum(confusion[i][c] for c in range(n) if c != i)
        support = sum(confusion[i])

        inv_for_class = sum(
            1 for t, p in zip(true_labels, pred_labels)
            if normalize_emotion(t) == label and (p is None or normalize_emotion(p) not in idx)
        )
        fn += inv_for_class
        support += inv_for_class

        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f = safe_div(2 * p * r, p + r)

        per_class[label] = {"precision": round(p, 4), "recall": round(r, 4),
                            "f1": round(f, 4), "support": support}
        macro_p.append(p); macro_r.append(r); macro_f1.append(f)
        wp_sum += p * support; wr_sum += r * support
        wf1_sum += f * support; w_total += support

    accuracy = safe_div(correct, total)
    m_p = safe_div(sum(macro_p), n)
    m_r = safe_div(sum(macro_r), n)
    m_f = safe_div(sum(macro_f1), n)
    w_p = safe_div(wp_sum, w_total)
    w_r = safe_div(wr_sum, w_total)
    w_f = safe_div(wf1_sum, w_total)

    tp_tot = correct
    fp_tot = sum(confusion[r][c] for r in range(n) for c in range(n) if r != c)
    fn_tot = fp_tot + invalid
    mi_p = safe_div(tp_tot, tp_tot + fp_tot)
    mi_r = safe_div(tp_tot, tp_tot + fn_tot)
    mi_f = safe_div(2 * mi_p * mi_r, mi_p + mi_r)

    return {
        "overall": {
            "num_samples": total,
            "num_correct": correct,
            "num_incorrect": total - correct,
            "num_invalid_predictions": invalid,
            "invalid_prediction_rate": round(safe_div(invalid, total), 4),
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(m_r, 4),
            "macro_precision": round(m_p, 4),
            "macro_recall": round(m_r, 4),
            "macro_f1": round(m_f, 4),
            "weighted_precision": round(w_p, 4),
            "weighted_recall": round(w_r, 4),
            "weighted_f1": round(w_f, 4),
            "micro_precision": round(mi_p, 4),
            "micro_recall": round(mi_r, 4),
            "micro_f1": round(mi_f, 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion,
            "rows": "true_labels",
            "columns": "predicted_labels",
            "note": "Invalid/unparsed predictions counted as errors, not shown as a column.",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def add_suffix(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}")
    return path.with_name(f"{path.name}{suffix}")


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_jsonl:
        return Path(args.checkpoint_jsonl)
    if args.predictions_jsonl:
        return add_suffix(args.predictions_jsonl, ".checkpoint.jsonl")
    return add_suffix(args.metrics_json, ".checkpoint.jsonl")


def resolve_partial_metrics_path(args: argparse.Namespace) -> Path:
    if args.partial_metrics_json:
        return Path(args.partial_metrics_json)
    return add_suffix(args.metrics_json, ".partial.json")


def load_prediction_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    loaded: Dict[int, Dict[str, Any]] = {}
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(item, dict) or "index" not in item:
                skipped += 1
                continue
            loaded[int(item["index"])] = item
    print(f"Loaded {len(loaded)} predictions from checkpoint: {path}")
    if skipped:
        print(f"Skipped {skipped} malformed checkpoint lines.")
    return loaded


def align_checkpoint(
    checkpoint: Dict[int, Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    aligned: Dict[int, Dict[str, Any]] = {}
    skipped = 0
    for idx, item in checkpoint.items():
        if idx < 0 or idx >= len(test_records):
            skipped += 1
            continue
        if str(item.get("image", "")) != str(test_records[idx].get("image", "")):
            skipped += 1
            continue
        norm = dict(item)
        norm["true_emotion"] = test_records[idx]["_true_emotion"]
        norm["is_correct"] = norm.get("predicted_emotion") == norm["true_emotion"]
        aligned[idx] = norm
    if skipped:
        print(f"Skipped {skipped} checkpoint entries that don't match current test set.")
    return aligned


def prepare_checkpoint_file(path: Path, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        path.touch(exist_ok=True)
    else:
        with path.open("w", encoding="utf-8") as f:
            f.flush()
            os.fsync(f.fileno())


def append_prediction_checkpoint(path: Path, item: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    predictions: Sequence[Dict[str, Any]],
    emotion_labels: Sequence[str],
    args: argparse.Namespace,
    status: str,
    total_samples: int,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    true_labels = [p["true_emotion"] for p in predictions]
    pred_labels = [p.get("predicted_emotion") for p in predictions]
    parse_counts = Counter(p.get("parse_status", "unknown") for p in predictions)

    metrics = compute_metrics(true_labels, pred_labels, emotion_labels)
    metrics["metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model": args.model_id,
        "mode": "base_no_adapter",
        "test_json": str(args.test_json),
        "target_schema": args.target_schema,
        "emotion_labels": list(emotion_labels),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "limit": args.limit,
        "total_samples": total_samples,
        "completed_samples": len(predictions),
        "remaining_samples": max(0, total_samples - len(predictions)),
        "checkpoint_jsonl": str(checkpoint_path),
    }
    metrics["prediction_parsing"] = dict(parse_counts)
    return metrics


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def save_predictions_jsonl(path: Path, predictions: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint_every must be positive.")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    emotion_labels = [normalize_emotion(e) for e in args.emotion_labels.split(",") if e.strip()]
    if not emotion_labels:
        raise ValueError("--emotion_labels cannot be empty.")

    raw = read_json_or_jsonl(args.test_json)
    test_records = validate_test_records(raw, emotion_labels)
    if args.limit is not None:
        test_records = test_records[:args.limit]
    print(f"Test set: {len(test_records)} valid samples.")

    model, processor = load_model_and_processor(args)
    prompt = build_prompt(emotion_labels, args.target_schema)

    ckpt_path = resolve_checkpoint_path(args)
    partial_path = resolve_partial_metrics_path(args)

    loaded_ckpt = load_prediction_checkpoint(ckpt_path) if args.resume_from_checkpoint else {}
    completed_by_idx = align_checkpoint(loaded_ckpt, test_records)
    predictions_list: List[Dict[str, Any]] = [
        completed_by_idx[i]
        for i in sorted(completed_by_idx)
        if 0 <= i < len(test_records)
    ]

    if predictions_list:
        print(f"Resuming: {len(predictions_list)}/{len(test_records)} already done.")
    prepare_checkpoint_file(ckpt_path, resume=args.resume_from_checkpoint)

    newly_done = 0
    interrupted = False

    try:
        for idx, record in enumerate(test_records):
            if idx in completed_by_idx:
                continue

            true_label = record["_true_emotion"]
            raw_output = predict_one(model, processor, record["image"], prompt, args)
            pred_label, parse_info = parse_predicted_emotion(raw_output, emotion_labels, args.target_schema)

            item: Dict[str, Any] = {
                "index": idx,
                "image": record["image"],
                "true_emotion": true_label,
                "predicted_emotion": pred_label,
                "is_correct": pred_label == true_label,
                "parse_status": parse_info["parse_status"],
                "raw_model_output": raw_output,
                "parsed_model_output": parse_info.get("parsed_json"),
            }

            append_prediction_checkpoint(ckpt_path, item)
            completed_by_idx[idx] = item
            predictions_list.append(item)
            newly_done += 1

            n_done = len(completed_by_idx)
            if newly_done % args.checkpoint_every == 0 or n_done == len(test_records):
                status = "completed" if n_done == len(test_records) else "partial"
                partial_report = build_report(
                    predictions=sorted(predictions_list, key=lambda x: x["index"]),
                    emotion_labels=emotion_labels,
                    args=args,
                    status=status,
                    total_samples=len(test_records),
                    checkpoint_path=ckpt_path,
                )
                save_json(partial_path, partial_report)
                print(
                    f"Progress: {n_done}/{len(test_records)} — "
                    f"acc={partial_report['overall']['accuracy']:.4f} "
                    f"macro_f1={partial_report['overall']['macro_f1']:.4f}"
                )

    except KeyboardInterrupt:
        interrupted = True
        print("Evaluation interrupted. Saving partial results...")

    predictions_list = sorted(predictions_list, key=lambda x: x["index"])
    n_done = len(predictions_list)
    status = (
        "interrupted" if interrupted
        else "completed" if n_done == len(test_records)
        else "partial"
    )

    final_report = build_report(
        predictions=predictions_list,
        emotion_labels=emotion_labels,
        args=args,
        status=status,
        total_samples=len(test_records),
        checkpoint_path=ckpt_path,
    )
    save_json(Path(args.metrics_json), final_report)
    print(f"\nMetrics saved to: {args.metrics_json}")

    if args.predictions_jsonl:
        save_predictions_jsonl(Path(args.predictions_jsonl), predictions_list)
        print(f"Predictions saved to: {args.predictions_jsonl}")

    # Print per-class summary
    print("\n── Per-class performance ──")
    print(f"{'Emotion':12s}  {'F1':>6}  {'Recall':>7}  {'Support':>8}  Comment")
    per = final_report["per_class"]
    for emo in emotion_labels:
        if emo not in per:
            continue
        f1  = per[emo]["f1"]
        rec = per[emo]["recall"]
        sup = per[emo]["support"]
        comment = (
            "Excellent" if f1 >= 0.85 else
            "Good"      if f1 >= 0.70 else
            "Decent"    if f1 >= 0.60 else
            "Moderate"  if f1 >= 0.50 else
            "Weak"      if f1 >= 0.30 else
            "Very weak"
        )
        print(f"{emo:12s}  {f1:6.3f}  {rec:7.3f}  {sup:8d}  {comment}")

    ov = final_report["overall"]
    print(f"\nAccuracy     : {ov['accuracy']:.4f}")
    print(f"Macro F1     : {ov['macro_f1']:.4f}")
    print(f"Weighted F1  : {ov['weighted_f1']:.4f}")
    print(f"Invalid preds: {ov['num_invalid_predictions']} ({ov['invalid_prediction_rate']:.2%})")

    if interrupted:
        print(f"\nResume with the same command. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()