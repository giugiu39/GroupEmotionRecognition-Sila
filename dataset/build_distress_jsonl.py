#!/usr/bin/env python3
"""
build_distress_jsonl.py
=======================
Creates a new JSONL test file where negative emotions are merged into
a single 'distress' class.

Mapping:
  sadness   -> distress
  anger     -> distress
  disgust   -> distress
  fear      -> distress
  contempt  -> distress
  neutral   -> neutral   (unchanged)
  happiness -> happiness (unchanged)
  surprise  -> surprise  (unchanged)

Usage (run on local PC):
  python build_distress_jsonl.py \
    --input  C:\\Users\\gianl\\OneDrive\\Desktop\\FER_plus_real\\jsonl\\test.jsonl \
    --output C:\\Users\\gianl\\OneDrive\\Desktop\\FER_plus_real\\jsonl\\test_distress.jsonl

Then upload to S3:
  aws s3 cp test_distress.jsonl s3://dimesvlm-data/dataset/test_distress.jsonl \
    --profile dimes-vlm --region eu-west-1
"""

import argparse
import json
from pathlib import Path

NEGATIVE_TO_DISTRESS = {"sadness", "anger", "disgust", "fear", "contempt"}


def remap_emotion(emotion: str) -> str:
    e = emotion.strip().lower().replace(" ", "_")
    return "distress" if e in NEGATIVE_TO_DISTRESS else e


def parse_output(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise ValueError(f"Unsupported output type: {type(raw)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge negative emotions into 'distress' class in a FER+ JSONL file."
    )
    parser.add_argument("--input",  required=True, help="Path to original test.jsonl")
    parser.add_argument("--output", required=True, help="Path to output test_distress.jsonl")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count stats
    stats = {"total": 0, "remapped": 0, "kept": 0}
    class_counts: dict = {}

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            output_payload = parse_output(record.get("output", {}))

            original_emotion = output_payload.get("primary_emotion", "")
            new_emotion = remap_emotion(original_emotion)

            # Build new output payload
            new_payload = {"primary_emotion": new_emotion}

            # Also handle group schema if present
            if "subjects" in output_payload:
                new_subjects = []
                for subj in output_payload["subjects"]:
                    new_subj = dict(subj)
                    new_subj["emotion"] = remap_emotion(subj.get("emotion", ""))
                    new_subjects.append(new_subj)
                new_payload["subjects"] = new_subjects
                new_payload["group_emotion"] = new_emotion

            new_record = dict(record)
            new_record["output"] = json.dumps(new_payload, ensure_ascii=True, separators=(",", ":"))

            fout.write(json.dumps(new_record, ensure_ascii=True) + "\n")

            stats["total"] += 1
            if new_emotion != original_emotion:
                stats["remapped"] += 1
            else:
                stats["kept"] += 1

            class_counts[new_emotion] = class_counts.get(new_emotion, 0) + 1

    print(f"\nDone!")
    print(f"  Total records : {stats['total']}")
    print(f"  Remapped      : {stats['remapped']} (negative -> distress)")
    print(f"  Kept          : {stats['kept']} (unchanged)")
    print(f"\nClass distribution in output:")
    for emotion, count in sorted(class_counts.items()):
        print(f"  {emotion:12s}: {count}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
