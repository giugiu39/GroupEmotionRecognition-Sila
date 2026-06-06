# PaliGemma 2 — Facial Emotion Recognition on FER+

**Author:** Gianluca Perrotta (ID: 277091)  
**Model:** `google/paligemma2-3b-mix-224`  
**Dataset:** FER+ (8-class and 4-class distress variants)

---

## Approach

PaliGemma 2 is a 3B-parameter Vision-Language Model (VLM) developed by Google, coupling a **SigLIP-So400M vision encoder** with a **Gemma 2 language decoder**. The SigLIP encoder was pre-trained contrastively on billions of image-text pairs, including a large fraction of facial images, making it inherently strong on facial expression tasks before any task-specific supervision.

### Why Fine-Tuning Was Not Used

Fine-tuning with QLoRA was attempted, but all runs produced **rapid and irrecoverable overfitting** across every tested learning rate. The underlying cause is that PaliGemma 2's vision encoder already places facial emotion features in a near-optimal position in representation space relative to the FER+ label set. Any gradient update from FER+ supervision drives the LoRA weights toward memorizing training samples rather than generalizing, because there is no meaningful "gap" left between the pre-trained representation and the target task.

**Fine-tuning attempts:**

| Learning Rate | LoRA r / α | Epochs | Outcome |
|--------------|------------|--------|---------|
| 1e-5 | 4 / 8 | 1 | Validation loss diverges from step ~50; training loss drops while val loss rises |
| 2e-6 | 4 / 8 | 1 | Same pattern; slower onset but still diverges before epoch end |
| 1e-6 | 4 / 8 | 1 | Marginal improvement; validation loss still increases throughout |

All runs used: 4-bit NF4 quantization (QLoRA), `paged_adamw_8bit`, cosine LR schedule, warmup_ratio=0.05, max_grad_norm=0.3, sample_fraction=1/3 (stratified per class), gradient_accumulation_steps=16.

**Conclusion:** The base model, prompted with a simple one-word instruction, consistently outperforms all fine-tuned checkpoints. Fine-tuning is counterproductive here.

---

## Final Approach: Base Model + Simple Prompt + Post-Processing

The base model is queried with a minimal direct prompt:

```
<image> What is the emotion of the person in this image?
Answer with exactly one word from this list: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
```

PaliGemma 2 base responds reliably with a **single emotion word** (e.g., `happiness`). A post-processing step converts this plain-word response into the group schema expected by the pipeline:

```json
{"subjects": [{"id": 1, "emotion": "happiness"}], "group_emotion": "happiness"}
```

The output parser resolves responses in three stages:
1. Attempt full JSON parse → extract `group_emotion` or `primary_emotion`
2. Check if the entire stripped response is a known emotion word (plain-word path)
3. Scan raw text for exactly one matching emotion word (fallback)

This design achieves **0.00% invalid predictions** — the simplicity of the prompt eliminates all parsing failures that occurred with complex JSON schema prompts.

**VRAM usage:** 2.3 GB (4-bit NF4 quantization, Tesla T4)

---

## Results

### 8-Class Evaluation (FER+ standard labels)

| Metric | Value |
|--------|-------|
| Accuracy | 0.691 |
| Macro F1 | 0.436 |
| Invalid predictions | 0.00% |

The low macro F1 (0.436) reflects strong performance on high-support classes (happiness, neutral) and near-zero F1 on severely imbalanced minority classes (contempt, disgust, fear), which have very few test samples and overlap in facial muscle activation patterns.

### 4-Class Distress Evaluation

Negative emotions (sadness, anger, disgust, fear, contempt) are merged into a single `distress` class before evaluation. This reduces the label space to {happiness, neutral, distress, surprise} and eliminates the minority-class imbalance penalty.

| Metric | Value |
|--------|-------|
| Accuracy | 0.695 |
| Macro F1 | 0.691 |
| Invalid predictions | 0.00% |

**Per-class F1 (4 classes):**

| Emotion | F1 | Comment |
|---------|----|---------|
| happiness | 0.880 | Excellent |
| neutral | 0.665 | Good |
| distress | 0.621 | Decent |
| surprise | 0.598 | Decent |

The distress grouping improves macro F1 from 0.436 to 0.691 — a +0.255 gain — without changing the model or prompting strategy.

---

## Confusion Matrix Analysis

Key observations from the 8-class confusion matrix:

- **Happiness** is the most reliably classified emotion — large class, visually distinctive (open smile, raised cheeks).
- **Neutral** prediction is strong; the model rarely hallucinates an expression on a neutral face.
- **Contempt, disgust, and fear** are systematically confused with each other and with sadness/anger. These emotions share subtle, overlapping facial action unit patterns (lowered brows, narrowed eyes) that are difficult to disambiguate from a single 224×224 frame without fine-grained feature analysis.
- **Surprise vs. neutral** is a secondary confusion point; mild surprise expressions can appear neutral at low resolution.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_paligemma2_base.py` | Evaluate base model (no adapter) on 8 standard FER+ classes |
| `evaluate_paligemma2_distress.py` | Evaluate base model with 4-class distress label mapping |
| `evaluate_paligemma2_FineTuning.py` | Evaluate a QLoRA adapter (kept for reference; not used in final results) |
| `trainGianlucaFineTuning.py` | QLoRA fine-tuning script (overfitting observed; not used in final results) |

All evaluation scripts support checkpoint/resume: if interrupted, re-running with the same arguments picks up from the last saved sample.

---

## How to Run

### Prerequisites

```bash
pip install torch transformers peft bitsandbytes pillow tqdm
```

Requires a CUDA GPU with at least 4 GB VRAM (2.3 GB used in practice at 4-bit).

---

### Evaluate base model — 8 classes

```bash
python evaluate_paligemma2_base.py \
  --test_json            /workspace/datasets/jsonl/test.jsonl \
  --metrics_json         /workspace/gianluca/paligemma2_base/test_metrics.json \
  --predictions_jsonl    /workspace/gianluca/paligemma2_base/test_predictions.jsonl \
  --checkpoint_jsonl     /workspace/gianluca/paligemma2_base/test_predictions.checkpoint.jsonl \
  --partial_metrics_json /workspace/gianluca/paligemma2_base/test_metrics.partial.json \
  --checkpoint_every 25 \
  --target_schema group
```

---

### Evaluate base model — 4-class distress mapping

**Step 1 — build the distress JSONL** (run from project root or `dataset/`):

```bash
python ../../dataset/build_distress_jsonl.py \
  --input  /workspace/datasets/jsonl/test.jsonl \
  --output /workspace/datasets/jsonl/test_distress.jsonl
```

**Step 2 — run evaluation:**

```bash
python evaluate_paligemma2_distress.py \
  --test_json            /workspace/datasets/jsonl/test_distress.jsonl \
  --metrics_json         /workspace/gianluca/paligemma2_distress/test_metrics.json \
  --predictions_jsonl    /workspace/gianluca/paligemma2_distress/test_predictions.jsonl \
  --checkpoint_jsonl     /workspace/gianluca/paligemma2_distress/test_predictions.checkpoint.jsonl \
  --partial_metrics_json /workspace/gianluca/paligemma2_distress/test_metrics.partial.json \
  --checkpoint_every 25 \
  --target_schema group
```

---

### Evaluate a QLoRA adapter (reference only)

```bash
python evaluate_paligemma2_FineTuning.py \
  --test_json    /workspace/datasets/jsonl/test.jsonl \
  --adapter_dir  /workspace/gianluca/paligemma2_fer_lora \
  --metrics_json /workspace/gianluca/paligemma2_fer_lora/test_metrics.json \
  --target_schema group
```

---

### Fine-tuning (reference; not recommended — see overfitting section above)

```bash
python trainGianlucaFineTuning.py \
  --train_json       /workspace/datasets/jsonl/train.jsonl \
  --eval_json        /workspace/datasets/jsonl/val.jsonl \
  --output_dir       /workspace/gianluca/paligemma2_fer_lora \
  --num_train_epochs 1 \
  --sample_fraction  0.3333333333 \
  --learning_rate    1e-5 \
  --lora_r 4 --lora_alpha 8
```

After training, the script automatically uploads the adapter to S3:

```bash
aws s3 sync /workspace/gianluca/paligemma2_fer_lora \
  s3://dimesvlm-data/models/paligemma2_fer_lora --region eu-west-1
```
