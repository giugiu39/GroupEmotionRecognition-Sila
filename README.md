# Smart Emotion Recognition System — Parco Nazionale della Sila

Academic project for the joint course **IoT Device Programming** + **Distributed Systems, Cloud and Edge Computing**  
University of Calabria (UNICAL) — A.Y. 2025/2026

---

## Project Overview

The system captures facial images from visitor groups at Parco Nazionale della Sila using a Raspberry Pi 4 edge device and classifies the group emotion in real-time through a cloud inference pipeline powered by Vision-Language Models (VLMs). Classified emotions are persisted to AWS DynamoDB and S3 and surfaced through a Flutter mobile application used by park staff.

---

## System Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  EDGE                                                                       │
 │                                                                             │
 │  ┌──────────────────────┐                                                   │
 │  │  Raspberry Pi 4       │  JPEG frame (REST/HTTPS)                        │
 │  │  Camera Module v2     │ ──────────────────────────────────────────────► │
 │  │  (capture + compress) │                                                  │
 │  └──────────────────────┘                                                   │
 └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  CLOUD (AWS eu-west-1)                                                      │
 │                                                                             │
 │  ┌──────────────────┐     ┌───────────────────────────────────────────────┐ │
 │  │  API Gateway     │────►│  Lambda                                       │ │
 │  │  (HTTPS endpoint)│     │  (routing, auth, S3 upload, job dispatch)     │ │
 │  └──────────────────┘     └───────────────────┬───────────────────────────┘ │
 │                                               │                             │
 │                                               ▼                             │
 │                           ┌───────────────────────────────────────────────┐ │
 │                           │  EC2 g4dn.xlarge                              │ │
 │                           │  NVIDIA Tesla T4 — 16 GB VRAM                 │ │
 │                           │  VLM inference:                               │ │
 │                           │    • PaliGemma 2 3B  (Gianluca)               │ │
 │                           │    • MiniCPM-V 2.6   (Orazio)                 │ │
 │                           │    • Moondream2       (Asrar)                 │ │
 │                           └───────────────────┬───────────────────────────┘ │
 │                                               │                             │
 │                    ┌──────────────────────────┴──────────────────────────┐  │
 │                    │                                                     │  │
 │                    ▼                                                     ▼  │
 │        ┌────────────────────┐                              ┌───────────────┐ │
 │        │  DynamoDB          │                              │  S3           │ │
 │        │  (emotion results) │                              │  (raw images) │ │
 │        └────────────────────┘                              └───────────────┘ │
 │                    │                                                        │
 │                    ▼                                                        │
 │        ┌────────────────────────────────────────┐                          │
 │        │  Flutter Mobile App                    │                          │
 │        │  (real-time group emotion dashboard)   │                          │
 │        └────────────────────────────────────────┘                          │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Team

| Name | Student ID | Contribution |
|------|------------|--------------|
| Gianluca Perrotta | 277091 | PaliGemma 2 — evaluation, fine-tuning attempts, distress grouping |
| Marco Macrì | 276608 | System architecture, AWS infrastructure, Flutter app |
| Orazio Ruberto | 276576 | MiniCPM-V — QLoRA fine-tuning & evaluation |
| Asrar Jemal Mohammed | 284598 | Moondream2 — evaluation |

---

## Repository Structure

```
.
├── README.md
├── .gitignore
├── vlm/
│   ├── paligemma2/     # PaliGemma 2 3B — eval & fine-tuning scripts (Gianluca)
│   ├── minicpmv/       # MiniCPM-V 2.6 — fine-tuning & eval (Orazio)
│   └── moondream2/     # Moondream2 — evaluation (Asrar)
├── dataset/            # FER+ JSONL builders and dataset split files
├── edge/               # Raspberry Pi 4 capture script
├── cloud/              # AWS Lambda function
└── docs/               # Project report and documentation
```

---

## Dataset — FER+

| Property | Value |
|----------|-------|
| Source | FER+ (Barsoum et al., 2016) |
| Classes | 8: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt |
| Total images | ~78,000 facial images |
| Original resolution | 48×48 px grayscale (resized to 224×224 RGB for VLM input) |
| Splits | train / validation / test |
| Distress variant | 4 classes — sadness + anger + disgust + fear + contempt → `distress` |

JSONL files in `dataset/` map each image path (as it appears on the EC2 instance at `/workspace/datasets/images/`) to its ground-truth emotion label.

---

## Results Summary

### 8-Class Evaluation (FER+ standard)

| Model | Strategy | Accuracy | Macro F1 | Invalid % |
|-------|----------|----------|----------|-----------|
| **PaliGemma 2 3B** | Base (no fine-tuning) + simple prompt | **0.691** | **0.436** | **0.00%** |
| MiniCPM-V 2.6 | QLoRA fine-tuning (1 epoch, 1/3 data) | — | — | — |
| Moondream2 | — | — | — | — |

### PaliGemma 2 — 4-Class Distress Evaluation

Grouping sadness / anger / disgust / fear / contempt into a single `distress` class eliminates the severe class imbalance caused by low-support minority emotions in FER+, substantially improving macro F1.

| Accuracy | Macro F1 |
|----------|----------|
| 0.695 | 0.691 |

**Per-class F1:**

| Emotion | F1 |
|---------|----|
| happiness | 0.880 |
| neutral | 0.665 |
| distress | 0.621 |
| surprise | 0.598 |

### MiniCPM-V 2.6 — Per-Class F1 (8 classes, after QLoRA fine-tuning)

| Emotion | F1 | Comment |
|---------|----|---------|
| happiness | 0.920 | Excellent |
| neutral | 0.788 | Good |
| surprise | 0.779 | Good |
| anger | 0.702 | Good |
| sadness | 0.571 | Moderate |
| fear | 0.396 | Weak |
| disgust | 0.253 | Very weak |
| contempt | 0.125 | Very weak |

---

## AWS Infrastructure

| Component | Specification |
|-----------|---------------|
| Instance type | `g4dn.xlarge` |
| GPU | NVIDIA Tesla T4 — 16 GB VRAM |
| Storage | EBS gp3 + S3 (`s3://dimesvlm-data/`) |
| Database | AWS DynamoDB |
| Routing & auth | API Gateway + Lambda |
| Region | `eu-west-1` |

---

## How to Run

See the model-specific READMEs for full instructions:

- **PaliGemma 2:** [`vlm/paligemma2/README.md`](vlm/paligemma2/README.md)
- **MiniCPM-V:** `vlm/minicpmv/`
- **Moondream2:** `vlm/moondream2/`

For dataset preparation: [`dataset/`](dataset/)

---

## Academic Context

- **University:** University of Calabria (UNICAL)
- **Courses:** IoT Device Programming + Distributed Systems, Cloud and Edge Computing
- **Academic Year:** 2025/2026
- **Report:** [`docs/GER_Report_final.docx`](docs/GER_Report_final.docx)
