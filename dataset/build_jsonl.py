import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# ─────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────
PROJECT_DIR     = Path("C:\\Users\\gianl\\OneDrive\\Desktop\\FER_plus_real")
IMAGES_DIR      = PROJECT_DIR / "images"
OUTPUT_JSONL    = PROJECT_DIR / "jsonl"

# Path che useremo sull'istanza EC2 (per riscrivere nei JSONL)
EC2_BASE_PATH = "/workspace/datasets/images"

# Mapping nomi cartelle → file output
SPLIT_MAP = {
    "train":      "train.jsonl",
    "validation": "val.jsonl",
    "test":       "test.jsonl",
}

# Mapping classi cartella → label JSON (nomi ufficiali FER+)
CLASS_MAP = {
    "angry":    "anger",
    "contempt": "contempt",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happiness",
    "neutral":  "neutral",
    "sad":      "sadness",
    "surprise": "surprise",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def build_jsonl():
    """Genera i 3 JSONL leggendo dalla struttura images/ già pronta."""

    # Verifiche preliminari
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"❌ {IMAGES_DIR} non trovata.")

    for split in SPLIT_MAP.keys():
        if not (IMAGES_DIR / split).exists():
            raise FileNotFoundError(
                f"❌ {IMAGES_DIR / split} non trovata."
            )

    OUTPUT_JSONL.mkdir(exist_ok=True)

    # Statistiche
    stats = {
        split: {"total": 0, "by_class": Counter()}
        for split in SPLIT_MAP.keys()
    }

    # Output file handles
    output_files = {
        split: open(OUTPUT_JSONL / fname, 'w', encoding='utf-8')
        for split, fname in SPLIT_MAP.items()
    }

    print(f"📂 Lettura da: {IMAGES_DIR}\n")

    # ──────────────────────────────────────────────
    # PROCESS EACH SPLIT
    # ──────────────────────────────────────────────
    for split in ["train", "validation", "test"]:
        print(f"\n🔄 Processing split: {split}")

        split_dir = IMAGES_DIR / split
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

        for class_dir in class_dirs:
            class_folder_name = class_dir.name

            if class_folder_name not in CLASS_MAP:
                print(f"   ⚠️  Cartella sconosciuta ignorata: {class_folder_name}")
                continue

            emotion_label = CLASS_MAP[class_folder_name]

            # Lista immagini in questa classe
            images = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]

            for img_path in tqdm(images,
                                 desc=f"  {class_folder_name:10s} → {emotion_label}",
                                 leave=False):
                # Path che apparirà nel JSONL (path EC2)
                ec2_img_path = f"{EC2_BASE_PATH}/{split}/{class_folder_name}/{img_path.name}"

                # Costruisci output JSON
                output_json = {"primary_emotion": emotion_label}

                # Costruisci record JSONL — solo image + output
                record = {
                    "image":  ec2_img_path,
                    "output": json.dumps(output_json, ensure_ascii=False),
                }

                output_files[split].write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                stats[split]["total"] += 1
                stats[split]["by_class"][emotion_label] += 1

    # Chiudi i file
    for f in output_files.values():
        f.close()

    # ──────────────────────────────────────────────
    # REPORT FINALE
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("✅ JSONL costruiti con successo!")
    print("="*60)

    grand_total = 0
    for split in ["train", "validation", "test"]:
        s = stats[split]
        out_name = SPLIT_MAP[split]
        print(f"\n📊 {split} → {out_name}")
        print(f"   Totale: {s['total']}")

        print(f"   ── Distribuzione classi ──")
        for emo in ["neutral", "happiness", "surprise", "sadness",
                    "anger", "fear", "disgust", "contempt"]:
            n = s["by_class"].get(emo, 0)
            pct = 100 * n / s["total"] if s["total"] > 0 else 0
            print(f"     {emo:12s}: {n:6d} ({pct:5.1f}%)")

        grand_total += s["total"]

    print(f"\n📈 Totale immagini indicizzate: {grand_total}")
    print(f"📂 Output JSONL: {OUTPUT_JSONL}")

    print(f"\n📄 Esempio record dal training set:")
    with open(OUTPUT_JSONL / "train.jsonl", 'r') as f:
        first = json.loads(f.readline())
        print(json.dumps(first, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    build_jsonl()