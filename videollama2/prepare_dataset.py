"""
Genera train.json y test.json (desde val CSV) para el entrenamiento de VideoLLaMA2 EVQA.

- train.json  → dataset de entrenamiento (106,192 vídeos)
- test.json   → dataset de validación oficial usado como eval_dataset en el Trainer
                (renombrado 'test' porque make_supervised_data_module hace
                 data_path.replace('train','test'))

Ejecutar una vez antes de lanzar train.sh:
  conda activate videollama2
  python prepare_dataset.py

Los JSON resultantes se guardan en PROCESSED_DIR.
"""

import pandas as pd
import json
import os
from decord import VideoReader, cpu

# ── Rutas del servidor ────────────────────────────────────────────────────────
RAW_DIR       = "/media/5tbraid/data/martugue/SnapUGC/raw"
PROCESSED_DIR = "/media/5tbraid/data/martugue/SnapUGC/processed"

TRAIN_CSV    = os.path.join(RAW_DIR, "train_data.csv")
VAL_CSV      = os.path.join(RAW_DIR, "val_data.csv")
TRAIN_VIDEOS = os.path.join(RAW_DIR, "train_videos")
VAL_VIDEOS   = os.path.join(RAW_DIR, "val_videos")

OUTPUT_TRAIN = os.path.join(PROCESSED_DIR, "train.json")
OUTPUT_TEST  = os.path.join(PROCESSED_DIR, "test.json")   # val data, nombre 'test' para el Trainer
OUTPUT_VAL   = os.path.join(PROCESSED_DIR, "val.json")    # mismo contenido, para run_validation.sh
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(PROCESSED_DIR, exist_ok=True)


def check_video(video_path):
    """Comprueba que el archivo de vídeo existe y se puede leer."""
    try:
        if not os.path.exists(video_path):
            print(f"  [MISSING] {video_path}")
            return False
        vr = VideoReader(video_path, ctx=cpu(0))
        if len(vr) == 0:
            print(f"  [EMPTY]   {video_path}")
            return False
        return True
    except Exception as e:
        print(f"  [ERROR]   {video_path}: {e}")
        return False


def create_conversation(title, description):
    title       = "None" if pd.isna(title)       else str(title)
    description = "None" if pd.isna(description) else str(description)
    return [
        {
            "from": "human",
            "value": (
                f"<video>\nHow would you judge the engagement continuation rate of the "
                f"given content, where engagement continuation rate represents the "
                f"probability of watch time exceeding 5 seconds. "
                f"The title of the video is {title}, and the description of the video is {description}"
            )
        },
        {
            "from": "gpt",
            "value": "The engagement continuation rate of the video."
        }
    ]


def build_json(csv_path, video_dir, label="dataset"):
    df = pd.read_csv(csv_path)
    has_ecr = "ECR" in df.columns

    entries = []
    total   = len(df)
    skipped = 0

    print(f"\n[{label}] Procesando {total:,} muestras desde {csv_path} ...")
    for idx, row in df.iterrows():
        video_path = os.path.join(video_dir, f"{row['Id']}.mp4")
        if not check_video(video_path):
            skipped += 1
            continue

        entry = {
            "id":            row["Id"],
            "video":         video_path,
            "conversations": create_conversation(row.get("Title"), row.get("Description")),
        }
        if has_ecr:
            entry["ECR"] = float(row["ECR"]) * 100  # escala [0,1] → [0,100]

        entries.append(entry)

        if len(entries) % 1000 == 0:
            print(f"  {len(entries):,} válidos / {idx+1:,} procesados ...")

    print(f"[{label}] Válidos: {len(entries):,} | Omitidos: {skipped:,} / {total:,}")
    return entries


def main():
    # ── TRAIN ────────────────────────────────────────────────────────────────
    train_entries = build_json(TRAIN_CSV, TRAIN_VIDEOS, label="TRAIN")
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    print(f"→ Guardado: {OUTPUT_TRAIN}  ({len(train_entries):,} muestras)")

    # ── VAL → test.json + val.json ───────────────────────────────────────────
    val_entries = build_json(VAL_CSV, VAL_VIDEOS, label="VAL")
    for out_path in [OUTPUT_TEST, OUTPUT_VAL]:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(val_entries, f, ensure_ascii=False, indent=2)
        print(f"→ Guardado: {out_path}  ({len(val_entries):,} muestras)")

    print("\nListo. Ahora puedes lanzar train.sh")


if __name__ == "__main__":
    main()
