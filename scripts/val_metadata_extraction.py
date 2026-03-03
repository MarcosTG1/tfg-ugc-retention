# -*- coding: utf-8 -*-
"""
Script para la extraccion de metadatos tecnicos de videos de validacion/test
con la etiqueta ECR asociada a cada video.

Extiende la funcionalidad de train_metadata_extraction.py anadiendo la columna
ECR al CSV de salida, uniendo por Id los metadatos extraidos con ffprobe y las
etiquetas ground-truth procedentes de un fichero de labels (val_truth.csv,
test_truth.csv o cualquier CSV que contenga las columnas Id y ECR).

Estructura de datos esperada en el servidor:
    /media/5tbraid/data/martugue/SnapUGC/
    ├── processed/
    │   └── train_metadata.csv
    └── raw/
        ├── test_data.csv
        ├── test_truth.csv
        ├── test_videos/
        ├── train_data.csv
        ├── train_videos/
        ├── val_data.csv
        ├── val_truth.csv
        └── val_videos/

Uso tipico (los defaults apuntan a val):
    python scripts/val_metadata_extraction.py

Para test, sobreescribir los tres argumentos:
    python scripts/val_metadata_extraction.py \
        --input  /media/5tbraid/data/martugue/SnapUGC/raw/test_videos \
        --labels /media/5tbraid/data/martugue/SnapUGC/raw/test_truth.csv \
        --output /media/5tbraid/data/martugue/SnapUGC/processed/test_metadata.csv
"""

import subprocess
import json
import csv
import os
import sys
import argparse
import logging
from typing import Dict, Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Carga de etiquetas ECR
# ---------------------------------------------------------------------------
def load_ecr_labels(labels_path: str) -> Dict[str, str]:
    """
    Lee un CSV que contenga al menos las columnas ``Id`` y ``ECR`` y devuelve
    un diccionario {id_video: ecr_value}.

    Soporta tanto los ficheros *_truth.csv (Id, ECR) como train_data.csv
    (Id, Title, Description, Download_link, ECR).

    Args:
        labels_path: Ruta al CSV con las etiquetas.

    Returns:
        Diccionario que mapea cada Id de video a su valor ECR (como string,
        para escribirlo tal cual en el CSV de salida sin perder precision).

    Raises:
        SystemExit: Si el fichero no existe, no tiene las columnas requeridas
                    o esta vacio.
    """
    if not os.path.isfile(labels_path):
        logging.error("El fichero de labels no existe: %s", labels_path)
        sys.exit(1)

    ecr_map: Dict[str, str] = {}
    with open(labels_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        # Validar que las columnas requeridas estan presentes
        if reader.fieldnames is None:
            logging.error("El fichero de labels esta vacio: %s", labels_path)
            sys.exit(1)

        missing = {"Id", "ECR"} - set(reader.fieldnames)
        if missing:
            logging.error(
                "Faltan columnas requeridas %s en %s (columnas encontradas: %s)",
                missing,
                labels_path,
                reader.fieldnames,
            )
            sys.exit(1)

        for row in reader:
            ecr_map[row["Id"]] = row["ECR"]

    logging.info(
        "Cargadas %d etiquetas ECR desde %s", len(ecr_map), labels_path
    )
    return ecr_map


# ---------------------------------------------------------------------------
# Extraccion de metadatos con ffprobe (identica a train_metadata_extraction.py)
# ---------------------------------------------------------------------------
def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extrae metadatos tecnicos de un archivo de video usando ffprobe.

    Args:
        video_path: Ruta completa al archivo de video.

    Returns:
        Diccionario con claves: duration, width, height, fps, has_audio,
        bitrate.  Los valores seran None/0 si ffprobe falla.
    """
    metadata: Dict[str, Any] = {
        "duration": None,
        "width": None,
        "height": None,
        "fps": None,
        "has_audio": 0,
        "bitrate": None,
    }

    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        output = subprocess.check_output(cmd)
        info = json.loads(output)

        # Informacion de formato general
        fmt = info.get("format", {})
        metadata["duration"] = float(fmt.get("duration", 0))
        metadata["bitrate"] = int(fmt.get("bit_rate", 0))

        # Streams
        streams = info.get("streams", [])
        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), {}
        )
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), None
        )

        if video_stream:
            metadata["width"] = int(video_stream.get("width", 0))
            metadata["height"] = int(video_stream.get("height", 0))

            fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
            if len(fps_parts) == 2 and int(fps_parts[1]) != 0:
                metadata["fps"] = round(
                    int(fps_parts[0]) / int(fps_parts[1]), 2
                )
            else:
                metadata["fps"] = 0

        metadata["has_audio"] = 1 if audio_stream else 0

    except (
        subprocess.CalledProcessError,
        ValueError,
        KeyError,
        ZeroDivisionError,
    ) as e:
        logging.error("Error procesando %s: %s", video_path, e)

    return metadata


# ---------------------------------------------------------------------------
# Procesado principal
# ---------------------------------------------------------------------------
def process_videos(
    video_dir: str,
    labels_path: str,
    output_csv: str,
) -> None:
    """
    Recorre los videos MP4 de ``video_dir``, extrae metadatos con ffprobe,
    anade la etiqueta ECR del fichero ``labels_path`` y escribe el resultado
    en ``output_csv``.

    Se garantiza que:
      - Solo se escriben filas para videos que existen fisicamente en disco
        Y tienen etiqueta ECR en el fichero de labels.
      - Se reportan warnings para discrepancias (videos sin etiqueta o
        etiquetas sin video).

    Args:
        video_dir:   Directorio con los .mp4.
        labels_path: CSV con columnas Id y ECR.
        output_csv:  Ruta de salida del CSV resultante.
    """
    if not os.path.isdir(video_dir):
        logging.error("El directorio de videos no existe: %s", video_dir)
        sys.exit(1)

    # 1. Cargar labels
    ecr_map = load_ecr_labels(labels_path)

    # 2. Listar videos
    files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    files.sort()  # orden determinista para reproducibilidad
    logging.info("Encontrados %d archivos .mp4 en %s", len(files), video_dir)

    # 3. Construir conjuntos de Ids para deteccion de discrepancias
    video_ids = {os.path.splitext(f)[0] for f in files}
    label_ids = set(ecr_map.keys())

    videos_sin_label = video_ids - label_ids
    labels_sin_video = label_ids - video_ids

    if videos_sin_label:
        logging.warning(
            "%d videos no tienen etiqueta ECR y seran omitidos: %s ...",
            len(videos_sin_label),
            list(videos_sin_label)[:5],
        )
    if labels_sin_video:
        logging.warning(
            "%d etiquetas ECR no tienen video correspondiente: %s ...",
            len(labels_sin_video),
            list(labels_sin_video)[:5],
        )

    # 4. Asegurar que el directorio de salida existe
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 5. Procesar y escribir
    written = 0
    skipped = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(
            ["Id", "duration", "width", "height", "fps", "has_audio", "bitrate", "ECR"]
        )

        for f in tqdm(files, desc="Extrayendo metadatos"):
            vid_id = os.path.splitext(f)[0]

            # Solo procesar si tenemos etiqueta ECR
            if vid_id not in ecr_map:
                skipped += 1
                continue

            path = os.path.join(video_dir, f)
            meta = get_video_metadata(path)

            writer.writerow([
                vid_id,
                meta["duration"],
                meta["width"],
                meta["height"],
                meta["fps"],
                meta["has_audio"],
                meta["bitrate"],
                ecr_map[vid_id],
            ])
            written += 1

    logging.info(
        "Extraccion completada: %d filas escritas, %d videos omitidos "
        "(sin etiqueta ECR). Resultados en: %s",
        written,
        skipped,
        output_csv,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extractor de metadatos de video para SnapUGC con etiqueta ECR. "
            "Combina la extraccion de metadatos tecnicos via ffprobe con el "
            "ground-truth de ECR de un fichero de labels."
        ),
    )
    # Raiz de datos en el servidor: /media/5tbraid/data/martugue/SnapUGC/
    DATA_ROOT = "/media/5tbraid/data/martugue/SnapUGC"

    parser.add_argument(
        "--input",
        default=os.path.join(DATA_ROOT, "raw", "val_videos"),
        help="Directorio de entrada con los videos .mp4",
    )
    parser.add_argument(
        "--labels",
        default=os.path.join(DATA_ROOT, "raw", "val_truth.csv"),
        help=(
            "CSV con las etiquetas ECR (columnas Id y ECR). "
            "Ejemplos: raw/val_truth.csv, raw/test_truth.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_ROOT, "processed", "val_metadata.csv"),
        help="Archivo CSV de salida con metadatos + ECR",
    )
    args = parser.parse_args()

    process_videos(args.input, args.labels, args.output)


if __name__ == "__main__":
    main()
