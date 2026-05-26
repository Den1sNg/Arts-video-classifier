import csv
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm


INPUT_CSV = Path("/content/arts_working_links.csv")
OUTPUT_DIR = Path("/content/embeddings")
MANIFEST_CSV = Path("/content/embeddings_manifest.csv")
TMP_VIDEO = Path("/content/tmp_video.mp4")

NUM_FRAMES = 16
MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_VIDEOS = None  # можно поставить 200


def append_manifest(row: dict):
    exists = MANIFEST_CSV.exists()

    fieldnames = [
        "video_id",
        "youtube_url",
        "category_1",
        "category_2",
        "label",
        "embedding_path",
        "status",
        "error",
    ]

    with MANIFEST_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_done_ids() -> set[str]:
    if not MANIFEST_CSV.exists():
        return set()

    df = pd.read_csv(MANIFEST_CSV)
    ok = df[df["status"] == "ok"]
    return set(ok["video_id"].astype(str))


def download_video(url: str, output_path: Path):
    if output_path.exists():
        output_path.unlink()

    command = [
        "yt-dlp",
        "-f",
        "bv*[height<=360][ext=mp4][vcodec!*=av01]+ba[ext=m4a]/b[height<=360][ext=mp4][vcodec!*=av01]/b[height<=360][vcodec!*=av01]/18",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path),
        url,
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    if not output_path.exists():
        raise RuntimeError("video was not downloaded")


def extract_frames(video_path: Path, num_frames: int):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("could not open video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise RuntimeError("could not read frame count")

    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("no frames extracted")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames[:num_frames]


@torch.no_grad()
def make_clip_embeddings(frames, processor, model, device):
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    features = model.vision_model(pixel_values=inputs["pixel_values"])

    if hasattr(features, "pooler_output"):
        features = features.pooler_output

    features = model.visual_projection(features)
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return features.cpu()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    links = pd.read_csv(INPUT_CSV)
    if MAX_VIDEOS is not None:
        links = links.head(MAX_VIDEOS)

    done_ids = get_done_ids()
    print("Already done:", len(done_ids))
    print("Total links:", len(links))

    for _, row in tqdm(links.iterrows(), total=len(links), desc="Embedding videos"):
        video_id = str(row["video_id"])

        if video_id in done_ids:
            continue

        label = str(row["category_2"]).lower().replace(" ", "_").replace("&", "and")
        class_dir = OUTPUT_DIR / label
        class_dir.mkdir(parents=True, exist_ok=True)

        embedding_path = class_dir / f"{video_id}.pt"

        try:
            download_video(row["youtube_url"], TMP_VIDEO)
            frames = extract_frames(TMP_VIDEO, NUM_FRAMES)
            embeddings = make_clip_embeddings(frames, processor, model, device)

            torch.save(
                {
                    "video_id": video_id,
                    "youtube_url": row["youtube_url"],
                    "category_1": row["category_1"],
                    "category_2": row["category_2"],
                    "label": label,
                    "embedding": embeddings,
                    "num_frames": NUM_FRAMES,
                    "model": MODEL_NAME,
                },
                embedding_path,
            )

            append_manifest({
                "video_id": video_id,
                "youtube_url": row["youtube_url"],
                "category_1": row["category_1"],
                "category_2": row["category_2"],
                "label": label,
                "embedding_path": str(embedding_path),
                "status": "ok",
                "error": "",
            })

            done_ids.add(video_id)
            print(f"OK: {video_id} -> {tuple(embeddings.shape)}")

        except Exception as e:
            append_manifest({
                "video_id": video_id,
                "youtube_url": row["youtube_url"],
                "category_1": row["category_1"],
                "category_2": row["category_2"],
                "label": str(row["category_2"]),
                "embedding_path": str(embedding_path),
                "status": "failed",
                "error": str(e)[:500],
            })

            print(f"FAILED: {video_id}: {str(e)[:160]}")

        finally:
            if TMP_VIDEO.exists():
                TMP_VIDEO.unlink()

        time.sleep(1)

    print("Done")
    print("Embeddings dir:", OUTPUT_DIR)
    print("Manifest:", MANIFEST_CSV)


if __name__ == "__main__":
    main()
