import csv
import random
import subprocess
import time
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("/content/arts_video_links.csv")
WORKING_CSV = Path("/content/arts_working_links.csv")
CHECKED_CSV = Path("/content/arts_checked_links.csv")

TARGET_TOTAL = 1100

MIN_SLEEP = 2.0
MAX_SLEEP = 5.0
SAVE_EVERY = 20


def load_existing_checked() -> set[str]:
    checked = set()

    if CHECKED_CSV.exists():
        df = pd.read_csv(CHECKED_CSV)
        checked.update(df["video_id"].astype(str).tolist())

    if WORKING_CSV.exists():
        df = pd.read_csv(WORKING_CSV)
        checked.update(df["video_id"].astype(str).tolist())

    return checked


def load_existing_working_ids() -> set[str]:
    if not WORKING_CSV.exists():
        return set()

    df = pd.read_csv(WORKING_CSV)
    return set(df["video_id"].astype(str).tolist())


def append_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def check_url(url: str) -> tuple[bool, str]:
    command = [
        "yt-dlp",
        "--simulate",
        "--quiet",
        "--no-warnings",
        url,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90,
    )

    if result.returncode == 0:
        return True, ""

    error_lines = result.stderr.strip().splitlines()
    return False, error_lines[-1] if error_lines else "unknown error"


def main() -> None:
    source_df = pd.read_csv(INPUT_CSV)

    source_df = source_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    checked_ids = load_existing_checked()
    working_ids = load_existing_working_ids()

    print("Already checked or working:", len(checked_ids))
    print("Already working:", len(working_ids))
    print("Target working total:", TARGET_TOTAL)

    checked_fieldnames = [
        "video_id",
        "youtube_url",
        "category_1",
        "category_2",
        "rank",
        "task_id",
        "status",
        "error",
    ]

    working_fieldnames = [
        "video_id",
        "youtube_url",
        "category_1",
        "category_2",
        "rank",
        "task_id",
    ]

    checked_this_run = 0

    for _, row in source_df.iterrows():
        video_id = str(row["video_id"])

        if video_id in checked_ids:
            continue

        if len(working_ids) >= TARGET_TOTAL:
            break

        url = row["youtube_url"]
        category = row["category_2"]

        print(f"[{len(working_ids)}/{TARGET_TOTAL}] Checking {video_id} | {category}")

        try:
            ok, error = check_url(url)
        except subprocess.TimeoutExpired:
            ok, error = False, "timeout"

        checked_row = {
            "video_id": video_id,
            "youtube_url": url,
            "category_1": row["category_1"],
            "category_2": row["category_2"],
            "rank": row["rank"],
            "task_id": row["task_id"],
            "status": "ok" if ok else "failed",
            "error": error,
        }

        append_row(CHECKED_CSV, checked_row, checked_fieldnames)
        checked_ids.add(video_id)
        checked_this_run += 1

        if ok:
            working_row = {
                "video_id": video_id,
                "youtube_url": url,
                "category_1": row["category_1"],
                "category_2": row["category_2"],
                "rank": row["rank"],
                "task_id": row["task_id"],
            }

            append_row(WORKING_CSV, working_row, working_fieldnames)
            working_ids.add(video_id)

            print("  OK")
        else:
            print("  FAILED:", error[:160])

        if checked_this_run % SAVE_EVERY == 0:
            print(f"Progress saved. Checked this run: {checked_this_run}, working: {len(working_ids)}")

        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

    print("Done")
    print("Working total:", len(working_ids))
    print("Working CSV:", WORKING_CSV)
    print("Checked CSV:", CHECKED_CSV)


if __name__ == "__main__":
    main()
