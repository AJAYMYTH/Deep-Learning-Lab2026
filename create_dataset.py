import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


DATASET_ID = "Project-AgML/tomato_leaf_disease"
DATASET_REVISION = "76a2f52138563e7b68ccda4d9de02e43c7fe0c94"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
SOURCE_IMAGES_PER_CLASS = 1100
SOURCE_IMAGE_COUNT = SOURCE_IMAGES_PER_CLASS * 10
CLASS_NAMES = (
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites Two-spotted Spider Mite",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
)


def get_json(url: str, params: dict[str, str]) -> dict:
    for attempt in range(6):
        try:
            response = requests.get(url, params=params, timeout=60)
        except requests.RequestException:
            if attempt == 5:
                raise
        else:
            if response.ok:
                return response.json()
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == 5:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 30)
            time.sleep(delay)
            continue
        time.sleep(min(2**attempt, 30))
    raise RuntimeError("Could not retrieve dataset rows.")


def collect_images(samples_per_class: int) -> list[tuple[int, str]]:
    if samples_per_class > SOURCE_IMAGES_PER_CLASS:
        raise ValueError(
            f"The source dataset has at most {SOURCE_IMAGES_PER_CLASS} images per class."
        )

    class_offsets: dict[int, int] = {}
    for block in range(len(CLASS_NAMES)):
        offset = block * SOURCE_IMAGES_PER_CLASS
        rows = get_json(
            ROWS_URL,
            {
                "dataset": DATASET_ID,
                "config": "default",
                "split": "train",
                "offset": str(offset),
                "length": "1",
            },
        )
        if rows["num_rows_total"] != SOURCE_IMAGE_COUNT or not rows["rows"]:
            raise RuntimeError("The source dataset layout no longer matches this downloader.")
        label = rows["rows"][0]["row"]["label"]
        class_offsets[label] = offset
        time.sleep(1)

    if len(class_offsets) != len(CLASS_NAMES):
        raise RuntimeError("Could not map every class in the source dataset.")

    selected: list[tuple[int, str]] = []
    for label in range(len(CLASS_NAMES)):
        offset = class_offsets[label]
        remaining = samples_per_class
        while remaining:
            rows = get_json(
                ROWS_URL,
                {
                    "dataset": DATASET_ID,
                    "config": "default",
                    "split": "train",
                    "offset": str(offset),
                    "length": str(min(100, remaining)),
                },
            )
            items = rows["rows"]
            if not items:
                raise RuntimeError(
                    f"Source rows ended before collecting enough {CLASS_NAMES[label]} images."
                )
            for item in items:
                row = item["row"]
                if row["label"] != label:
                    raise RuntimeError(
                        f"Unexpected class label at source row {offset}: {row['label']}."
                    )
                selected.append((label, row["image"]["src"]))
            remaining -= len(items)
            offset += len(items)
            if remaining:
                time.sleep(1)

    return selected


def download_image(task: tuple[int, int, str, Path]) -> None:
    label, index, image_url, output_dir = task
    for attempt in range(3):
        response = requests.get(image_url, timeout=60)
        if response.ok:
            image_path = output_dir / CLASS_NAMES[label] / f"{index:04d}.jpg"
            image_path.write_bytes(response.content)
            return
        if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
            response.raise_for_status()
        time.sleep(2**attempt)
    raise RuntimeError(f"Could not download image {index} for {CLASS_NAMES[label]}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a balanced tomato-leaf dataset from Project-AgML."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Number of images to download for each class (default: 100).",
    )
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be at least 1.")

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "dataset"
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"{output_dir} already exists and is not empty.")

    images = collect_images(args.samples_per_class)
    for class_name in CLASS_NAMES:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    class_counts = [0] * len(CLASS_NAMES)
    tasks = []
    for label, image_url in images:
        index = class_counts[label]
        class_counts[label] += 1
        tasks.append((label, index, image_url, output_dir))
    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(download_image, tasks))

    metadata = {
        "source": f"https://huggingface.co/datasets/{DATASET_ID}",
        "revision": DATASET_REVISION,
        "license": "CC0-1.0",
        "samples_per_class": args.samples_per_class,
        "classes": list(CLASS_NAMES),
    }
    (output_dir / "DATASET_INFO.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(images)} images across {len(CLASS_NAMES)} classes in {output_dir}")


if __name__ == "__main__":
    main()
