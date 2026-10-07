"""Download and prepare the ISIC 2018 Task 3 training dataset."""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "configs" / "HAM10000_metadata.csv"

INPUT_URL = (
    "https://isic-challenge-data.s3.amazonaws.com/2018/"
    "ISIC2018_Task3_Training_Input.zip"
)
GROUNDTRUTH_URL = (
    "https://isic-challenge-data.s3.amazonaws.com/2018/"
    "ISIC2018_Task3_Training_GroundTruth.zip"
)
METADATA_URL = "https://dataverse.harvard.edu/api/access/datafile/3172582"
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    print(f"\rProgress: {percent:6.2f}%", end="", flush=True)


def download_file(url: str, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Already exists: {output}")
        return
    print(f"\nDownloading:\n{url}\nTarget: {output}\n")
    urllib.request.urlretrieve(url, output, reporthook=progress)
    print()


def extract_zip(zip_file: Path, output_dir: Path):
    if output_dir.exists():
        print(f"Already extracted: {output_dir}")
        return
    print(f"Extracting: {zip_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(output_dir)
    print("Extraction complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "skin_cancer_data"))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    metadata_path = Path(args.metadata).resolve()
    downloads = ROOT / "downloads"
    input_zip = downloads / "ISIC2018_Task3_Training_Input.zip"
    groundtruth_zip = downloads / "ISIC2018_Task3_Training_GroundTruth.zip"
    input_dir = downloads / "ISIC2018_Task3_Training_Input"
    groundtruth_dir = downloads / "ISIC2018_Task3_Training_GroundTruth"

    print("=" * 68)
    print("ISIC 2018 Task 3 dataset preparation")
    print("=" * 68)

    download_file(INPUT_URL, input_zip)
    download_file(GROUNDTRUTH_URL, groundtruth_zip)
    download_file(METADATA_URL, metadata_path)

    extract_zip(input_zip, input_dir)
    extract_zip(groundtruth_zip, groundtruth_dir)

    df = pd.read_csv(metadata_path)
    required = {"image_id", "dx", "lesion_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Metadata is missing columns: {sorted(missing)}")

    data_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (data_dir / cls).mkdir(parents=True, exist_ok=True)

    # The official input archive may contain an extra top-level directory.
    image_candidates = [p for p in input_dir.rglob("*.jpg")]
    image_map = {p.stem: p for p in image_candidates}

    copied = 0
    missing_images = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Organizing images"):
        image_id = str(row.image_id)
        label = str(row.dx).upper()
        src = image_map.get(image_id)
        if src is None:
            missing_images += 1
            continue

        dst = data_dir / label / f"{image_id}.jpg"
        if dst.exists():
            continue

        try:
            # Hard-link when possible to avoid a second full image copy.
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
        copied += 1

    print("\n" + "=" * 68)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 68)
    print(f"Metadata rows : {len(df)}")
    print(f"Organized     : {copied}")
    print(f"Missing images: {missing_images}")
    print("\nClass counts:")
    for cls in CLASSES:
        count = len(list((data_dir / cls).glob("*.jpg")))
        print(f"{cls:6s}: {count}")


if __name__ == "__main__":
    main()
