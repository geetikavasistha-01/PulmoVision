"""
PulmoVision — Programmatic Dataset Access & Downloader
Downloads and extracts the COVID-CTset dataset from Kaggle into data/raw/.
"""

import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


DATASET_SLUG = "mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset"
TARGET_DIR = "data/raw"


def download_covid_ctset(target_dir: str = TARGET_DIR, force: bool = False):
    """
    Downloads and extracts COVID-CTset from Kaggle.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "covidctset-a-large-covid19-ct-scans-dataset.zip")

    # Check if already downloaded or extracted
    extracted_flag = os.path.join(target_dir, ".extracted")
    if os.path.exists(extracted_flag) and not force:
        print(f"[Dataset Access] Dataset already extracted in: {target_dir}")
        return target_dir

    print(f"\n[Dataset Access] Initializing Kaggle API to download: {DATASET_SLUG}")
    api = KaggleApi()
    api.authenticate()

    print(f"[Dataset Access] Downloading archive to {target_dir} (this may take a few minutes)...")
    api.dataset_download_files(DATASET_SLUG, path=target_dir, unzip=True, quiet=False)

    with open(extracted_flag, "w") as f:
        f.write("download_complete")

    print(f"[Dataset Access] Successfully downloaded and extracted COVID-CTset to: {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_covid_ctset()

