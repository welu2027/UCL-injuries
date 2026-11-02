"""
Step 1 — download the Driveline OpenBiomechanics pitching data, verify it, convert to parquet.

Downloads ~240 MB of zipped full-signal tables from the dataset's GitHub release (tag dataset-v1),
checks each against the SHA-256 published in the OpenBiomechanics repo, unzips them into
data/raw/, and writes fast parquet copies into data/.

Data license: CC BY-NC-SA 4.0 (Driveline Baseball). Non-commercial use, attribution, share-alike.
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
REPO = "https://github.com/drivelineresearch/openbiomechanics"
RAW_URL = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_pitching/data"

# from scripts/release_checksums.sha256 in the OpenBiomechanics repo
ASSETS = {
    "pitching_landmarks.zip": "efcc078d5d8e77ed1fbf0b9f6a78f750c57a624165f71f30e34142b74db7ec22",
    "pitching_forces_moments.zip": "ff96d23cfccd3a43473c591166fab1c4317019cd0aec999611e8dc8401246a02",
    "pitching_joint_angles.zip": "b4c4854ffcc52a0f757e0b3a3c86acdc4a2fc2e50516a76e62aabdb916afed46",
    "pitching_joint_velos.zip": "5752f571b40acc4f02afc42dcfdb4b9eb191fcffd5fb25bc3c585e149941fb69",
    "pitching_energy_flow.zip": "7732e35fbb3e53a2a6bc87b5118fcd6b7bc81496dffd0156f5d3561cc7854467",
    "pitching_force_plate.zip": "f36a1722e23101be2387acc8f91ec38d2ee5d640388f688af7fd8fa52eef29ed",
}
TABLES = ["landmarks", "forces_moments", "joint_angles", "joint_velos", "energy_flow", "force_plate"]
SMALL = {"metadata.csv": "metadata.csv", "poi/poi_metrics.csv": "poi_metrics.csv",
         "data_dictionary.csv": "data_dictionary.csv"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        z = RAW / name
        if not z.exists() or sha256(z) != expected:
            download(f"{REPO}/releases/download/dataset-v1/{name}", z)
        got = sha256(z)
        if got != expected:
            sys.exit(f"checksum mismatch for {name}: {got}")
        print(f"  verified {name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)

    for src, dst in SMALL.items():
        if not (DATA / dst).exists():
            download(f"{RAW_URL}/{src}", DATA / dst)

    for t in TABLES:
        out = DATA / f"{t}.parquet"
        if not out.exists():
            print(f"  converting {t}.csv -> parquet")
            pd.read_csv(RAW / f"{t}.csv").to_parquet(out)
    print("data ready in", DATA)


if __name__ == "__main__":
    main()
