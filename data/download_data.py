"""Download the NASA C-MAPSS turbofan engine degradation dataset.

Source: A. Saxena & K. Goebel (2008), "Turbofan Engine Degradation Simulation
Data Set", NASA Prognostics Center of Excellence (PCoE), NASA Ames.
Mirror: PHM Society dataset repository (verified reachable 2026-10-18,
HTTP 200).

Contents: FD001–FD004 sub-datasets, each with train_*.txt, test_*.txt and
RUL_*.txt. 26 columns: unit id, cycle, 3 operational settings, 21 sensors.
Space-delimited ASCII, no missing values.

Run from the repo root:  python data/download_data.py
"""
import os
import urllib.request
import zipfile

# PHM Society mirror of the NASA PCoE repository (verified live 2026-10-18)
URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
# NASA's own page also hosts it:
# https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED = [
    "train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt",
    "test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt",
    "RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt",
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    missing = [f for f in EXPECTED if not os.path.exists(os.path.join(DATA_DIR, f))]
    if not missing:
        print(f"All {len(EXPECTED)} files already present in {DATA_DIR}. Nothing to do.")
        return
    zip_path = os.path.join(DATA_DIR, "cmapss.zip")
    print("Downloading C-MAPSS dataset from PHM Society mirror ...")
    urllib.request.urlretrieve(URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        # the mirror nests the data one level deep: outer zip -> CMAPSSData.zip
        # -> the .txt files. Flatten everything into data/.
        def iter_txt_files(zf):
            for member in zf.namelist():
                if member.endswith(".zip"):
                    with zf.open(member) as src:
                        inner_path = os.path.join(DATA_DIR, "_inner.zip")
                        with open(inner_path, "wb") as dst:
                            dst.write(src.read())
                    with zipfile.ZipFile(inner_path) as z2:
                        yield from iter_txt_files(z2)
                    os.remove(inner_path)
                elif member.endswith(".txt"):
                    yield member, zf

        for member, zf in iter_txt_files(z):
            target = os.path.join(DATA_DIR, os.path.basename(member))
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    os.remove(zip_path)
    n = len([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
    print(f"Done. {n} .txt files in {DATA_DIR}")
    absent = [f for f in EXPECTED if not os.path.exists(os.path.join(DATA_DIR, f))]
    if absent:
        print(f"WARNING: missing after extract: {absent}")


if __name__ == "__main__":
    main()
