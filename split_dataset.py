
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd

# ===============================================
# CONFIGURATION
# ===============================================

INPUT_DIR = Path("data/ml_ready")
OUTPUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

FEATURES_FILE = REPORT_DIR / "feature_columns.json"

RANDOM_SEED = 42

SPLITS = {
    "train": (0, 7000),
    "validation": (7000, 8500),
    "test": (8500, 10000),
}

files = sorted(INPUT_DIR.rglob("*.parquet"))

if not files:
    raise SystemExit("Aucun fichier Parquet dans data/ml_ready")

if OUTPUT_DIR.exists() and any(
    OUTPUT_DIR.rglob("*.parquet")
):
    raise SystemExit(
        "Des partitions existent deja dans data/processed. "
        "Conserve ou deplace-les avant de relancer."
    )

with FEATURES_FILE.open(encoding="utf-8") as f:
    feature_columns = json.load(f)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

stats = {
    name: {
        "rows": 0,
        "binary": Counter(),
        "original_labels": Counter(),
        "source_files": Counter(),
    }
    for name in SPLITS
}

# ===============================================
# HASH DETERMINISTE DES FEATURES
# ===============================================

def assign_partitions(df):
    # Les features identiques recoivent le meme hash.
    hashes = pd.util.hash_pandas_object(
        df[feature_columns],
        index=False
    ).to_numpy(dtype=np.uint64)

    # Melange reproductible des bits avec une graine.
    x = hashes ^ np.uint64(RANDOM_SEED)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xbf58476d1ce4e5b9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)

    return x % np.uint64(10000)


# ===============================================
# CREATION DES PARTITIONS
# ===============================================

print("=" * 60)
print("CYBERSOC AWS - TRAIN / VALIDATION / TEST")
print("=" * 60)

for i, file in enumerate(files, 1):

    df = pd.read_parquet(file)

    expected = feature_columns + ["Label"]

    if list(df.columns) != expected:
        raise ValueError(
            f"Colonnes inattendues dans : {file}"
        )

    # Classification binaire
    df["Target"] = (
        df["Label"] != "BENIGN"
    ).astype("int8")

    # Label original conserve uniquement pour l'analyse
    df = df.rename(columns={
        "Label": "OriginalLabel"
    })

    # La source sera utile pour analyser les biais.
    df["SourceFile"] = file.parent.name

    buckets = assign_partitions(df)

    for name, (low, high) in SPLITS.items():

        mask = (buckets >= low) & (buckets < high)

        subset = df.loc[mask].copy()

        if subset.empty:
            continue

        # Colonnes du modele + metadonnees d'evaluation
        ordered = (
            feature_columns
            + ["Target", "OriginalLabel", "SourceFile"]
        )

        subset = subset[ordered]

        destination = OUTPUT_DIR / name
        destination.mkdir(
            parents=True, exist_ok=True
        )

        # Nom unique pour chaque fichier source
        output_name = (
            file.parent.name + "__" + file.name
        )

        subset.to_parquet(
            destination / output_name,
            index=False
        )

        stats[name]["rows"] += len(subset)

        stats[name]["binary"].update(
            subset["Target"].value_counts().to_dict()
        )

        stats[name]["original_labels"].update(
            subset["OriginalLabel"].value_counts().to_dict()
        )

        stats[name]["source_files"].update(
            subset["SourceFile"].value_counts().to_dict()
        )

    if i % 10 == 0 or i == len(files):
        print(f"Fichiers traites : {i}/{len(files)}")

# ===============================================
# RAPPORT FINAL
# ===============================================

report = {
    "method": "deterministic_feature_hash_split",
    "seed": RANDOM_SEED,
    "features": len(feature_columns),
    "splits": {},
}

for name, result in stats.items():

    report["splits"][name] = {
        "rows": result["rows"],
        "binary_distribution": dict(
            result["binary"]
        ),
        "original_label_distribution": dict(
            result["original_labels"]
        ),
        "source_file_distribution": dict(
            result["source_files"]
        )
    }

    print(f"\n{name.upper()}")
    print("Lignes :", f'{result["rows"]:,}')
    print("BENIGN :", f'{result["binary"][0]:,}')
    print("ATTACK :", f'{result["binary"][1]:,}')

total = sum(x["rows"] for x in stats.values())

print("\nTOTAL :", f"{total:,}")

if total != 2_519_404:
    raise ValueError(
        f"Total inattendu : {total}"
    )

report_file = REPORT_DIR / "split_report.json"

with report_file.open("w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nPartitionnement termine.")
print("Rapport :", report_file)
print("Fichiers :", OUTPUT_DIR)
