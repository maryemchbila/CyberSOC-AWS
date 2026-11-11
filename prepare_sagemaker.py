
from pathlib import Path
import json
import numpy as np
import pandas as pd

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/sagemaker")
FEATURE_FILE = Path("reports/feature_columns.json")

with FEATURE_FILE.open(encoding="utf-8") as f:
    features = json.load(f)

assert len(features) == 70, "Nombre de features inattendu"

expected_rows = {
    "train": 1763292,
    "validation": 377789,
    "test": 378323
}

print("=" * 55)
print("CYBERSOC AWS - EXPORT SAGEMAKER XGBOOST")
print("=" * 55)

for split in ("train", "validation", "test"):
    files = sorted((INPUT_DIR / split).glob("*.parquet"))

    if not files:
        raise RuntimeError(f"Aucun fichier pour {split}")

    output_folder = OUTPUT_DIR / split
    output_folder.mkdir(parents=True, exist_ok=True)

    # Sécurité : éviter les anciens exports mélangés
    if list(output_folder.glob("*.csv")):
        raise RuntimeError(
            f"Des CSV existent déjà dans {output_folder}"
        )

    total = 0
    benign = 0
    attack = 0

    for i, file in enumerate(files):
        df = pd.read_parquet(file)

        columns = ["Target"] + features

        if not set(columns).issubset(df.columns):
            raise ValueError(f"Colonnes manquantes : {file}")

        output = df[columns].copy()

        if output.isna().any().any():
            raise ValueError(f"NaN détecté : {file}")

        if not np.isfinite(output.to_numpy()).all():
            raise ValueError(f"Infini détecté : {file}")

        if not output["Target"].isin([0, 1]).all():
            raise ValueError(f"Target invalide : {file}")

        n = len(output)
        total += n
        benign += int((output["Target"] == 0).sum())
        attack += int((output["Target"] == 1).sum())

        path = output_folder / f"part_{i:04d}.csv"

        output.to_csv(
            path,
            index=False,
            header=False,
            float_format="%.12g"
        )

    if total != expected_rows[split]:
        raise ValueError(
            f"{split}: {total} lignes au lieu de "
            f"{expected_rows[split]}"
        )

    print(f"\n{split.upper()}")
    print(f"Fichiers : {len(files)}")
    print(f"Lignes   : {total:,}")
    print(f"BENIGN   : {benign:,}")
    print(f"ATTACK   : {attack:,}")

print("\nEXPORT TERMINE")
print("Dossier : data/sagemaker/")
