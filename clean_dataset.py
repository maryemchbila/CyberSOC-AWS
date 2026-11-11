
from pathlib import Path
import json
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")
REPORT_DIR = Path("reports")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50000

def normalize_label(value):
    label = str(value).strip()

    if label.startswith("Web Attack"):
        if "Brute Force" in label:
            return "Web Attack - Brute Force"
        if "XSS" in label:
            return "Web Attack - XSS"
        if "Sql Injection" in label:
            return "Web Attack - Sql Injection"

    return label


def clean_chunk(df):
    # Harmonisation des noms de colonnes
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("Colonne Label introuvable")

    # Correction des labels Web Attack
    df["Label"] = df["Label"].map(normalize_label)

    # Conserver les colonnes originales
    feature_columns = [
        col for col in df.columns if col != "Label"
    ]

    # Conversion numérique
    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Remplacement des valeurs infinies
    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Suppression des lignes avec données invalides
    invalid_rows = int(
        df[feature_columns].isna().any(axis=1).sum()
    )

    df = df.dropna(subset=feature_columns).copy()

    return df, invalid_rows


def main():
    files = sorted(RAW_DIR.rglob("*.csv"))

    if not files:
        raise SystemExit("Aucun CSV dans data/raw")

    report = {
        "files": [],
        "total_before": 0,
        "total_after": 0,
        "total_removed": 0
    }

    for file in files:
        print(f"\nNettoyage : {file.name}")

        before = 0
        after = 0
        removed = 0

        for i, chunk in enumerate(
            pd.read_csv(
                file,
                chunksize=CHUNK_SIZE,
                low_memory=False,
                encoding="utf-8-sig"
            )
        ):
            before += len(chunk)

            cleaned, invalid = clean_chunk(chunk)

            after += len(cleaned)
            removed += invalid

            # Un fichier Parquet par bloc
            output_folder = CLEAN_DIR / file.stem
            output_folder.mkdir(
                parents=True, exist_ok=True
            )

            output_file = (
                output_folder / f"part_{i:04d}.parquet"
            )

            cleaned.to_parquet(
                output_file,
                index=False
            )

        print(f"Lignes originales : {before:,}")
        print(f"Lignes propres    : {after:,}")
        print(f"Lignes supprimées : {removed:,}")

        report["files"].append({
            "file": file.name,
            "before": before,
            "after": after,
            "removed": removed
        })

        report["total_before"] += before
        report["total_after"] += after
        report["total_removed"] += removed

    output = REPORT_DIR / "cleaning_report.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    print("NETTOYAGE TERMINÉ")
    print("=" * 55)
    print("Avant :", report["total_before"])
    print("Après :", report["total_after"])
    print("Lignes supprimées :", report["total_removed"])
    print("Rapport :", output)


if __name__ == "__main__":
    main()
