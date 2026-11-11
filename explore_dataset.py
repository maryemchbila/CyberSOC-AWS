
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import json

# =====================================================
# CONFIGURATION
# =====================================================

DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50000

# Rechercher tous les fichiers CSV
files = sorted(DATA_DIR.rglob("*.csv"))

if not files:
    raise SystemExit(
        "ERREUR : Aucun fichier CSV trouvé dans data/raw/"
    )

print("=" * 60)
print("CYBERSOC AWS - EXPLORATION CIC-IDS2017")
print("=" * 60)

print(f"Fichiers CSV trouvés : {len(files)}")

if len(files) != 8:
    print("ATTENTION : 8 fichiers CSV sont attendus.")

# =====================================================
# VARIABLES POUR LES STATISTIQUES
# =====================================================

total_rows = 0

global_labels = Counter()
global_missing = Counter()
global_infinite = Counter()

file_reports = []

# =====================================================
# ANALYSE DES FICHIERS
# =====================================================

for file in files:

    print(f"\nAnalyse : {file.name}")

    rows = 0
    columns = None
    file_labels = Counter()

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8-sig"
    ):

        # Nettoyer les noms des colonnes
        chunk.columns = chunk.columns.str.strip()

        if columns is None:
            columns = list(chunk.columns)

        # Nombre de lignes
        rows += len(chunk)

        # Valeurs manquantes
        missing_counts = chunk.isna().sum()

        global_missing.update(
            missing_counts.to_dict()
        )

        # Distribution des labels
        if "Label" in chunk.columns:

            counts = (
                chunk["Label"]
                .astype(str)
                .str.strip()
                .value_counts()
            )

            file_labels.update(counts.to_dict())
            global_labels.update(counts.to_dict())

        # Valeurs infinies dans les colonnes numériques
        numeric = chunk.select_dtypes(
            include="number"
        )

        inf_counts = np.isinf(
            numeric.to_numpy()
        ).sum(axis=0)

        for name, count in zip(
            numeric.columns,
            inf_counts
        ):
            if count:
                global_infinite[name] += int(count)

    total_rows += rows

    file_reports.append({
        "file": file.name,
        "rows": rows,
        "columns": len(columns),
        "labels": dict(file_labels)
    })

    print(f"Nombre de lignes : {rows:,}")
    print(f"Nombre de colonnes : {len(columns)}")
    print(f"Classes : {dict(file_labels)}")

# =====================================================
# RAPPORT GLOBAL
# =====================================================

report = {
    "dataset": "CIC-IDS2017",
    "total_files": len(files),
    "total_rows": total_rows,
    "files": file_reports,
    "label_distribution": dict(global_labels),
    "missing_values_by_column": dict(global_missing),
    "infinite_values_by_column": dict(global_infinite)
}

output = REPORT_DIR / "data_quality_report.json"

with output.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )

# =====================================================
# AFFICHAGE DES RESULTATS
# =====================================================

print("\n" + "=" * 60)
print("RESUME GLOBAL DU DATASET")
print("=" * 60)

print(f"Nombre de fichiers : {len(files)}")
print(f"Nombre total de flux : {total_rows:,}")

print("\nDISTRIBUTION DES ATTAQUES :")

for label, count in global_labels.most_common():

    percentage = (
        count / total_rows * 100
        if total_rows else 0
    )

    print(
        f"{label:30s} "
        f"{count:10,d} "
        f"({percentage:.2f} %)"
    )

print("\nQUALITE DES DONNEES :")

print(
    "Valeurs manquantes :",
    sum(global_missing.values())
)

print(
    "Valeurs infinies :",
    sum(global_infinite.values())
)

print("\nColonnes contenant des valeurs manquantes :")

for name, count in global_missing.items():
    if count > 0:
        print(f"  {name}: {count:,}")

print("\nColonnes contenant des valeurs infinies :")

for name, count in global_infinite.items():
    if count > 0:
        print(f"  {name}: {count:,}")

print("\nRapport sauvegardé dans :", output)

print("\nEXPLORATION TERMINEE")
