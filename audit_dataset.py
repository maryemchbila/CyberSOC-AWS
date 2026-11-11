
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd

CLEAN_DIR = Path("data/cleaned")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(CLEAN_DIR.rglob("*.parquet"))

if not files:
    raise SystemExit("Aucun fichier Parquet trouvé.")

seen_hashes = set()
labels = Counter()
duplicate_labels = Counter()

total_rows = 0
total_duplicates = 0
total_missing = 0
total_infinite = 0
reference_columns = None
schema_errors = []

print("=" * 60)
print("CYBERSOC AWS - AUDIT DES DONNEES NETTOYEES")
print("=" * 60)

for i, file in enumerate(files, 1):
    df = pd.read_parquet(file)

    total_rows += len(df)

    if "Label" not in df.columns:
        raise ValueError(f"Label absent : {file}")

    # Vérifier la cohérence des colonnes
    cols = list(df.columns)

    if reference_columns is None:
        reference_columns = cols
    elif cols != reference_columns:
        schema_errors.append(str(file))

    labels.update(df["Label"].value_counts().to_dict())

    # Vérifier les valeurs invalides
    total_missing += int(df.isna().sum().sum())

    numeric = df.select_dtypes(include="number")
    total_infinite += int(
        np.isinf(numeric.to_numpy()).sum()
    )

    # Détecter les doublons exacts sur toutes les colonnes
    row_hashes = pd.util.hash_pandas_object(
        df, index=False
    ).to_numpy()

    for row_hash, label in zip(
        row_hashes, df["Label"].to_numpy()
    ):
        h = int(row_hash)

        if h in seen_hashes:
            total_duplicates += 1
            duplicate_labels[str(label)] += 1
        else:
            seen_hashes.add(h)

    if i % 10 == 0 or i == len(files):
        print(f"Fichiers analyses : {i}/{len(files)}")

report = {
    "total_rows": total_rows,
    "exact_duplicate_rows": total_duplicates,
    "unique_row_hashes": len(seen_hashes),
    "duplicate_rate_percent": round(
        100 * total_duplicates / total_rows, 4
    ),
    "missing_values": total_missing,
    "infinite_values": total_infinite,
    "schema_errors": schema_errors,
    "label_distribution": dict(labels),
    "duplicate_labels": dict(duplicate_labels)
}

output = REPORT_DIR / "audit_report.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n========== RESULTATS ==========")
print("Lignes :", total_rows)
print("Doublons exacts :", total_duplicates)
print("Taux de doublons :",
      report["duplicate_rate_percent"], "%")
print("Valeurs manquantes :", total_missing)
print("Valeurs infinies :", total_infinite)
print("Incoherences de schema :", len(schema_errors))

print("\nDoublons par classe :")
for label, count in duplicate_labels.most_common():
    print(f"  {label}: {count:,}")

print("\nDistribution des classes :")
for label, count in labels.most_common():
    print(f"  {label}: {count:,}")

print("\nRapport :", output)
