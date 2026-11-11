
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd

INPUT_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("data/deduplicated")
REPORT_DIR = Path("reports")

files = sorted(INPUT_DIR.rglob("*.parquet"))

if not files:
    raise SystemExit("Aucun fichier Parquet trouvé.")

# Ne pas écraser un résultat existant
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*")):
    raise SystemExit(
        "Le dossier data/deduplicated existe déjà. "
        "Déplace-le avant de relancer le script."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

seen_hashes = set()
removed_by_class = Counter()
remaining_by_class = Counter()

total_before = 0
total_after = 0
total_removed = 0

print("=" * 60)
print("CYBERSOC AWS - DEDUPLICATION")
print("=" * 60)

for i, file in enumerate(files, 1):

    df = pd.read_parquet(file)
    total_before += len(df)

    # Empreinte de chaque ligne, Label inclus
    hashes = pd.util.hash_pandas_object(
        df, index=False
    ).to_numpy(dtype=np.uint64)

    keep_mask = np.empty(len(df), dtype=bool)

    # Rechercher les doublons entre tous les fichiers
    for j, h in enumerate(hashes):
        key = int(h)

        if key in seen_hashes:
            keep_mask[j] = False
        else:
            seen_hashes.add(key)
            keep_mask[j] = True

    duplicates = df.loc[~keep_mask]
    unique = df.loc[keep_mask]

    removed_by_class.update(
        duplicates["Label"].value_counts().to_dict()
    )
    remaining_by_class.update(
        unique["Label"].value_counts().to_dict()
    )

    total_removed += len(duplicates)
    total_after += len(unique)

    relative_path = file.relative_to(INPUT_DIR)
    output_file = OUTPUT_DIR / relative_path
    output_file.parent.mkdir(
        parents=True, exist_ok=True
    )

    unique.to_parquet(
        output_file,
        index=False
    )

    print(
        f"[{i}/{len(files)}] "
        f"{file.name}: "
        f"{len(duplicates):,} doublons supprimés"
    )

report = {
    "total_before": total_before,
    "total_after": total_after,
    "total_removed": total_removed,
    "duplicate_rate_percent": round(
        100 * total_removed / total_before, 4
    ),
    "removed_by_class": dict(removed_by_class),
    "remaining_by_class": dict(remaining_by_class)
}

report_path = REPORT_DIR / "deduplication_report.json"

with report_path.open("w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n" + "=" * 60)
print("DEDUPLICATION TERMINEE")
print("=" * 60)

print(f"Lignes initiales  : {total_before:,}")
print(f"Lignes conservees : {total_after:,}")
print(f"Doublons retires  : {total_removed:,}")

print("\nDistribution finale des classes :")
for label, count in remaining_by_class.most_common():
    print(f"{label:30s} {count:10,d}")

print("\nRapport :", report_path)
