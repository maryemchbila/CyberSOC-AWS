
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd

INPUT_DIR = Path("data/deduplicated")
OUTPUT_DIR = Path("data/ml_ready")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.rglob("*.parquet"))

if not files:
    raise SystemExit("Aucun fichier Parquet trouvé.")

if OUTPUT_DIR.exists() and any(
    OUTPUT_DIR.rglob("*.parquet")
):
    raise SystemExit(
        "data/ml_ready contient déjà des fichiers. "
        "Déplace ou supprime l'ancien résultat avant de continuer."
    )

CONSTANT_COLUMNS = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate"
]

# =============================================
# PASSE 1 : DETECTER LES LABELS CONTRADICTOIRES
# =============================================

print("PHASE 1 : Detection des contradictions")

first_label = {}
conflicting_hashes = set()
feature_columns = None

for i, file in enumerate(files, 1):

    df = pd.read_parquet(file)

    cols = [c for c in df.columns if c != "Label"]

    if feature_columns is None:
        feature_columns = cols
    elif cols != feature_columns:
        raise ValueError(f"Schema different : {file}")

    hashes = pd.util.hash_pandas_object(
        df[feature_columns],
        index=False
    ).to_numpy(dtype=np.uint64)

    for h, label in zip(hashes, df["Label"]):
        key = int(h)
        label = str(label)

        if key not in first_label:
            first_label[key] = label
        elif first_label[key] != label:
            conflicting_hashes.add(key)

    if i % 10 == 0 or i == len(files):
        print(f"  {i}/{len(files)} fichiers analyses")

print(
    "Groupes contradictoires :",
    len(conflicting_hashes)
)

missing_constants = [
    c for c in CONSTANT_COLUMNS
    if c not in feature_columns
]
if missing_constants:
    raise ValueError(
        f"Colonnes constantes absentes : {missing_constants}"
    )

final_features = [
    c for c in feature_columns
    if c not in CONSTANT_COLUMNS
]

# =============================================
# PASSE 2 : EXCLURE LES GROUPES CONTRADICTOIRES
# =============================================

print("\nPHASE 2 : Preparation du dataset ML")

before = 0
after = 0
removed_by_class = Counter()
remaining_by_class = Counter()

for i, file in enumerate(files, 1):

    df = pd.read_parquet(file)
    before += len(df)

    hashes = pd.util.hash_pandas_object(
        df[feature_columns],
        index=False
    ).to_numpy(dtype=np.uint64)

    keep = ~np.isin(
        hashes,
        np.array(
            list(conflicting_hashes),
            dtype=np.uint64
        )
    )

    excluded = df.loc[~keep]
    result = df.loc[keep, final_features + ["Label"]]

    removed_by_class.update(
        excluded["Label"].value_counts().to_dict()
    )
    remaining_by_class.update(
        result["Label"].value_counts().to_dict()
    )

    after += len(result)

    output_file = (
        OUTPUT_DIR / file.relative_to(INPUT_DIR)
    )

    output_file.parent.mkdir(
        parents=True, exist_ok=True
    )

    result.to_parquet(
        output_file,
        index=False
    )

    if i % 10 == 0 or i == len(files):
        print(f"  {i}/{len(files)} fichiers traites")

# =============================================
# RAPPORT ET METADONNEES
# =============================================

report = {
    "initial_rows": before,
    "final_rows": after,
    "excluded_rows": before - after,
    "conflicting_groups": len(conflicting_hashes),
    "initial_feature_count": len(feature_columns),
    "final_feature_count": len(final_features),
    "removed_constant_columns": CONSTANT_COLUMNS,
    "excluded_by_class": dict(removed_by_class),
    "remaining_by_class": dict(remaining_by_class)
}

with open(
    REPORT_DIR / "ml_preparation_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

with open(
    REPORT_DIR / "feature_columns.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(final_features, f, indent=2)

print("\n" + "=" * 55)
print("PREPARATION ML TERMINEE")
print("=" * 55)
print(f"Lignes initiales : {before:,}")
print(f"Lignes conservees : {after:,}")
print(f"Lignes exclues : {before - after:,}")
print(f"Features finales : {len(final_features)}")

print("\nDistribution finale :")
for label, count in remaining_by_class.most_common():
    print(f"  {label}: {count:,}")

print("\nFichiers generes dans : data/ml_ready/")
