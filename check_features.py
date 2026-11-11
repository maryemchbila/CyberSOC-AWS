
from pathlib import Path
from collections import Counter
import json
import pandas as pd
import numpy as np

INPUT_DIR = Path("data/deduplicated")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

files = sorted(INPUT_DIR.rglob("*.parquet"))

if not files:
    raise SystemExit("Aucun fichier Parquet trouvé.")

# Hash des features -> premier label rencontré
seen_features = {}

conflict_rows = 0
conflict_groups = set()
same_features_count = 0

feature_min = {}
feature_max = {}
label_distribution = Counter()
reference_columns = None

print("=" * 60)
print("CYBERSOC AWS - VERIFICATION DES FEATURES")
print("=" * 60)

for i, file in enumerate(files, 1):

    df = pd.read_parquet(file)

    if "Label" not in df.columns:
        raise ValueError(f"Label absent : {file}")

    feature_cols = [
        c for c in df.columns if c != "Label"
    ]

    if reference_columns is None:
        reference_columns = feature_cols
    elif feature_cols != reference_columns:
        raise ValueError(f"Schema different : {file}")

    label_distribution.update(
        df["Label"].value_counts().to_dict()
    )

    # Détection des colonnes constantes
    for col in feature_cols:
        values = df[col].to_numpy()
        if len(values) == 0:
            continue

        low = float(np.min(values))
        high = float(np.max(values))

        feature_min[col] = min(
            feature_min.get(col, low), low
        )
        feature_max[col] = max(
            feature_max.get(col, high), high
        )

    # Hash calculé uniquement sur les features
    hashes = pd.util.hash_pandas_object(
        df[feature_cols], index=False
    ).to_numpy(dtype=np.uint64)

    for h, label in zip(
        hashes, df["Label"].to_numpy()
    ):
        key = int(h)
        label = str(label)

        if key not in seen_features:
            seen_features[key] = label
        else:
            same_features_count += 1

            if seen_features[key] != label:
                conflict_rows += 1
                conflict_groups.add(key)

    if i % 10 == 0 or i == len(files):
        print(f"Fichiers vérifiés : {i}/{len(files)}")

constant_columns = [
    col for col in reference_columns
    if feature_min[col] == feature_max[col]
]

report = {
    "total_rows": sum(label_distribution.values()),
    "feature_count": len(reference_columns),
    "constant_columns": constant_columns,
    "identical_features_extra_rows": same_features_count,
    "contradictory_label_rows": conflict_rows,
    "contradictory_feature_groups": len(conflict_groups),
    "label_distribution": dict(label_distribution)
}

output = REPORT_DIR / "feature_audit_report.json"

with output.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n========== RESULTATS ==========")
print("Lignes :", report["total_rows"])
print("Features :", report["feature_count"])
print("Colonnes constantes :", constant_columns)
print("Occurrences supplémentaires avec mêmes features :",
      same_features_count)
print("Lignes avec labels contradictoires :",
      conflict_rows)
print("Groupes de features contradictoires :",
      len(conflict_groups))
print("Rapport :", output)
