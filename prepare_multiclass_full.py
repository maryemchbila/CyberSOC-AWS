
from pathlib import Path
from collections import Counter
import json
import pandas as pd
import numpy as np

SOURCE = Path("data/processed")
DEST = Path("data/sagemaker_multiclass_full")
REPORTS = Path("reports")

with open(REPORTS / "feature_columns.json",
          encoding="utf-8") as f:
    features = json.load(f)

with open(REPORTS / "multiclass_label_mapping.json",
          encoding="utf-8") as f:
    mapping = json.load(f)

label_map = mapping["original_label_mapping"]
class_names = mapping["class_names"]

assert len(features) == 70
assert set(label_map.values()) == set(range(7))

summary = {}

for split in ["train", "validation"]:
    input_dir = SOURCE / split
    output_dir = DEST / split
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"Aucun Parquet : {input_dir}")

    # Éviter les doublons d'une ancienne exécution
    if list(output_dir.glob("*.csv")):
        raise RuntimeError(
            f"Des CSV existent déjà dans {output_dir}. "
            "Nettoie ce dossier avant de relancer."
        )

    counts = Counter()
    excluded = Counter()
    total = 0

    print(f"\n===== {split.upper()} =====")

    for index, file in enumerate(files):
        df = pd.read_parquet(
            file,
            columns=features + ["OriginalLabel"]
        )

        unknown = df.loc[
            ~df["OriginalLabel"].isin(label_map),
            "OriginalLabel"
        ]
        excluded.update(unknown.value_counts().to_dict())

        unexpected = set(unknown.dropna().unique()) - {
            "Infiltration", "Heartbleed"
        }
        if unexpected:
            raise RuntimeError(
                f"Labels inconnus : {unexpected}"
            )

        df["ClassID"] = df["OriginalLabel"].map(label_map)
        df = df.dropna(subset=["ClassID"])

        if df.empty:
            continue

        df["ClassID"] = df["ClassID"].astype("int32")

        export = df[["ClassID"] + features]

        if export.shape[1] != 71:
            raise RuntimeError("Nombre de colonnes incorrect")

        if not np.isfinite(
            export.to_numpy(dtype=np.float64)
        ).all():
            raise RuntimeError(
                f"Valeurs invalides dans {file}"
            )

        name = f"part_{index:04d}.csv"
        destination = output_dir / name
        temp_file = output_dir / (name + ".tmp")

        export.to_csv(
            temp_file,
            index=False,
            header=False,
            float_format="%.12g"
        )
        temp_file.replace(destination)

        counts.update(
            df["ClassID"].value_counts().to_dict()
        )
        total += len(export)

        print(
            f"[{index + 1}/{len(files)}] "
            f"{destination.name} : {len(export)} flux"
        )

    missing = [
        class_names[str(i)]
        for i in range(7) if counts[i] == 0
    ]
    if missing:
        raise RuntimeError(
            f"Classes absentes : {missing}"
        )

    summary[split] = {
        "rows": total,
        "csv_count": len(list(output_dir.glob("*.csv"))),
        "classes": {
            class_names[str(i)]: counts[i]
            for i in range(7)
        },
        "excluded": dict(excluded)
    }

    print(f"\nTotal {split} : {total}")
    print("Distribution :", summary[split]["classes"])
    print("Exclusions :", dict(excluded))

REPORTS.mkdir(exist_ok=True)

(REPORTS / "multiclass_full_report.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nDATASET MULTICLASSE COMPLET PRÊT")
