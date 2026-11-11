
from pathlib import Path
from collections import Counter
import json
import pandas as pd

ROOT = Path("data/processed")
OUT = Path("data/sagemaker_multiclass")
REPORTS = Path("reports")
OUT.mkdir(parents=True, exist_ok=True)

with open(REPORTS / "feature_columns.json",
          encoding="utf-8") as f:
    FEATURES = json.load(f)

CLASS_NAMES = {
    0: "BENIGN",
    1: "DoS",
    2: "DDoS",
    3: "PortScan",
    4: "BruteForce",
    5: "Bot",
    6: "WebAttack"
}

LABEL_MAP = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "DoS GoldenEye": 1,
    "DoS slowloris": 1,
    "DoS Slowhttptest": 1,
    "DDoS": 2,
    "PortScan": 3,
    "FTP-Patator": 4,
    "SSH-Patator": 4,
    "Bot": 5,
    "Web Attack - Brute Force": 6,
    "Web Attack - XSS": 6,
    "Web Attack - Sql Injection": 6
}

# Échantillon du premier essai AWS
FRACTIONS = {
    "train": 0.06,
    "validation": 0.08
}

summary = {}

for split, fraction in FRACTIONS.items():
    files = sorted((ROOT / split).glob("*.parquet"))

    if not files:
        raise RuntimeError(f"Aucun fichier dans {split}")

    output_file = OUT / f"{split}.csv"

    if output_file.exists():
        raise RuntimeError(
            f"{output_file} existe déjà. "
            "Déplace-le avant de relancer."
        )

    counts = Counter()
    excluded = Counter()
    written = 0

    print(f"\nPréparation : {split}")

    for i, file in enumerate(files):
        df = pd.read_parquet(
            file,
            columns=FEATURES + ["OriginalLabel"]
        )

        excluded.update(
            df.loc[
                ~df["OriginalLabel"].isin(LABEL_MAP),
                "OriginalLabel"
            ].value_counts().to_dict()
        )

        df["ClassID"] = df["OriginalLabel"].map(LABEL_MAP)
        df = df.dropna(subset=["ClassID"])

        # Échantillonnage par classe, par fichier
        sampled = []

        for class_id, group in df.groupby("ClassID"):
            sample = group.sample(
                frac=fraction,
                random_state=42 + i
            )
            if not sample.empty:
                sampled.append(sample)

        if not sampled:
            continue

        result = pd.concat(sampled, ignore_index=True)
        result["ClassID"] = result["ClassID"].astype("int32")

        counts.update(
            result["ClassID"].value_counts().to_dict()
        )

        export = result[["ClassID"] + FEATURES]

        export.to_csv(
            output_file,
            mode="a",
            header=False,
            index=False,
            float_format="%.12g"
        )

        written += len(export)

    summary[split] = {
        "rows": written,
        "classes": {
            CLASS_NAMES[c]: counts[c] for c in CLASS_NAMES
        },
        "excluded_labels": dict(excluded)
    }

    print("Total :", written)
    for c in CLASS_NAMES:
        print(f"  {CLASS_NAMES[c]}: {counts[c]}")
        if counts[c] == 0:
            raise RuntimeError(
                f"Classe absente : {CLASS_NAMES[c]}"
            )

    print("Labels exclus :", dict(excluded))
    print("Fichier :", output_file)

with open(REPORTS / "multiclass_label_mapping.json",
          "w", encoding="utf-8") as f:
    json.dump({
        "class_names": CLASS_NAMES,
        "original_label_mapping": LABEL_MAP,
        "excluded_labels": ["Infiltration", "Heartbleed"]
    }, f, indent=2, ensure_ascii=False)

with open(REPORTS / "multiclass_sample_report.json",
          "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nPRÉPARATION MULTICLASSE TERMINÉE")
