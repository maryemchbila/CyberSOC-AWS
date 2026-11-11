
from pathlib import Path
import json
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

REPORTS = Path("reports")

MODELS = {
    "baseline_6pct":
        Path("models/sagemaker_multiclass/model.tar.gz"),
    "full_100pct":
        Path("models/sagemaker_multiclass_full/model.tar.gz")
}

with open(REPORTS / "feature_columns.json",
          encoding="utf-8") as f:
    features = json.load(f)

with open(REPORTS / "multiclass_label_mapping.json",
          encoding="utf-8") as f:
    mapping = json.load(f)

label_map = mapping["original_label_mapping"]
names = [
    mapping["class_names"][str(i)]
    for i in range(7)
]

def load_model(path):
    with tarfile.open(path, "r:gz") as archive:
        members = [
            member for member in archive.getmembers()
            if member.isfile()
            and Path(member.name).name == "xgboost-model"
        ]

        if len(members) != 1:
            raise RuntimeError(
                f"Modèle absent ou ambigu : {path}"
            )

        data = archive.extractfile(members[0]).read()

    booster = xgb.Booster()
    booster.load_model(bytearray(data))

    if booster.num_features() != len(features):
        raise RuntimeError(
            f"Features incompatibles : {path}"
        )

    return booster

models = {
    name: load_model(path)
    for name, path in MODELS.items()
}

# Sauvegarder le modèle complet en JSON
output_model = Path(
    "models/sagemaker_multiclass_full/model.json"
)
models["full_100pct"].save_model(str(output_model))

print("Modèles chargés avec XGBoost", xgb.__version__)
print("Modèle complet JSON :", output_model)

y_true = []
predictions = {
    name: [] for name in models
}

files = sorted(
    Path("data/processed/validation").glob("*.parquet")
)

if not files:
    raise RuntimeError("Aucun fichier Validation trouvé")

excluded = {}

for i, path in enumerate(files, 1):
    df = pd.read_parquet(
        path,
        columns=features + ["OriginalLabel"]
    )

    known = df["OriginalLabel"].isin(label_map)

    for label, count in df.loc[
        ~known, "OriginalLabel"
    ].value_counts().items():
        excluded[label] = excluded.get(label, 0) + int(count)

    df = df.loc[known]

    if df.empty:
        continue

    targets = (
        df["OriginalLabel"]
        .map(label_map)
        .astype(int)
        .to_numpy()
    )

    X = df[features].to_numpy(dtype=np.float32)
    matrix = xgb.DMatrix(X)

    y_true.extend(targets.tolist())

    for name, booster in models.items():
        probs = booster.predict(matrix)

        if probs.ndim != 2 or probs.shape[1] != 7:
            raise RuntimeError(
                f"Forme incorrecte : {name}, {probs.shape}"
            )

        predicted = np.argmax(probs, axis=1)
        predictions[name].extend(predicted.tolist())

    print(f"Validation : {i}/{len(files)}")

results = {}

for name, predicted in predictions.items():
    print("\n" + "=" * 55)
    print("MODÈLE :", name)
    print("=" * 55)

    accuracy = accuracy_score(y_true, predicted)
    macro_f1 = f1_score(
        y_true, predicted,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        predicted,
        labels=list(range(7)),
        target_names=names,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        predicted,
        labels=list(range(7))
    )

    print("Accuracy :", round(accuracy, 5))
    print("Macro F1 :", round(macro_f1, 5))

    print("\nPerformances par classe :")
    for class_name in names:
        row = report[class_name]
        print(
            f"{class_name:12s} "
            f"Precision={row['precision']:.4f} "
            f"Recall={row['recall']:.4f} "
            f"F1={row['f1-score']:.4f}"
        )

    print("\nMatrice de confusion :")
    print(matrix)

    results[name] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": matrix.tolist()
    }

print("\n" + "=" * 55)
print("COMPARAISON FINALE — VALIDATION")
print("=" * 55)

for metric in ["accuracy", "macro_f1"]:
    print(f"\n{metric}:")
    for name in models:
        print(f"  {name}: {results[name][metric]:.5f}")

for class_name in ["Bot", "WebAttack"]:
    print(f"\nRecall {class_name}:")
    for name in models:
        recall = results[name]["report"][class_name]["recall"]
        print(f"  {name}: {recall:.5f}")

print("\nFlux évalués :", len(y_true))
print("Labels exclus :", excluded)

output = {
    "dataset": "validation",
    "evaluated_rows": len(y_true),
    "excluded": excluded,
    "models": results
}

(REPORTS / "sagemaker_models_comparison.json").write_text(
    json.dumps(output, indent=2),
    encoding="utf-8"
)

print("\nRapport enregistré :")
print("reports/sagemaker_models_comparison.json")
