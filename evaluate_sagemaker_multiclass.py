
from pathlib import Path
import json
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

MODEL_ARCHIVE = Path(
    "models/sagemaker_multiclass/model.tar.gz"
)
TEST_DIR = Path("data/processed/test")
REPORTS = Path("reports")

with open(REPORTS / "feature_columns.json",
          encoding="utf-8") as f:
    features = json.load(f)

with open(REPORTS / "multiclass_label_mapping.json",
          encoding="utf-8") as f:
    mapping = json.load(f)

label_map = mapping["original_label_mapping"]
class_names = [
    mapping["class_names"][str(i)]
    for i in range(7)
]

# Charger le modèle depuis l'archive SageMaker
with tarfile.open(MODEL_ARCHIVE, "r:gz") as archive:
    members = [
        m for m in archive.getmembers()
        if m.isfile()
        and Path(m.name).name == "xgboost-model"
    ]
    if len(members) != 1:
        raise RuntimeError(
            f"Modèle introuvable : {archive.getnames()}"
        )
    model_data = archive.extractfile(members[0]).read()

# Charger le modèle directement depuis les octets
# sans créer de fichier temporaire Windows
booster = xgb.Booster()
booster.load_model(bytearray(model_data))
print("Modèle chargé avec XGBoost", xgb.__version__)
print("Nombre de features :", booster.num_features())

# Export moderne pour une future utilisation
booster.save_model("models/sagemaker_multiclass/model.json")
print("Modèle JSON exporté.")

print("Modèle SageMaker chargé.")

y_true = []
y_pred = []
excluded = {}

files = sorted(TEST_DIR.glob("*.parquet"))

if not files:
    raise RuntimeError("Aucun fichier de test trouvé")

for i, file in enumerate(files, start=1):
    df = pd.read_parquet(
        file,
        columns=features + ["OriginalLabel"]
    )

    labels = df["OriginalLabel"]
    known = labels.isin(label_map)

    for name, count in labels[~known].value_counts().items():
        excluded[name] = excluded.get(name, 0) + int(count)

    unknown = set(labels[~known].dropna().unique()) - {
        "Infiltration", "Heartbleed"
    }
    if unknown:
        raise RuntimeError(
            f"Labels inattendus : {unknown}"
        )

    df = df.loc[known]

    if df.empty:
        continue

    targets = df["OriginalLabel"].map(label_map).astype(int)

    # Ordre des 70 features identique à l'entraînement
    X = df[features].to_numpy(dtype=np.float32)

    predictions = booster.predict(xgb.DMatrix(X))

    if predictions.ndim != 2 or predictions.shape[1] != 7:
        raise RuntimeError(
            f"Forme inattendue : {predictions.shape}"
        )

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(targets.tolist())
    y_pred.extend(predicted_classes.tolist())

    print(f"Fichier {i}/{len(files)} évalué")

print("\n===== RÉSULTATS TEST =====")
print("Flux évalués :", len(y_true))
print("Flux exclus :", excluded)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

report_text = classification_report(
    y_true, y_pred,
    labels=list(range(7)),
    target_names=class_names,
    zero_division=0
)
report_dict = classification_report(
    y_true, y_pred,
    labels=list(range(7)),
    target_names=class_names,
    zero_division=0,
    output_dict=True
)
cm = confusion_matrix(
    y_true, y_pred, labels=list(range(7))
)

print("Accuracy :", round(accuracy, 4))
print("Macro F1 :", round(macro_f1, 4))
print("Weighted F1 :", round(weighted_f1, 4))
print("\n", report_text)
print("Matrice de confusion :\n", cm)

REPORTS.mkdir(exist_ok=True)

output = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "class_names": class_names,
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "excluded": excluded,
    "evaluated_rows": len(y_true)
}

(REPORTS / "sagemaker_multiclass_test_metrics.json").write_text(
    json.dumps(output, indent=2),
    encoding="utf-8"
)

print("\nRapport enregistré dans reports/")
