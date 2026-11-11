
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path("data/processed/test")
REPORT_DIR = Path("reports")
MODEL_PATH = Path("models/local/xgboost_baseline.json")

with open(REPORT_DIR / "feature_columns.json",
          encoding="utf-8") as f:
    features = json.load(f)

files = sorted(ROOT.glob("*.parquet"))

if not files:
    raise SystemExit("Aucun fichier de test trouvé.")

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

all_true = []
all_pred = []
all_proba = []
source_stats = {}

print("=" * 60)
print("CYBERSOC AWS - EVALUATION DU MODELE SUR TEST")
print("=" * 60)

for i, file in enumerate(files, 1):
    df = pd.read_parquet(file)

    X = df[features].astype("float32")
    y = df["Target"].astype("int32").to_numpy()

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    all_true.append(y)
    all_pred.append(predictions)
    all_proba.append(probabilities)

    source = file.name.split("__")[0]

    if source not in source_stats:
        source_stats[source] = Counter()

    source_stats[source]["total"] += len(y)
    source_stats[source]["attacks"] += int(y.sum())
    source_stats[source]["errors"] += int(
        (predictions != y).sum()
    )

    if i % 10 == 0 or i == len(files):
        print(f"Fichiers testes : {i}/{len(files)}")

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)
y_proba = np.concatenate(all_proba)

matrix = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()

metrics = {
    "test_rows": int(len(y_true)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_true, y_pred, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_true, y_proba)),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
    "false_positive_rate": float(fp / (fp + tn)),
    "source_stats": {
        k: dict(v) for k, v in source_stats.items()
    },
}

print("\n========== TEST FINAL ==========")

print(f"Flux testes : {len(y_true):,}")

for name in (
    "accuracy", "precision", "recall",
    "f1_score", "roc_auc"
):
    print(f"{name}: {metrics[name]:.4f}")

print("\nMatrice de confusion :")
print(matrix)

print("\nFalse Positive Rate : "
      f"{100 * metrics['false_positive_rate']:.4f}%")

print("\nClassification Report :")
print(classification_report(
    y_true,
    y_pred,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

print("\nErreurs par fichier source :")
for source, counts in sorted(source_stats.items()):
    print(
        f"{source}: "
        f"{counts['errors']} erreurs / "
        f"{counts['total']} flux"
    )

output = REPORT_DIR / "test_metrics.json"

with output.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nRapport :", output)
print("EVALUATION TERMINEE")
