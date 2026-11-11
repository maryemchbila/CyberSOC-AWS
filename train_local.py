
from pathlib import Path
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

ROOT = Path("data/processed")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models/local")

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(REPORT_DIR / "feature_columns.json",
          encoding="utf-8") as f:
    FEATURES = json.load(f)

# Entraînement rapide sur des échantillons
# Le jeu de test reste totalement intouché.
def load_sample(split, fraction):
    files = sorted((ROOT / split).glob("*.parquet"))
    if not files:
        raise ValueError(f"Aucun fichier : {split}")

    parts = []

    for i, file in enumerate(files):
        df = pd.read_parquet(
            file, columns=FEATURES + ["Target"]
        )

        sample = df.sample(
            frac=fraction,
            random_state=42 + i
        )

        if not sample.empty:
            parts.append(sample)

    return pd.concat(parts, ignore_index=True)


print("Chargement des données...")

train = load_sample("train", 0.10)
val = load_sample("validation", 0.15)

X_train = train[FEATURES].astype("float32")
y_train = train["Target"].astype("int32")

X_val = val[FEATURES].astype("float32")
y_val = val["Target"].astype("int32")

print(f"Train : {len(X_train):,} flux")
print(f"Validation : {len(X_val):,} flux")
print(f"Features : {len(FEATURES)}")

print("\nEntraînement XGBoost...")

model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    early_stopping_rounds=15,
    random_state=42,
    n_jobs=4
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)[:, 1]

metrics = {
    "train_rows": len(X_train),
    "validation_rows": len(X_val),
    "accuracy": float(
        accuracy_score(y_val, predictions)
    ),
    "precision": float(
        precision_score(y_val, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_val, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_val, predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_val, probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_val, predictions
    ).tolist(),
    "best_iteration": int(model.best_iteration),
}

print("\n========== RESULTATS ==========")

for name in ("accuracy", "precision", "recall",
             "f1_score", "roc_auc"):
    print(f"{name}: {metrics[name]:.4f}")

print("\nMatrice de confusion :")
print(metrics["confusion_matrix"])

print("\nRapport de classification :")
print(classification_report(
    y_val,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

model.save_model(
    str(MODEL_DIR / "xgboost_baseline.json")
)

with open(REPORT_DIR / "baseline_metrics.json",
          "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nModèle sauvegardé.")
print("Rapport : reports/baseline_metrics.json")
print("Test final non utilisé.")
