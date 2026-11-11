
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score
)

REPORTS = Path("reports")
MODEL = Path("models/sagemaker_multiclass_full/model.json")

with open(REPORTS / "feature_columns.json", encoding="utf-8") as f:
    features = json.load(f)

with open(REPORTS / "multiclass_label_mapping.json", encoding="utf-8") as f:
    labels = json.load(f)["original_label_mapping"]

booster = xgb.Booster()
booster.load_model(str(MODEL))

true_parts = []
prob_parts = []

files = sorted(Path("data/processed/validation").glob("*.parquet"))

for i, file in enumerate(files, 1):
    df = pd.read_parquet(
        file, columns=features + ["OriginalLabel"]
    )
    df = df[df["OriginalLabel"].isin(labels)]

    if df.empty:
        continue

    y = df["OriginalLabel"].map(labels).astype(int).to_numpy()
    X = df[features].to_numpy(dtype=np.float32)

    p = booster.predict(xgb.DMatrix(X))
    if p.ndim != 2 or p.shape[1] != 7:
        raise RuntimeError("Probabilités multiclasse invalides")

    true_parts.append(y)
    prob_parts.append(p)
    print(f"Validation : {i}/{len(files)}")

y_true = np.concatenate(true_parts)
probs = np.concatenate(prob_parts)
base_pred = np.argmax(probs, axis=1)

results = []

for threshold in np.linspace(0.05, 0.50, 19):
    pred = base_pred.copy()
    pred[probs[:, 5] >= threshold] = 5

    true_bot = (y_true == 5)
    pred_bot = (pred == 5)

    precision = precision_score(
        true_bot, pred_bot, zero_division=0
    )
    recall = recall_score(
        true_bot, pred_bot, zero_division=0
    )
    macro_f1 = f1_score(
        y_true, pred, average="macro", zero_division=0
    )

    results.append({
        "threshold": round(float(threshold), 3),
        "bot_precision": round(float(precision), 5),
        "bot_recall": round(float(recall), 5),
        "macro_f1": round(float(macro_f1), 5)
    })

    print(
        f"Seuil={threshold:.3f} "
        f"Precision Bot={precision:.4f} "
        f"Recall Bot={recall:.4f} "
        f"Macro F1={macro_f1:.4f}"
    )

# Garde-fou : ne retenir que les seuils
# respectant une precision Bot >= 90 %
eligible = [
    r for r in results
    if r["bot_precision"] >= 0.90
]

best = max(
    eligible,
    key=lambda r: (r["bot_recall"], r["macro_f1"])
) if eligible else None

output = {
    "dataset": "validation",
    "baseline_full_bot_recall": 0.70607,
    "precision_guardrail": 0.90,
    "best_candidate": best,
    "results": results
}

(REPORTS / "bot_threshold_optimization.json").write_text(
    json.dumps(output, indent=2),
    encoding="utf-8"
)

print("\nMEILLEUR CANDIDAT :", best)
print("Rapport : reports/bot_threshold_optimization.json")
