
from pathlib import Path
from collections import defaultdict
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from predict import CyberSOCPredictor

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "validation"
OUTPUT = BASE / "reports" / "prediction_tests"
OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS = 20
SEED = 42

predictor = CyberSOCPredictor()

with open(
    BASE / "models" / "final_candidate" / "label_mapping.json",
    encoding="utf-8"
) as f:
    mapping = json.load(f)

original_to_id = mapping["original_label_mapping"]
class_names = {
    int(k): v
    for k, v in mapping["class_names"].items()
}

files = sorted(DATA.glob("*.parquet"))

if not files:
    raise RuntimeError("Aucun Parquet dans Validation")

# Collecter des candidats sans charger tout le
# dataset dans un seul DataFrame.
candidates = defaultdict(list)

for i, file in enumerate(files):
    df = pd.read_parquet(
        file,
        columns=predictor.features + ["OriginalLabel"]
    )

    df["ClassID"] = df["OriginalLabel"].map(original_to_id)
    df = df.dropna(subset=["ClassID"])

    for class_id, group in df.groupby("ClassID"):
        class_id = int(class_id)

        sample = group.sample(
            n=min(SAMPLES_PER_CLASS, len(group)),
            random_state=SEED + i
        )

        candidates[class_id].append(sample)

print("\n===== TEST DES 7 CATÉGORIES =====")

results = []
y_true = []
y_pred = []

for class_id in range(7):
    name = class_names[class_id]

    if class_id not in candidates:
        raise RuntimeError(f"Classe absente : {name}")

    pool = pd.concat(
        candidates[class_id],
        ignore_index=True
    )

    # Sélection déterministe de 20 exemples
    sample = pool.sample(
        n=min(SAMPLES_PER_CLASS, len(pool)),
        random_state=SEED + class_id
    )

    correct = 0

    print(f"\n--- {name} ---")

    for index, (_, row) in enumerate(sample.iterrows()):
        flow = {
            feature: float(row[feature])
            for feature in predictor.features
        }

        result = predictor.predict(flow)

        predicted = result["predicted_class"]
        expected = name
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        y_true.append(class_id)
        y_pred.append(result["class_id"])

        results.append({
            "expected": expected,
            "original_label": str(row["OriginalLabel"]),
            "predicted": predicted,
            "correct": is_correct,
            "confidence": result["confidence"],
            "threshold_override": result["threshold_override"],
            "probabilities": result["probabilities"]
        })

        # Garder un exemple JSON par catégorie
        if index == 0:
            (OUTPUT / f"flow_{name}.json").write_text(
                json.dumps(flow, indent=2),
                encoding="utf-8"
            )

            print("Label réel :", expected)
            print("Label original :", row["OriginalLabel"])
            print("Prédiction :", predicted)
            print("Score :", result["confidence"])
            print("Alerte :", result["is_attack"])
            print(
                "Seuil Bot appliqué :",
                result["threshold_override"]
            )

    print(
        f"Correctement classés : {correct}/{len(sample)}"
    )

accuracy = accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(7)),
    target_names=[class_names[i] for i in range(7)],
    zero_division=0
)

matrix = confusion_matrix(
    y_true, y_pred, labels=list(range(7))
)

print("\n===== RÉSULTATS GLOBAUX =====")
print("Nombre de flux :", len(y_true))
print("Accuracy :", round(accuracy, 4))
print("\n", report)
print("Matrice de confusion :")
print(matrix)

(OUTPUT / "results.json").write_text(
    json.dumps({
        "accuracy": accuracy,
        "sample_count": len(y_true),
        "samples_per_class": SAMPLES_PER_CLASS,
        "results": results,
        "confusion_matrix": matrix.tolist()
    }, indent=2),
    encoding="utf-8"
)

print("\nRésultats sauvegardés dans :", OUTPUT)
