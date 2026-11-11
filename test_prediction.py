
from pathlib import Path
import json

import pandas as pd
from predict import CyberSOCPredictor

predictor = CyberSOCPredictor()

files = sorted(
    Path("data/processed/validation").glob("*.parquet")
)

if not files:
    raise RuntimeError("Aucune donnée Validation")

# Choisir un flux existant, avec un label connu.
flow = None
actual_label = None

mapping_path = Path(
    "models/final_candidate/label_mapping.json"
)

with open(mapping_path, encoding="utf-8") as f:
    label_mapping = json.load(f)["original_label_mapping"]

for file in files:
    df = pd.read_parquet(
        file,
        columns=predictor.features + ["OriginalLabel"]
    )

    valid = df[df["OriginalLabel"].isin(label_mapping)]

    if not valid.empty:
        row = valid.iloc[0]
        flow = {
            name: float(row[name])
            for name in predictor.features
        }
        actual_label = row["OriginalLabel"]
        break

if flow is None:
    raise RuntimeError("Aucun flux compatible trouvé")

Path("sample_flow.json").write_text(
    json.dumps(flow, indent=2),
    encoding="utf-8"
)

result = predictor.predict(flow)

print("Label réel :", actual_label)
print("Classe prédite :", result["predicted_class"])
print("Score :", result["confidence"])
print("Alerte :", result["is_attack"])
print("Seuil Bot appliqué :", result["threshold_override"])
print("\nProbabilités :")
print(json.dumps(result["probabilities"], indent=2))

print("\nFlux sauvegardé : sample_flow.json")
