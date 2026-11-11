
from pathlib import Path
import argparse
import json
import math

import numpy as np
import xgboost as xgb

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models" / "final_candidate"


class CyberSOCPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)

        with open(model_dir / "feature_columns.json",
                  encoding="utf-8") as f:
            self.features = json.load(f)

        with open(model_dir / "label_mapping.json",
                  encoding="utf-8") as f:
            mapping = json.load(f)

        with open(model_dir / "inference_config.json",
                  encoding="utf-8") as f:
            self.config = json.load(f)

        self.class_names = {
            int(k): v
            for k, v in mapping["class_names"].items()
        }

        self.model = xgb.Booster()
        self.model.load_model(str(model_dir / "model.json"))

        if self.model.num_features() != len(self.features):
            raise ValueError("Nombre de features incompatible")

        if len(self.class_names) != 7:
            raise ValueError("Mapping multiclasse incompatible")

    def predict(self, flow):
        # Accepte un dictionnaire de 70 features
        # ou une liste de 70 valeurs dans l'ordre prévu.
        if isinstance(flow, dict):
            missing = [
                name for name in self.features
                if name not in flow
            ]
            extra = set(flow) - set(self.features)

            if missing or extra:
                raise ValueError(
                    f"Features manquantes : {missing}; "
                    f"features inconnues : {sorted(extra)}"
                )

            values = [flow[name] for name in self.features]

        elif isinstance(flow, list):
            values = flow
        else:
            raise TypeError(
                "Le flux doit être une liste ou un dictionnaire"
            )

        if len(values) != len(self.features):
            raise ValueError(
                f"Attendu : {len(self.features)} features, "
                f"reçu : {len(values)}"
            )

        if any(
            isinstance(value, (str, bytes, bool))
            or value is None
            for value in values
        ):
            raise ValueError("Toutes les features doivent être numériques")

        x = np.asarray(values, dtype=np.float32)

        if not np.isfinite(x).all():
            raise ValueError("NaN ou infini détecté")

        matrix = x.reshape(1, -1)
        probs = self.model.predict(
            xgb.DMatrix(matrix)
        )[0]

        if len(probs) != 7:
            raise RuntimeError("Sortie modèle inattendue")

        # Décision classique : classe la plus probable
        raw_class = int(np.argmax(probs))

        # Règle de priorité Bot expérimentale
        bot_id = int(self.config["bot_class_id"])
        threshold = float(self.config["bot_threshold"])

        predicted_id = raw_class

        if (
            self.config.get("bot_threshold_enabled", False)
            and probs[bot_id] >= threshold
        ):
            predicted_id = bot_id

        predicted_name = self.class_names[predicted_id]

        probabilities = {
            self.class_names[i]: round(float(probs[i]), 6)
            for i in sorted(self.class_names)
        }

        return {
            "predicted_class": predicted_name,
            "class_id": predicted_id,
            "confidence": round(
                float(probs[predicted_id]), 6
            ),
            "raw_class": self.class_names[raw_class],
            "threshold_override": predicted_id != raw_class,
            "is_attack": predicted_id != 0,
            "probabilities": probabilities,
            "model_version": self.config["model_version"]
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Fichier JSON avec 70 features"
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        flow = json.load(f)

    predictor = CyberSOCPredictor()
    result = predictor.predict(flow)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
