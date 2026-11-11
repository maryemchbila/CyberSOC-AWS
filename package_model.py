
from pathlib import Path
import json
import shutil

source = Path("models/sagemaker_multiclass_full/model.json")
dest = Path("models/final_candidate")
dest.mkdir(parents=True, exist_ok=True)

if not source.exists():
    raise FileNotFoundError(source)

shutil.copy2(source, dest / "model.json")

shutil.copy2(
    "reports/feature_columns.json",
    dest / "feature_columns.json"
)

shutil.copy2(
    "reports/multiclass_label_mapping.json",
    dest / "label_mapping.json"
)

config = {
    "model_type": "XGBoost",
    "model_version": "multiclass-full-v1",
    "num_classes": 7,
    "default_decision": "argmax",
    "bot_class_id": 5,
    "bot_threshold": 0.35,
    "bot_threshold_enabled": True,
    "threshold_tuned_on": "validation",
    "excluded_labels": [
        "Infiltration",
        "Heartbleed"
    ]
}

(dest / "inference_config.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8"
)

print("Package du modèle créé dans :", dest)
print("Seuil Bot :", config["bot_threshold"])
