
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "dist"
DEST.mkdir(exist_ok=True)

required = [
    Path("predict.py"),
    Path("models/final_candidate/model.json"),
    Path("models/final_candidate/feature_columns.json"),
    Path("models/final_candidate/label_mapping.json"),
    Path("models/final_candidate/inference_config.json"),
]

for name in [
    "BENIGN", "DoS", "DDoS", "PortScan",
    "BruteForce", "Bot", "WebAttack"
]:
    required.append(
        Path(f"reports/prediction_tests/flow_{name}.json")
    )

for file in required:
    if not (ROOT / file).is_file():
        raise FileNotFoundError(file)

requirements = """xgboost==2.1.4
numpy>=1.26,<3
"""

readme = """# CyberSOC ML — XGBoost Multiclasse

Modèle entraîné sur CIC-IDS2017 avec Amazon SageMaker.

## Classes
BENIGN, DoS, DDoS, PortScan, BruteForce, Bot, WebAttack.

## Installation
python -m pip install -r requirements.txt

## Exemple
python predict.py --input examples/flow_DDoS.json

## Entrées
70 caractéristiques numériques CIC-IDS2017.
L'ordre exact se trouve dans feature_columns.json.

## Sortie
JSON avec classe, score, probabilités et is_attack.

## Limites
Le modèle ne prend pas directement des paquets ou logs réseau bruts.
Il nécessite une extraction des 70 features compatible avec
celle utilisée pendant l'entraînement.

Infiltration et Heartbleed ne sont pas des classes reconnues.
Le seuil Bot 0.35 a été choisi expérimentalement sur Validation.
Le modèle n'est pas validé pour un SOC de production.
"""

archive_path = DEST / "CyberSOC-ML.zip"

with ZipFile(archive_path, "w", ZIP_DEFLATED) as z:
    for file in required:
        if file.parts[0] == "reports":
            target = Path("CyberSOC-ML/examples") / file.name
        else:
            target = Path("CyberSOC-ML") / file

        z.write(ROOT / file, arcname=str(target))

    z.writestr(
        "CyberSOC-ML/requirements.txt", requirements
    )
    z.writestr("CyberSOC-ML/README.md", readme)

print("Package créé :", archive_path)

with ZipFile(archive_path) as z:
    print("Fichiers :", len(z.namelist()))
    print("Archive valide :", z.testzip() is None)
