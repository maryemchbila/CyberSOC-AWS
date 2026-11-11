
from pathlib import Path
from datetime import datetime, timezone
import json

import boto3

try:
    from sagemaker.core import image_uris
except ImportError:
    from sagemaker import image_uris

REGION = "us-east-1"
BUCKET = "cybersoc-ml-023010856349-use1"
INSTANCE = "ml.m5.large"
PREFIX = "experiments/xgboost-multiclass"

session = boto3.Session(region_name=REGION)
s3 = session.client("s3")
sm = session.client("sagemaker")
sts = session.client("sts")

account_id = sts.get_caller_identity()["Account"]
role_arn = f"arn:aws:iam::{account_id}:role/LabRole"

# 1. Transfert des fichiers multiclasse
for split in ("train", "validation"):
    local_path = Path(
        f"data/sagemaker_multiclass/{split}.csv"
    )
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    key = f"{PREFIX}/{split}/data.csv"
    s3.upload_file(str(local_path), BUCKET, key)
    print(f"Upload OK : s3://{BUCKET}/{key}")

# 2. Image SageMaker XGBoost officielle
image_uri = image_uris.retrieve(
    "xgboost",
    REGION,
    version="1.7-1"
)

def channel(name):
    return {
        "ChannelName": name,
        "ContentType": "text/csv",
        "InputMode": "File",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/{PREFIX}/{name}/",
                "S3DataDistributionType": "FullyReplicated"
            }
        }
    }

job_name = (
    "cybersoc-multiclass-"
    + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
)

print("Job :", job_name)
print("Instance :", INSTANCE)
print("Classes : 7")

# 3. Création du Training Job : opération facturable
response = sm.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        "TrainingImage": image_uri,
        "TrainingInputMode": "File"
    },
    RoleArn=role_arn,
    InputDataConfig=[
        channel("train"),
        channel("validation")
    ],
    OutputDataConfig={
        "S3OutputPath": f"s3://{BUCKET}/models/multiclass/"
    },
    ResourceConfig={
        "InstanceType": INSTANCE,
        "InstanceCount": 1,
        "VolumeSizeInGB": 10
    },
    HyperParameters={
        "objective": "multi:softprob",
        "num_class": "7",
        "eval_metric": "mlogloss",
        "num_round": "80",
        "max_depth": "5",
        "eta": "0.1",
        "subsample": "0.8",
        "colsample_bytree": "0.8",
        "tree_method": "hist"
    },
    StoppingCondition={
        "MaxRuntimeInSeconds": 1800
    },
    Tags=[
        {"Key": "Project", "Value": "CyberSOC-AWS"}
    ]
)

Path("reports").mkdir(exist_ok=True)

Path("reports/sagemaker_multiclass_job.json").write_text(
    json.dumps({
        "job_name": job_name,
        "region": REGION,
        "instance": INSTANCE,
        "classes": 7
    }, indent=2),
    encoding="utf-8"
)

print("\nTraining Job lance.")
print("ARN :", response["TrainingJobArn"])
