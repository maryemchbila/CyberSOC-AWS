
from datetime import datetime, timezone
from pathlib import Path
import json
import boto3

try:
    from sagemaker.core import image_uris
except ImportError:
    from sagemaker import image_uris

REGION = "us-east-1"
BUCKET = "cybersoc-ml-023010856349-use1"
PREFIX = "experiments/xgboost-multiclass-full"
INSTANCE = "ml.m5.xlarge"

session = boto3.Session(region_name=REGION)
sts = session.client("sts")
sm = session.client("sagemaker")
s3 = session.client("s3")

account = sts.get_caller_identity()["Account"]
role_arn = f"arn:aws:iam::{account}:role/LabRole"

# Contrôler les 59 CSV sur chaque canal.
for split in ["train", "validation"]:
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=f"{PREFIX}/{split}/"
    ):
        objects.extend(
            x for x in page.get("Contents", [])
            if x["Key"].endswith(".csv") and x["Size"] > 0
        )

    total_size = sum(x["Size"] for x in objects)
    print(
        f"{split}: {len(objects)} CSV, "
        f"{total_size / 1e6:.2f} MB"
    )

    if len(objects) != 59:
        raise RuntimeError(
            f"Nombre de CSV incorrect pour {split}"
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

image = image_uris.retrieve(
    "xgboost",
    REGION,
    version="1.7-1"
)

job_name = (
    "cybersoc-multiclass-full-"
    + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
)

print("Job :", job_name)
print("Instance :", INSTANCE)
print("Image :", image)

# ATTENTION : démarre une instance AWS facturable.
response = sm.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        "TrainingImage": image,
        "TrainingInputMode": "File"
    },
    RoleArn=role_arn,
    InputDataConfig=[
        channel("train"),
        channel("validation")
    ],
    OutputDataConfig={
        "S3OutputPath": (
            f"s3://{BUCKET}/models/multiclass-full/"
        )
    },
    ResourceConfig={
        "InstanceType": INSTANCE,
        "InstanceCount": 1,
        "VolumeSizeInGB": 15
    },
    HyperParameters={
        "objective": "multi:softprob",
        "num_class": "7",
        "eval_metric": "mlogloss",
        "num_round": "100",
        "max_depth": "5",
        "eta": "0.1",
        "subsample": "0.8",
        "colsample_bytree": "0.8",
        "tree_method": "hist"
    },
    StoppingCondition={
        "MaxRuntimeInSeconds": 2700
    },
    Tags=[
        {"Key": "Project", "Value": "CyberSOC-AWS"},
        {"Key": "Experiment", "Value": "MulticlassFull"}
    ]
)

Path("reports").mkdir(exist_ok=True)
Path("reports/sagemaker_full_job.json").write_text(
    json.dumps({
        "job_name": job_name,
        "region": REGION,
        "instance": INSTANCE
    }, indent=2),
    encoding="utf-8"
)

print("\nTRAINING JOB CRÉÉ")
print("ARN :", response["TrainingJobArn"])
