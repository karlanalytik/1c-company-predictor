"""
Bronze layer ingestion script.

Downloads raw data from Kaggle and uploads it to S3 Bronze layer.

Flow:
1. Load Kaggle credentials from AWS Secrets Manager
2. Download competition data using kagglehub
3. Upload files to: s3://<bucket>/sales_predict/bronze/
"""

import argparse
import logging
from pathlib import Path
import os
import json

import boto3
import kagglehub


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Args
# ============================================================================


def parse_args():
    """
    Parse CLI arguments.

    Returns:
        Namespace with:
            - bucket (str): target S3 bucket
            - secret_name (str): Secrets Manager key
            - region_name (str): AWS region
    """
    parser = argparse.ArgumentParser(description="Bronze layer ingestion")

    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--secret-name", default="kaggle/credentials")
    parser.add_argument("--region-name", default="us-east-1")

    return parser.parse_args()


# ============================================================================
# Functions
# ============================================================================


def load_kaggle_credentials(secret_name: str, region_name: str) -> None:
    """
    Load Kaggle credentials from AWS Secrets Manager and
    set them as environment variables.

    Args:
        secret_name (str): name of the secret
        region_name (str): AWS region
    """
    client = boto3.client("secretsmanager", region_name=region_name)

    response = client.get_secret_value(SecretId=secret_name)
    secret = json.loads(response["SecretString"])

    os.environ["KAGGLE_USERNAME"] = secret["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = secret["KAGGLE_KEY"]


def upload_to_s3(local_path: str, bucket: str) -> None:
    """
    Upload all files from a local directory to S3 Bronze layer.

    Files are stored at:
        s3://<bucket>/sales_predict/bronze/<file_name>

    Args:
        local_path (str): directory containing downloaded files
        bucket (str): target S3 bucket
    """
    logger.info("Loading raw data")

    s3 = boto3.client("s3")
    local_path = Path(local_path)

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            s3_key = f"sales_predict/bronze/{file_path.name}"

            logger.info(f"Uploading {file_path.name} to s3://{bucket}/{s3_key}")
            s3.upload_file(str(file_path), bucket, s3_key)
            logger.info("Loaded successfully")


# ============================================================================
# Main
# ============================================================================


def main():
    """
    Orchestrates the ingestion process:
    - Loads credentials
    - Downloads data from Kaggle
    - Uploads raw files to S3
    """
    args = parse_args()

    logger.info("Starting raw data extraction")

    logger.info("Loading Kaggle credentials")
    try:
        load_kaggle_credentials(
            secret_name=args.secret_name, region_name=args.region_name
        )
    except Exception as e:
        logger.error(f"Error loading Kaggle credentials: {e}")
        raise

    logger.info("Downloading Kaggle data")
    try:
        path = kagglehub.competition_download(
            "competitive-data-science-predict-future-sales"
        )
    except Exception as e:
        logger.error(f"Error downloading Kaggle credentials: {e}")
        raise

    logger.info(f"Kaggle files downloaded to: {path}")

    # TODO: Add asserts with data validation
    logger.info("Loading raw data to s3")
    try:
        upload_to_s3(local_path=path, bucket=args.bucket)
    except Exception as e:
        logger.error(f"Error loading raw data to s3: {e}")
        raise

    logger.info("Bronze layer completed successfully")


if __name__ == "__main__":
    main()
