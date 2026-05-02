"""
Silver layer ingestion script.

Reads raw CSV files from S3 Bronze layer, applies basic validations and
transformations, and uploads clean Parquet tables to S3 Silver layer.

Flow:
1. Read Bronze CSV files from S3
2. Validate schema, data types, duplicated records and null columns
3. Upload Silver tables to: s3://<bucket>/sales_predict/silver/
"""

import argparse
import logging

import awswrangler as wr
import pandas as pd


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
    """
    parser = argparse.ArgumentParser(description="Silver layer ingestion")

    parser.add_argument("--bucket", required=True, help="S3 bucket name")

    return parser.parse_args()


# ============================================================================
# Config
# ============================================================================

TABLE_CONFIG = {
    "sales": {
        "file_name": "sales_train.csv",
        "columns": [
            "date",
            "date_block_num",
            "shop_id",
            "item_id",
            "item_price",
            "item_cnt_day",
        ],
        "dtypes": {
            "date_block_num": "int64",
            "shop_id": "int64",
            "item_id": "int64",
            "item_price": "float64",
            "item_cnt_day": "float64",
        },
        "unique_cols": [],
    },
    "items": {
        "file_name": "items.csv",
        "columns": ["item_name", "item_id", "item_category_id"],
        "dtypes": {
            "item_name": "string",
            "item_id": "int64",
            "item_category_id": "int64",
        },
        "unique_cols": ["item_id"],
    },
    "item_categories": {
        "file_name": "item_categories.csv",
        "columns": ["item_category_name", "item_category_id"],
        "dtypes": {
            "item_category_name": "string",
            "item_category_id": "int64",
        },
        "unique_cols": ["item_category_id"],
    },
    "shops": {
        "file_name": "shops.csv",
        "columns": ["shop_name", "shop_id"],
        "dtypes": {
            "shop_name": "string",
            "shop_id": "int64",
        },
        "unique_cols": ["shop_id"],
    },
    "test": {
        "file_name": "test.csv",
        "columns": ["ID", "shop_id", "item_id"],
        "dtypes": {
            "ID": "int64",
            "shop_id": "int64",
            "item_id": "int64",
        },
        "unique_cols": ["ID"],
    },
}


# ============================================================================
# Functions
# ============================================================================


def read_bronze_table(bucket: str, file_name: str, table_name: str) -> pd.DataFrame:
    """
    Read one Bronze CSV table from S3.

    Args:
        bucket (str): source S3 bucket
        file_name (str): file name in Bronze layer
        table_name (str): table name

    Returns:
        Loaded DataFrame.
    """
    path = f"s3://{bucket}/sales_predict/bronze/{file_name}"

    logger.info(f"Reading {table_name} from {path}")

    try:
        return wr.s3.read_csv(path)
    except Exception as e:
        logger.error(f"Error reading {table_name}: {e}")
        raise


def clean_table(df: pd.DataFrame, table_name: str, config: dict) -> pd.DataFrame:
    """
    Apply Silver validations and transformations.

    Args:
        df (pd.DataFrame): raw DataFrame
        table_name (str): table name
        config (dict): table configuration

    Returns:
        Clean DataFrame.
    """
    logger.info(f"Cleaning {table_name}")

    logger.info("Checking that all columns are present")
    missing_cols = set(config["columns"]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{table_name} is missing columns: {missing_cols}")

    logger.info("Keeping only the necessary columns")
    df = df[config["columns"]].copy()

    logger.info("Casting to the expected data type")
    df = df.astype(config["dtypes"])
    if table_name == "sales":
        df['date'] = pd.to_datetime(df['date'], format = '%d-%m-%Y')

    logger.info("Removing duplicates")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    if before != after:
        logger.warning(f"{table_name}: removed {before - after} duplicated rows")

    logger.info("Checking for empty columns")
    null_cols = df.columns[df.isna().all()].tolist()
    if null_cols:
        logger.error(f"{table_name} has all-null columns: {null_cols}")
        raise ValueError(f"{table_name} null columns detected")

    logger.info("Validation of unique IDs")
    for col in config["unique_cols"]:
        if df[col].duplicated().any():
            logger.error(f"{table_name}.{col} has duplicates")
            raise ValueError(f"{table_name}.{col} not unique")

    if table_name == "test":
        df = df.rename(columns={"ID": "id"})
        df["date_block_num"] = 34

    logger.info(f"{table_name} cleaned successfully: {len(df)} rows")

    return df


def write_silver_table(df: pd.DataFrame, bucket: str, table_name: str) -> None:
    """
    Write clean table to S3 Silver layer as Parquet.

    Args:
        df (pd.DataFrame): clean DataFrame
        bucket (str): target S3 bucket
        table_name (str): table name
    """
    path = f"s3://{bucket}/sales_predict/silver/{table_name}/"

    logger.info(f"Writing {table_name} to {path}")

    wr.s3.to_parquet(
        df=df,
        path=path,
        dataset=True,
        mode="overwrite",
        index=False,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    """
    Orchestrates the Silver layer process:
    - Reads Bronze files
    - Cleans and validates data
    - Writes Parquet tables to S3
    """
    args = parse_args()

    logger.info("Starting Silver layer processing")

    for table_name, config in TABLE_CONFIG.items():
        try:
            df = read_bronze_table(
                bucket=args.bucket,
                file_name=config["file_name"],
                table_name=table_name,
            )

            df_clean = clean_table(
                df=df,
                table_name=table_name,
                config=config,
            )

            write_silver_table(
                df=df_clean,
                bucket=args.bucket,
                table_name=table_name,
            )

        except Exception as e:
            logger.error(f"Error processing {table_name}: {e}")
            raise

    logger.info("Silver layer completed successfully")


if __name__ == "__main__":
    main()
