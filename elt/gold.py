"""
Gold layer processing script.

Reads Silver Parquet tables from S3, creates a monthly modeling dataset,
adds product/shop/category attributes, and computes naive baseline features.

Flow:
1. Read Silver tables from S3
2. Aggregate sales to monthly level
3. Merge sales with item, category and shop dimensions
4. Add 3-month naive baseline prediction
5. Write Gold table to: s3://<bucket>/sales_predict/gold/modeling_sales/
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
    parser = argparse.ArgumentParser(description="Gold layer ingestion")

    parser.add_argument("--bucket", required=True, help="S3 bucket name")

    return parser.parse_args()


# ============================================================================
# Functions
# ============================================================================


def read_silver_table(bucket: str, table_name: str) -> pd.DataFrame:
    """
    Read one Silver table from S3.

    Args:
        bucket (str): source S3 bucket
        table_name (str): Silver table name

    Returns:
        Loaded DataFrame.
    """
    path = f"s3://{bucket}/sales_predict/silver/{table_name}/"

    logger.info(f"Reading {table_name} from {path}")

    try:
        return wr.s3.read_parquet(path)
    except Exception as e:
        logger.error(f"Error reading {table_name}: {e}")
        raise


def create_monthly_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales to monthly shop-item level.

    Args:
        sales (pd.DataFrame): Silver sales table

    Returns:
        Monthly sales DataFrame.
    """
    logger.info("Creating monthly sales table")

    sales["revenue"] = sales["item_price"] * sales["item_cnt_day"]

    monthly_sales = sales.groupby(
        ["date_block_num", "shop_id", "item_id"], as_index=False
    ).agg(
        item_cnt_month=("item_cnt_day", "sum"),
        avg_item_price=("item_price", "mean"),
        revenue_month=("revenue", "sum"),
    )

    logger.info(f"Monthly sales created: {len(monthly_sales)} rows")

    return monthly_sales


def merge_tables(
    monthly_sales: pd.DataFrame,
    items: pd.DataFrame,
    item_categories: pd.DataFrame,
    shops: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge monthly sales with item, category and shop dimensions.

    Args:
        monthly_sales (pd.DataFrame): monthly sales table
        items (pd.DataFrame): items dimension
        item_categories (pd.DataFrame): item categories dimension
        shops (pd.DataFrame): shops dimension

    Returns:
        Enriched modeling DataFrame.
    """
    logger.info("Merging monthly sales with catalogs")

    df = monthly_sales.merge(items, on="item_id", how="left")
    df = df.merge(item_categories, on="item_category_id", how="left")
    df = df.merge(shops, on="shop_id", how="left")

    logger.info(f"Gold table after merges: {len(df)} rows")

    return df


def add_naive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features and 3-month naive baseline prediction.

    Args:
        df (pd.DataFrame): modeling DataFrame

    Returns:
        DataFrame with naive baseline columns.
    """
    logger.info("Adding naive baseline features")

    df = df.sort_values(["shop_id", "item_id", "date_block_num"]).copy()

    group_cols = ["shop_id", "item_id"]

    df["item_cnt_month_lag_1"] = df.groupby(group_cols)["item_cnt_month"].shift(1)
    df["item_cnt_month_lag_2"] = df.groupby(group_cols)["item_cnt_month"].shift(2)
    df["item_cnt_month_lag_3"] = df.groupby(group_cols)["item_cnt_month"].shift(3)

    df["naive_3m_prediction"] = df[
        [
            "item_cnt_month_lag_1",
            "item_cnt_month_lag_2",
            "item_cnt_month_lag_3",
        ]
    ].mean(axis=1)

    logger.info("Naive baseline features added")

    return df


def validate_gold_table(df: pd.DataFrame) -> None:
    """
    Validate Gold table before writing to S3.

    Args:
        df (pd.DataFrame): Gold modeling table
    """
    logger.info("Validating Gold table")

    required_columns = [
        "date_block_num",
        "shop_id",
        "item_id",
        "item_cnt_month",
        "item_category_id",
        "item_category_name",
        "shop_name",
        "naive_3m_prediction",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        logger.error(f"Gold table missing columns: {missing}")
        raise ValueError(f"Gold table missing columns: {missing}")

    if df.empty:
        logger.error("Gold table is empty")
        raise ValueError("Gold table is empty")

    logger.info("Gold table validation completed")


def write_gold_table(df: pd.DataFrame, bucket: str, table_name: str) -> None:
    """
    Write Gold modeling table to S3 as Parquet.

    Args:
        df (pd.DataFrame): Gold modeling table
        bucket (str): target S3 bucket
        table_name (str): table name
    """
    path = f"s3://{bucket}/sales_predict/gold/{table_name}/"

    logger.info(f"Writing Gold table to {path}")

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
    Orchestrates the Gold layer process:
    - Reads Silver tables
    - Creates monthly modeling dataset
    - Adds naive baseline features
    - Writes Gold table to S3
    """
    args = parse_args()

    logger.info("Starting Gold layer processing")

    try:
        sales = read_silver_table(args.bucket, "sales")
        items = read_silver_table(args.bucket, "items")
        item_categories = read_silver_table(args.bucket, "item_categories")
        shops = read_silver_table(args.bucket, "shops")

        monthly_sales = create_monthly_sales(sales)

        gold_df = merge_tables(
            monthly_sales=monthly_sales,
            items=items,
            item_categories=item_categories,
            shops=shops,
        )

        gold_df = add_naive_features(gold_df)

        validate_gold_table(gold_df)

        write_gold_table(gold_df, args.bucket, "modeling_sales")

    except Exception:
        logger.exception("Error processing Gold layer")
        raise

    logger.info("Gold layer completed successfully")


if __name__ == "__main__":
    main()
