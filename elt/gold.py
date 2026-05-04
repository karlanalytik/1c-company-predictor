"""
Gold layer processing script.

Reads Silver Parquet tables from S3, creates a monthly modeling dataset,
adds product/shop/category attributes, creates modeling features, and computes 
naive baseline features.

Flow:
1. Read Silver tables from S3
2. Aggregate sales to monthly level
3. Merge sales with item, category and shop dimensions
4. Create modeling features
5. Add 3-month naive baseline prediction
6. Write Gold table to: s3://<bucket>/sales_predict/gold/modeling_sales/
"""

import argparse
import datetime
import logging
import numpy as np

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
        return wr.s3.read_parquet(path, dataset=True)
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

    monthly_sales["item_cnt_month"] = monthly_sales["item_cnt_month"].clip(0, 20)

    logger.info(f"Monthly sales created: {len(monthly_sales)} rows")

    return monthly_sales


def add_city_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract city name from shop_name and encode it as a numeric feature.
    """
    df = df.copy()

    df["city_name"] = (
        df["shop_name"]
        .str.strip()
        .str.replace(r"^[^A-Za-zА-Яа-я]+", "", regex=True)
        .str.split()
        .str[0]
    )

    city_mapping = {
        "СПб": "Санкт-Петербург",
        "Н.Новгород": "Нижний_Новгород",
        "РостовНаДону": "Ростов_на_Дону",
        "Выездная": "online_other",
        "Интернет-магазин": "online_other",
        "Цифровой": "online_other",
    }

    df["city_name"] = df["city_name"].replace(city_mapping)
    df["city_code"] = df["city_name"].astype("category").cat.codes

    return df

def add_main_category_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract main category from item_category_name and encode it as a numeric feature.
    """
    df = df.copy()

    df["main_category"] = (
        df["item_category_name"]
        .str.strip()
        .str.split(" - ")
        .str[0]
    )

    category_mapping = {
        "Игры Android": "Игры",
        "Игры MAC": "Игры",
        "Карты оплаты (Кино, Музыка, Игры)": "Карты оплаты",
        "Чистые носители (шпиль)": "Чистые носители",
        "Чистые носители (штучные)": "Чистые носители",
        "Билеты (Цифра)": "Билеты",
    }

    df["main_category_name"] = df["main_category"].replace(category_mapping)

    df["main_category_code"] = df["main_category"].astype("category").cat.codes

    return df

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

    lag_cols = []
    for lag in [1, 2, 3]:
        col = f"item_cnt_month_lag_{lag}"
        lag_cols.append(col)
    
    df[lag_cols + ["naive_3m_prediction"]] = (
        df[lag_cols + ["naive_3m_prediction"]]
        .fillna(0)
        .clip(lower=0, upper=20)
    )
    
    df["has_history"] = (df["item_cnt_month_lag_1"] > 0).astype(int)

    logger.info("Naive baseline features added")

    return df

def identify_inactive_items(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Identify inactive items based on their last observed sale date.

    An item is marked as inactive when its last sale date is older than the
    threshold defined by the maximum date in the dataset minus the number of days.

    Args:
        df (pd.DataFrame): sales DataFrame with item_id and date columns.
        days (int): inactivity threshold in days.

    Returns:
        DataFrame with item_id, last_sale_date, and inactive flag.
    """
    logger.info("Identifying inactive items")
    
    last_sales = (
        df.groupby("item_id", as_index=False)["date"]
        .max()
        .reset_index()
    )

    last_sales["date"] = pd.to_datetime(last_sales["date"], format = "%d-%m-%Y")
    
    last_sale_limit = last_sales["date"].max() - datetime.timedelta(days=days)

    last_sales["inactive"] = np.where(last_sales["date"] <= last_sale_limit, 1, 0)

    logger.info("Inactive item flag created")

    return last_sales


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
        "city_name",
        "city_code",
        "item_id",
        "item_cnt_month",
        "item_category_id",
        "item_category_name",
        "main_category_code",
        "main_category_name",
        "shop_name",
        "naive_3m_prediction",
        "inactive"
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
        partition_cols=['date_block_num'],
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
        shops = add_city_feature(shops)
        item_categories = add_main_category_feature(item_categories)

        gold_df = merge_tables(
            monthly_sales=monthly_sales,
            items=items,
            item_categories=item_categories,
            shops=shops,
        )

        gold_df = add_naive_features(gold_df)
        
        inactive_items = identify_inactive_items(sales, days=365)

        gold_df = gold_df.merge(
            inactive_items[["item_id", "inactive"]],
            on="item_id",
            how="left",
        )

        validate_gold_table(gold_df)

        write_gold_table(gold_df, args.bucket, "modeling_sales")

    except Exception:
        logger.exception("Error processing Gold layer")
        raise

    logger.info("Gold layer completed successfully")


if __name__ == "__main__":
    main()
