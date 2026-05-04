"""
XGBoost inference script.

Reads the trained XGBoost model from S3, prepares month 34 data using
the Silver test table and Gold historical features, and writes predictions
to S3 in Parquet format.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import awswrangler as wr
import joblib
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
    parser = argparse.ArgumentParser(description="Generate XGBoost predictions")

    parser.add_argument("--bucket", required=True, help="S3 bucket name")

    return parser.parse_args()


# ============================================================================
# Config
# ============================================================================

FEATURES = [
    "date_block_num",
    "shop_id",
    #"city_code",
    "item_id",
    "item_category_id",
    #"main_category_code",
    #"avg_item_price",
    #"revenue_month",
    "item_cnt_month_lag_1",
    "item_cnt_month_lag_2",
    "item_cnt_month_lag_3",
    "naive_3m_prediction",
]

MODEL_NAME = "xgboost_simple"
FORECAST_MONTH = 34


# ============================================================================
# Functions
# ============================================================================


def read_gold_data(bucket: str) -> pd.DataFrame:
    """
    Read Gold modeling table from S3.
    """
    path = f"s3://{bucket}/sales_predict/gold/modeling_sales/"

    logger.info(f"Reading Gold data from {path}")

    return wr.s3.read_parquet(path, dataset=True)


def read_test_data(bucket: str) -> pd.DataFrame:
    """
    Read Silver test table from S3.
    """
    path = f"s3://{bucket}/sales_predict/silver/test/"

    logger.info(f"Reading Silver test data from {path}")

    return wr.s3.read_parquet(path)


def load_model_from_s3(bucket: str):
    """
    Load trained model artifact from S3.
    """
    s3_path = f"s3://{bucket}/sales_predict/models/xgboost_model.joblib"

    logger.info(f"Loading model from {s3_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / "xgboost_model.joblib"

        wr.s3.download(
            path=s3_path,
            local_file=str(local_path),
        )

        model = joblib.load(local_path)

    return model


def create_forecast_dataset(
    test_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create month 34 feature table using Silver test and Gold history.
    """
    logger.info("Creating forecast dataset for month 34")

    last_history = (
        gold_df.sort_values(["shop_id", "item_id", "date_block_num"])
        .groupby(["shop_id", "item_id"], as_index=False)
        .tail(1)
        .copy()
    )

    last_history["item_cnt_month_lag_3"] = last_history["item_cnt_month_lag_2"]
    last_history["item_cnt_month_lag_2"] = last_history["item_cnt_month_lag_1"]
    last_history["item_cnt_month_lag_1"] = last_history["item_cnt_month"]

    lag_cols = [
        "item_cnt_month_lag_1",
        "item_cnt_month_lag_2",
        "item_cnt_month_lag_3",
    ]

    last_history[lag_cols] = last_history[lag_cols].fillna(0).clip(lower=0, upper=20)

    last_history["naive_3m_prediction"] = last_history[lag_cols].mean(axis=1)

    feature_cols = [
        "shop_id",
        "item_id",
        "city_code",
        "item_category_id",
        "main_category_code",
        "inactive",
        "item_cnt_month",
        "item_cnt_month_lag_1",
        "item_cnt_month_lag_2",
        "item_cnt_month_lag_3",
        "naive_3m_prediction",
    ]

    forecast_df = test_df.merge(
        last_history[feature_cols],
        on=["shop_id", "item_id"],
        how="left",
    )

    forecast_df["date_block_num"] = FORECAST_MONTH

    missing_rows = forecast_df[FEATURES].isna().any(axis=1).sum()

    if missing_rows > 0:
        logger.warning(f"Filling {missing_rows} rows with missing features")

    forecast_df[FEATURES] = forecast_df[FEATURES].fillna(0)

    return forecast_df


def generate_predictions(model, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate XGBoost predictions.
    """
    logger.info("Generating predictions")

    predictions = forecast_df.copy()

    predictions["prediction"] = model.predict(predictions[FEATURES])
    predictions["prediction"] = predictions["prediction"].clip(lower=0, upper=20)

    if "inactive" in predictions.columns:
        predictions.loc[predictions["inactive"] == 1, "prediction"] = 0
    predictions["model_name"] = MODEL_NAME

    return predictions


def save_predictions_to_s3(predictions: pd.DataFrame, bucket: str) -> None:
    """
    Save predictions to S3 as Parquet.
    """
    output_path = f"s3://{bucket}/sales_predict/predictions/xgboost_forecast/"

    logger.info(f"Writing predictions to {output_path}")

    cols = [
        "id",
        "shop_id",
        "item_id",
        "item_category_id",
        "main_category_code",
        "city_code",
        "inactive",
        "date_block_num",
        "prediction",
        "model_name",
    ]

    wr.s3.to_parquet(
        df=predictions[cols],
        path=output_path,
        dataset=True,
        mode="overwrite",
        index=False,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    """
    Orchestrates XGBoost inference:
    - Reads Silver test table
    - Reads Gold historical features
    - Loads trained model
    - Generates month 34 predictions
    - Writes predictions to S3
    """
    args = parse_args()

    logger.info("Starting XGBoost inference")

    try:
        test_df = read_test_data(args.bucket)
        gold_df = read_gold_data(args.bucket)
        model = load_model_from_s3(args.bucket)

        forecast_df = create_forecast_dataset(
            test_df=test_df,
            gold_df=gold_df,
        )

        predictions = generate_predictions(
            model=model,
            forecast_df=forecast_df,
        )

        save_predictions_to_s3(
            predictions=predictions,
            bucket=args.bucket,
        )

    except Exception:
        logger.exception("Error generating XGBoost predictions")
        raise

    logger.info("XGBoost inference completed successfully")


if __name__ == "__main__":
    main()