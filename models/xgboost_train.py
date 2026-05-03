"""
XGBoost training script.

Reads Gold modeling data from S3, trains a XGBoost regression model,
and writes model artifacts and predictions to S3.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import awswrangler as wr
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


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
    parser = argparse.ArgumentParser(description="Train simple XGBoost model")

    parser.add_argument("--bucket", required=True, help="S3 bucket name")

    return parser.parse_args()


# ============================================================================
# Config
# ============================================================================

FEATURES = [
    "date_block_num",
    "shop_id",
    "city_code",
    "item_id",
    "item_category_id",
    #"avg_item_price",
    #"revenue_month",
    "item_cnt_month_lag_1",
    "item_cnt_month_lag_2",
    "item_cnt_month_lag_3",
    "naive_3m_prediction",
]

TARGET = "item_cnt_month"
MODEL_NAME = "xgboost_simple"
BACKTEST_MONTH = 33
FORECAST_MONTH = 34


# ============================================================================
# Functions
# ============================================================================


def read_gold_data(bucket: str) -> pd.DataFrame:
    """
    Read Gold modeling table from S3.

    Args:
        bucket (str): source S3 bucket

    Returns:
        Gold DataFrame.
    """
    path = f"s3://{bucket}/sales_predict/gold/modeling_sales/"

    logger.info(f"Reading Gold data from {path}")

    return wr.s3.read_parquet(path, dataset=True)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare training data.

    Args:
        df (pd.DataFrame): Gold DataFrame

    Returns:
        Clean modeling DataFrame.
    """
    logger.info("Preparing training data")

    required_cols = FEATURES + [TARGET, "inactive"]
    missing = set(required_cols) - set(df.columns)

    if missing:
        logger.error(f"Missing columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_cols].copy()
    df = df[df["inactive"] == 0]
    df = df.dropna()
    df['date_block_num'] = df['date_block_num'].astype(int)

    logger.info(f"Training data prepared: {len(df)} rows")

    return df


def train_model(df: pd.DataFrame) -> tuple[XGBRegressor, pd.DataFrame]:
    """
    Train a simple XGBoost model.

    Args:
        df (pd.DataFrame): modeling DataFrame

    Returns:
        Trained model and predictions DataFrame.
    """
    logger.info("Training XGBoost model")

    train_df = df[df["date_block_num"] < 33].copy()
    test_df = df[df["date_block_num"] == 33].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].clip(lower=0, upper=20)

    X_test = test_df[FEATURES]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    test_df["prediction"] = model.predict(X_test)
    test_df["prediction"] = test_df["prediction"].clip(lower=0, upper=20)
    test_df["model_name"] = "xgboost_simple"

    logger.info("Model training completed")

    return model, test_df


def evaluate_backtesting(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute backtesting metrics for XGBoost and naive baseline.

    Args:
        predictions (pd.DataFrame): test DataFrame with actuals, model predictions
            and naive baseline predictions.

    Returns:
        DataFrame with backtesting metrics.
    """
    logger.info("Evaluating backtesting performance")

    y_true = predictions[TARGET].clip(lower=0, upper=20)
    y_pred = predictions["prediction"].clip(lower=0, upper=20)
    y_naive = predictions["naive_3m_prediction"].clip(lower=0, upper=20)

    mae_model = mean_absolute_error(y_true, y_pred)
    rmse_model = mean_squared_error(y_true, y_pred) ** 0.5

    mae_naive = mean_absolute_error(y_true, y_naive)
    rmse_naive = mean_squared_error(y_true, y_naive) ** 0.5

    metrics = pd.DataFrame(
        [
            {
                "model_name": "xgboost_simple",
                "backtest_month": 33,
                "mae_model": mae_model,
                "rmse_model": rmse_model,
                "mae_naive": mae_naive,
                "rmse_naive": rmse_naive,
                "mae_improvement": mae_naive - mae_model,
                "rmse_improvement": rmse_naive - rmse_model,
            }
        ]
    )

    logger.info(f"Backtesting metrics:\n{metrics.to_string(index=False)}")

    return metrics


def save_model_to_s3(model: XGBRegressor, bucket: str) -> None:
    """
    Save trained model artifact to S3.

    Args:
        model (XGBRegressor): trained model
        bucket (str): target S3 bucket
    """
    s3_path = f"s3://{bucket}/sales_predict/models/xgboost_model.joblib"

    logger.info(f"Saving model to {s3_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / "xgboost_model.joblib"
        joblib.dump(model, local_path)

        wr.s3.upload(
            local_file=str(local_path),
            path=s3_path,
        )


def save_backtest_predictions_to_s3(
    predictions: pd.DataFrame,
    bucket: str,
) -> None:
    output_path = (
        f"s3://{bucket}/sales_predict/predictions/backtesting_{MODEL_NAME}.csv"
    )

    logger.info(f"Writing backtesting predictions to {output_path}")

    cols = [
        "shop_id",
        "item_id",
        "item_category_id",
        "date_block_num",
        TARGET,
        "naive_3m_prediction",
        "prediction",
        "model_name",
    ]

    wr.s3.to_csv(
        df=predictions[cols],
        path=output_path,
        index=False,
    )


def save_metrics_to_s3(metrics: pd.DataFrame, bucket: str) -> None:
    output_path = f"s3://{bucket}/sales_predict/artifacts/metrics_{MODEL_NAME}.csv"

    logger.info(f"Writing metrics to {output_path}")

    wr.s3.to_csv(
        df=metrics,
        path=output_path,
        index=False,
    )

def save_feature_importance_to_s3(model: XGBRegressor, bucket: str) -> None:
    """
    Save feature importance to S3.

    Args:
        model (XGBRegressor): trained model
        bucket (str): target S3 bucket
    """
    output_path = (
        f"s3://{bucket}/sales_predict/artifacts/xgboost_feature_importance.csv"
    )

    logger.info(f"Writing feature importance to {output_path}")

    df_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    wr.s3.to_csv(
        df=df_importance,
        path=output_path,
        index=False,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    """
    Orchestrates XGBoost training:
    - Reads Gold data
    - Prepares training dataset
    - Trains model
    - Saves model and prediction artifacts
    """
    args = parse_args()

    logger.info("Starting XGBoost training")

    try:
        gold_df = read_gold_data(args.bucket)
        modeling_df = prepare_data(gold_df)

        model, predictions = train_model(modeling_df)

        metrics = evaluate_backtesting(predictions)

        save_model_to_s3(model, args.bucket)
        save_backtest_predictions_to_s3(predictions, args.bucket)
        save_metrics_to_s3(metrics, args.bucket)
        save_feature_importance_to_s3(model, args.bucket)

    except Exception:
        logger.exception("Error training XGBoost model")
        raise

    logger.info("XGBoost training completed successfully")


if __name__ == "__main__":
    main()