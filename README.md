# Title

## Project Objective and Description
Text

## Repository Structure

## Architecture
- Link to .drawio file
- Image (PNG / SVG)
- Architecture justification

## ERD
- Link to .drawio file
- Image (PNG / SVG)
- Architecture justification

## Installation and Setup


## ELT

### Bronze Layer Ingestion

The script downloads the Kaggle *Predict Future Sales* competition data and uploads the raw files to the Bronze layer in S3.

#### What it does

1. Reads Kaggle credentials from AWS Secrets Manager.
2. Downloads the competition files using `kagglehub`.
3. Uploads all raw files to: `s3://<bucket>/sales_predict/bronze/`

#### How to Run
```bash
uv run python elt/bronze.py \
  --bucket your-bucket-name \
  --secret-name kaggle/credentials \
  --region-name us-east-1
```

### Silver Layer Processing

The script reads the raw CSV files from the Bronze layer, applies basic data validations and transformations, and writes clean Parquet tables to the Silver layer in S3.

#### What it does

1. Reads raw CSV files from: `s3://<bucket>/sales_predict/bronze/`
2. Validates expected columns and data types.
3. Removes duplicated records.
4. Validates that key ID columns are unique in dimension tables.
5. Validates that no column contains only null values.
6. Writes clean Parquet tables to: `s3://<bucket>/sales_predict/silver/`

#### How to Run

```bash
uv run python elt/silver.py \
  --bucket your-bucket-name
```

## Use of AI Tools in the Project

- Generate documentation for functions and scripts.
- Questions: Add a role to SageMaker to access Secrets Manager for Kaggle.
- 
