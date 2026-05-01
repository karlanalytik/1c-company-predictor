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

This script downloads the Kaggle *Predict Future Sales* competition data and uploads the raw files to the Bronze layer in S3.

#### What it does

1. Reads Kaggle credentials from AWS Secrets Manager.
2. Downloads the competition files using `kagglehub`.
3. Uploads all raw files to: `s3://<bucket>/sales_predict/bronze/`

### How to Run
```bash
uv run python elt/bronze.py \
  --bucket your-bucket-name \
  --secret-name kaggle/credentials \
  --region-name us-east-1
```

## Use of AI Tools in the Project

- Generate documentation for functions and scripts.
- Questions: Add a role to SageMaker to access Secrets Manager for Kaggle.
- 
