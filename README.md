# Demand Forecasting Data Product (MVP)

## Project Objective and Description

This repository contains the implementation of a cloud-based data product designed to support demand planning, financial reporting, and operational decision-making through machine learning–driven forecasts.

The solution integrates data ingestion, transformation, and modeling pipelines to generate one-month-ahead demand forecasts, which are made accessible through an interactive web application. Business users can explore forecasts across multiple levels of aggregation (product, category, store, region, and channel), evaluate model performance through key metrics, generate downloadable reports on demand, and provide feedback on forecast quality.

The system is built on AWS using a modular architecture that separates data storage, machine learning workflows, and the application layer, ensuring scalability, reliability, and ease of use for non-technical stakeholders.

## Repository Structure

```
.
├── docs
│   ├── Demand Forecasting App.html
│   ├── diagrams
│   │   ├── Arquitectura.jpeg
│   │   └── ERD_1cCompany.png
│   ├── final_report.qmd
│   ├── images
│   │   ├── AWS_Secret_Manager2.jpeg
│   │   ├── AWS_Secret_Manager.jpeg
│   │   ├── Data_ML_Pipeline.png
│   │   ├── ECR.jpeg
│   │   ├── S3_bronze.png
│   │   ├── S3_gold.png
│   │   ├── S3_metrics_importance.png
│   │   ├── S3_model.png
│   │   ├── S3_predictions.png
│   │   ├── S3_silver.png
│   │   ├── S3_structure.png
│   │   ├── Streamlit_cluster.jpeg
│   │   └── Streamlit.jpeg
│   └── video
│       └── Tour de aplicacion.mp4
├── elt
│   ├── bronze.py
│   ├── gold.py
│   └── silver.py
├── LICENSE
├── models
│   ├── xgboost_predict.py
│   └── xgboost_train.py
├── pyproject.toml
├── README.md
├── run_pipeline.ipynb
├── streamlit
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── run_streamlit.ipynb
└── uv.lock
```

## Architecture

### Diagram

![Arquitectura](docs/diagrams/Arquitectura.jpeg)

### Services Justification

| Service | Justification |
|----------|---------------|
| **Amazon S3** | Provides secure and persistent storage outside the application, ensuring data availability even when the app is not running. Supports raw (CSV), processed data, models, and artifacts in a cost-effective and scalable way. Suitable for batch predictions with no transactional requirements. |
| **AWS Secrets Manager** | Securely manages credentials for data ingestion and service access, preventing exposure in code or configuration files and reducing security risks. |
| **AWS Glue Data Catalog** | Centralizes metadata for tables across Bronze, Silver, and Gold layers, enabling structured data discovery and consumption with low maintenance as a serverless service. |
| **SageMaker Processing Jobs / Notebooks** | Enables scheduled data processing, model training, and prediction generation. Cost-efficient as compute resources are used only during execution; easily scalable if needed. |
| **Amazon ECR** | Stores Docker images of the application, enabling reproducible builds and seamless deployment to ECS Fargate. |
| **Amazon ECS (Fargate)** | Deploys the web application in containers without managing servers, ensuring availability via URL and scalability based on demand. |
| **Amazon RDS** | Stores transactional data such as user feedback and application metadata, enabling structured data capture for future improvements. |
| **AWS CloudFormation** | Defines and deploys infrastructure as code, ensuring reproducibility, consistency, and easier scaling or modification of the solution. |


## ERD

![ERD](docs/diagrams/ERD_1cCompany.png)

[ERD Link](https://drive.google.com/file/d/1AD8gMyTQiBvfQw9bfuljlbslZcRxYDkO/view?usp=sharing)


## Installation and Setup

This project uses **`uv`** for Python environment and dependency management.

```bash
uv sync
```

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

### Gold Layer Processing

The script reads the clean Parquet tables from the Silver layer, creates a monthly modeling dataset, adds product, shop, and category attributes, and computes modeling features for training and evaluation.

#### What it does

1. Reads Silver Parquet tables from: `s3://<bucket>/sales_predict/silver/`
2. Aggregates daily sales to monthly shop-item level.
3. Merges sales with item, category, and shop dimensions.
4. Creates modeling features.
5. Adds 3-month naive baseline prediction.
6. Writes the modeling table to: `s3://<bucket>/sales_predict/gold/modeling_sales/`

#### How to Run

```bash
uv run python elt/gold.py \
  --bucket your-bucket-name
```

## Use of AI Tools in the Project

AI tools were used as a support resource throughout the development of this project, primarily to improve efficiency in documentation and communication tasks, as well as to assist in resolving specific technical questions.

Their use included:

Generating and refining documentation for functions and scripts.
Translating and synthesizing sections of the README (e.g., Project Objective and Description) based on the report content.
Reviewing and improving English writing, particularly for logging messages and technical descriptions, to ensure clarity and natural language usage.
Assisting with specific technical questions, such as configuring access roles (e.g., enabling SageMaker to securely access Secrets Manager for external sources like Kaggle), and setting up the pipeline for the notebook.
Supporting the refactoring of the prediction code.

All architectural decisions, system design, implementation, and validation of the solution were carried out by us.