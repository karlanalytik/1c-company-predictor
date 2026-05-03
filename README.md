# Demand Forecasting Data Product (MVP)

## Project Objective and Description

This repository contains the implementation of a cloud-based data product designed to support demand planning, financial reporting, and operational decision-making through machine learning–driven forecasts.

The solution integrates data ingestion, transformation, and modeling pipelines to generate one-month-ahead demand forecasts, which are made accessible through an interactive web application. Business users can explore forecasts across multiple levels of aggregation (product, category, store, region, and channel), evaluate model performance through key metrics, generate downloadable reports on demand, and provide feedback on forecast quality.

The system is built on AWS using a modular architecture that separates data storage, machine learning workflows, and the application layer, ensuring scalability, reliability, and ease of use for non-technical stakeholders.

## Repository Structure

```
.
├── docs
│   ├── diagrams
│   │   └── ERD_silver.drawio.png
│   ├── entregables.txt
│   ├── final_report_files
│   │   └── libs
│   │       ├── bootstrap
│   │       │   ├── bootstrap-756ac36e9b56ac242e07f29663160013.min.css
│   │       │   ├── bootstrap-82eac07570d2056c6c0099389084f61d.min.css
│   │       │   ├── bootstrap-icons.css
│   │       │   ├── bootstrap-icons.woff
│   │       │   └── bootstrap.min.js
│   │       ├── clipboard
│   │       │   └── clipboard.min.js
│   │       └── quarto-html
│   │           ├── anchor.min.js
│   │           ├── axe
│   │           │   └── axe-check.js
│   │           ├── popper.min.js
│   │           ├── quarto.js
│   │           ├── quarto-syntax-highlighting-ed96de9b727972fe78a7b5d16c58bf87.css
│   │           ├── tabsets
│   │           │   └── tabsets.js
│   │           ├── tippy.css
│   │           └── tippy.umd.min.js
│   ├── final_report.html
│   ├── final_report.qmd
│   └── images
├── elt
│   ├── bronze.py
│   └── silver.py
├── LICENSE
├── prueba.ipynb
├── pyproject.toml
├── README.md
└── uv.lock
```

## Architecture

### Diagram
- Link to .drawio file
- Image (PNG / SVG)

### Services Justification



- Architecture justification

- **Amazon S3**
- **Amazon ECS (Fargate)**
- **Amazon ECR**
- **AWS Glue Data Catalog**
- **Amazon RDS** — aunque sea para un subconjunto de datos operacionales (catálogos, metadata del POC, estado de jobs, feedback de negocio, lo que ustedes decidan)
- **AWS CloudFormation** — para el despliegue de la capa de infraestructura persistente del POC (la app y la base de datos), no a mano desde la consola
- **AWS Secrets Manager** — para gestionar las credenciales de la base de datos que lea su aplicación.
- **Amazon SageMaker — Batch Transform (opcional)** — si deciden usarlo como mecanismo para ejecutar inferencia batch desde la app, inclúyanlo en el diagrama y justifíquenlo. Si no lo usan, está bien — pero justifiquen en el reporte la alternativa que escogieron.

en el README deben poder defender por qué cada uno está ahí.



## ERD
- Link to .drawio file
- Image (PNG / SVG)

![ERD](docs/diagrams/ERD_1cCompany.png){width="75%"}

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

- Generate documentation for functions and scripts.
- Translate and sinthetize the README 'Project Objective and Description' section, based on our report section 1.
- Validate and improve (make more natural) or english translations for logging.
- Questions: Add a role to SageMaker to access Secrets Manager for Kaggle.
- 
