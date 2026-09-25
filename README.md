# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using multiple classification models, ensemble learning, hyperparameter tuning, and threshold optimization.

## Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent a very small portion of all transactions.

This project builds an end-to-end fraud detection pipeline covering:

- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Feature Selection
- Class Imbalance Handling
- Multiple Machine Learning Models
- Cross-Validation
- Hyperparameter Tuning
- Classification Threshold Optimization
- Streamlit Deployment

## Dataset

The dataset contains credit card transaction records with:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

Target variable:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

The original dataset contains 284,807 transactions.

After removing 1,081 exact duplicate rows, the cleaned dataset contains 283,726 transactions.

## Project Structure

```text
fraud-detection/
├── data/
│   ├── raw/
│   │   └── fraud_data.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── models/
│
├── tests/
│   └── test_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE