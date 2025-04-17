# Fraud Detection
Fraud Detection with GPU Acceleration Using RAPIDS

1. Problem Statement

The financial industry faces a significant challenge with transaction fraud. To mitigate these risks, we aim to detect fraudulent transactions from a dataset containing millions of records using GPU-accelerated machine learning.

2. Dataset Description

Dataset Source: Kaggle - Synthetic Financial Datasets For Fraud Detection

Filename: PS_20174392719_1491204439457_log.csv

Total Records: Over 6.3 million

Features: Transaction type, amount, origin and destination balance, etc.

Target: isFraud (0 = legitimate, 1 = fraud)

The dataset is highly imbalanced with ~8200 fraud cases among 6.3 million records.

3. Methodology and Implementation

Environment Setup

Due to memory limitations on Kaggle and local machines, we recommend using Google Colab with GPU runtime for RAPIDS support. RAPIDS libraries used:

cuDF for DataFrame operations

cuML for machine learning models

Preprocessing Steps

Removed nameOrig and nameDest (categorical identifiers)

One-hot encoded type using get_dummies

Split dataset with train_test_split (stratified to preserve class distribution)

from cuml.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

Models Implemented with cuML

1. Logistic Regression

from cuml.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

2. Random Forest

from cuml.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

3. K-Nearest Neighbors

from cuml.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

4. Model Performance Comparison

Model

Accuracy Score

Logistic Regression

~0.9960

Random Forest

~0.9992

K-Nearest Neighbors

~0.9989

Note: Metrics are biased due to class imbalance. Accuracy alone is insufficient. Precision, Recall, and F1-score for the minority class should be included in full analysis.

5. GPU Utilization and Performance Analysis

GPU Metrics Tracked

Utilization: ~80-95% during model training

Memory Usage: ~1.5–2.5 GB depending on model complexity

Training Time:

Logistic Regression: Fastest (~2s)

Random Forest: ~8–10s

KNN: ~6–8s

Observations

GPU-accelerated models run significantly faster than CPU-based equivalents

Larger datasets benefit greatly from RAPIDS due to parallel computation

Random Forest performed best in accuracy but required more memory

6. Conclusion and Recommendations

Summary

RAPIDS provides an effective GPU-accelerated workflow for large-scale fraud detection

Random Forest achieved the best accuracy but is more resource-intensive

Logistic Regression is fast and suitable for real-time inference

Recommendations

Address class imbalance with SMOTE, under-sampling, or anomaly detection techniques

Deploy models with precision-recall tradeoffs in mind (especially for rare events)

Consider using ROC-AUC and PR-AUC for performance evaluation
