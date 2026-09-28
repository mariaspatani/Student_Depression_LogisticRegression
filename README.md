# Student Depression Prediction using Logistic Regression

**Machine Learning Course Project – Team 16 (Metric Choice Investigation)**

---

## Overview

This project predicts **student depression** using a Logistic Regression classifier and investigates how different evaluation metrics influence model interpretation.

The objective is not only to train a classification model, but also to understand why **Accuracy alone is insufficient** for evaluating a mental health prediction model. Team 16 investigates the impact of **Precision, Recall, F1-Score, ROC-AUC, and decision threshold selection** on model performance.

---

## Project Objective

- Build a reproducible Logistic Regression model for student depression prediction.
- Perform preprocessing, feature encoding, train-test splitting, and feature scaling.
- Evaluate the model using multiple classification metrics.
- Investigate how changing the **decision threshold (0.3, 0.5, 0.7)** affects Precision, Recall, and F1-Score.
- Interpret the most influential features contributing to depression prediction.

---

## Dataset

The project uses the **Student Depression Dataset** stored in:

dataset/Student_Depression_Dataset.csv

The dataset contains demographic, academic, lifestyle, and mental health related attributes of students, along with a binary target variable indicating whether a student is experiencing depression.

### Target Variable

| Value | Meaning |
|-------|---------|
| 0 | No Depression |
| 1 | Depression |

---

## Project Structure

```text
Student_Depression_LogisticRegression/
├── dataset/
│   └── Student_Depression_Dataset.csv
│
├── notebooks/
│   └── Team16_Student_Depression.ipynb
│
├── outputs/
│   ├── confusion_matrix/
│   │   └── confusion_matrix.png
│   ├── graphs/
│   │   ├── roc_curve.png
│   │   ├── feature_importance.png
│   │   └── threshold_comparison.png
│   ├── metrics/
│   │   ├── model_metrics_summary.csv
│   │   └── threshold_comparison.csv
│   └── predictions/
│       └── student_predictions.csv
│
├── report/
│   ├── Investigation_Report.docx
│   └── figures/
│
├── presentation/
│   └── Team16_Presentation.pptx
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Python Environment Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Launch Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/Team16_Student_Depression.ipynb
```

Run all notebook cells sequentially.

---

## Machine Learning Workflow

The notebook follows the complete machine learning pipeline:

1. Load Dataset
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Label Encoding
5. Feature Selection
6. Train-Test Split (80% Train / 20% Test)
7. Standard Feature Scaling
8. Logistic Regression Training
9. Model Prediction
10. Model Evaluation
11. Investigation and Analysis

---

## Model Evaluation Metrics

The Logistic Regression model was evaluated using multiple classification metrics.

| Metric | Value |
|--------|--------|
| Accuracy | **84.48%** |
| Precision | **85.87%** |
| Recall (Sensitivity) | **87.97%** |
| F1-Score | **86.91%** |
| ROC-AUC Score | **91.82%** |

### Why Multiple Metrics?

- **Accuracy** measures overall correctness.
- **Precision** measures how many predicted depression cases are actually correct.
- **Recall** measures how many actual depression cases are successfully identified.
- **F1-Score** balances Precision and Recall.
- **ROC-AUC** measures the model's discrimination ability across thresholds.

---

## Investigation 1 — Decision Threshold Comparison

Team 16 investigated how changing the decision threshold affects model performance.

| Threshold | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| **0.3** | 82.76% | 79.75% | **94.58%** | 86.53% |
| **0.5** | **84.48%** | 85.87% | 87.97% | **86.91%** |
| **0.7** | 82.67% | **90.84%** | 78.30% | 84.11% |

### Key Findings

- **Threshold 0.3** identifies more students with depression (highest Recall) but increases false positives.
- **Threshold 0.5** provides the best overall balance between Precision and Recall.
- **Threshold 0.7** minimizes false positives (highest Precision) but misses more actual depression cases.

This investigation demonstrates that selecting a decision threshold depends on the application's objective.

---

## Investigation 2 — Feature Importance Analysis

The Logistic Regression coefficients were used to identify the most influential predictors.

### Top Influential Features

1. Have you ever had suicidal thoughts?
2. Academic Pressure
3. Financial Stress
4. Work/Study Hours
5. Family History of Mental Illness
6. Dietary Habits
7. Study Satisfaction
8. Sleep Duration
9. Age
10. Degree (Class 12)

### Observation

Mental health history and academic stress related features contribute more strongly to depression prediction than demographic features in this dataset.

---

## Investigation 3 — Confusion Matrix Analysis

The confusion matrix produced the following results.

| Prediction Outcome | Count |
|--------------------|------:|
| True Positive (TP) | **2875** |
| False Negative (FN) | **393** |
| False Positive (FP) | **473** |
| True Negative (TN) | **1840** |

### Interpretation

- The model correctly identified **2875 students with depression**.
- **393 students with depression were missed**.
- **473 students without depression** were incorrectly classified as depressed.
- **1840 students without depression** were correctly identified.

For mental health prediction, reducing **False Negatives** is especially important because missing a depressed student may delay intervention.

---

## Generated Outputs

The notebook automatically generates and saves:

### Graphs

- ROC Curve
- Confusion Matrix
- Feature Importance Graph
- Threshold Comparison Graph

### CSV Files

- Model Evaluation Metrics
- Threshold Comparison Metrics
- Student Prediction Results

All outputs are saved inside the `outputs/` directory.

---

## Results Summary

The Logistic Regression model achieved **84.48% accuracy** and **91.82% ROC-AUC**, indicating strong classification performance.

The threshold investigation showed that **Accuracy alone is not sufficient** for evaluating a mental health prediction model. Precision, Recall, and F1-Score provide additional insights for selecting an appropriate prediction threshold depending on the application.

---

## Team 16 Investigation Focus

**Metric Choice Investigation**

This project specifically investigates:

- Accuracy vs Precision vs Recall vs F1-Score.
- ROC-AUC interpretation.
- Decision threshold selection (0.3, 0.5, 0.7).
- Confusion Matrix based error analysis.
- Feature importance interpretation using Logistic Regression coefficients.

---

## Team

**Machine Learning Course Project**

**Team 16**
