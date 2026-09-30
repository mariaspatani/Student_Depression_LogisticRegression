# Investigating the Effect of Evaluation Metric Choice on Model Conclusions

## Team 16 — Metric Choice

### Logistic Regression on the Student Depression Dataset

| **Name** |
|---|
| Alvin Joseph Austin |
| Alwin Jose George |
| Maria S Patani |
| Nandana Hasheed |

**College:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)  
**Course:** 24SJPCCST503 – Machine Learning  
**Academic Year:** 2026–2027

---

## 1. Investigation Topic

**Metric Choice**

## 2. Research Question

> **How does the choice of evaluation metric change the conclusion about a model?**

## 3. Hypothesis

> Different evaluation metrics will provide different interpretations of the performance of the same machine-learning model because they emphasize different aspects of classification behaviour.

---

## 4. Overview

This project is an **experimental Machine Learning investigation**, not simply a model-building task.

A single **Logistic Regression** model was trained on the Student Depression Dataset. The same test set and the same predictions were then evaluated using multiple evaluation metrics.

The investigation examines how the interpretation of model performance changes when the evaluation metric changes.

### Reasoning Chain

**Question → Hypothesis → Experiment → Evidence → Analysis → Conclusion**

---

## 5. Dataset

### Student Depression Dataset

The dataset was obtained from Kaggle.

- **Records:** 27,901
- **Columns after ID removal:** 17
- **Target:** `Depression`
- **Task:** Binary classification
- **Encoded input features:** 102
- **Training observations:** 22,320
- **Test observations:** 5,581
- **Train-test split:** 80/20 stratified split
- **Random state:** 42

The test set contains:

- **Positive class:** 58.56%
- **Negative class:** 41.44%

The dataset contains demographic, academic, lifestyle, financial and family-related variables.

### Dataset Limitation

The dataset is based on self-reported student survey data, and its sampling procedure is not documented. Therefore, the findings may not generalize to all student populations.

---

## 6. Experimental Design

The central experimental principle is:

> **The model, test set, and predictions remain constant while the evaluation metric changes.**

| **Controlled Element** | **Design** |
|---|---|
| Model | Logistic Regression |
| Input Features | 102 encoded features |
| Train/Test Split | 80% / 20%, stratified |
| Random State | 42 |
| Test Observations | 5,581 |
| Prediction Output | Class labels and probabilities |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-score, ROC-AUC |

The investigation therefore isolates the effect of **metric choice** on the interpretation of model performance.

---

## 7. Evaluation Metrics

| **Metric** | **What it measures** |
|---|---|
| **Accuracy** | Proportion of all predictions that are correct |
| **Precision** | Proportion of predicted positive cases that are actually positive |
| **Recall** | Proportion of actual positive cases correctly detected |
| **F1-score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Ability of probability scores to discriminate between the two classes across thresholds |

These metrics are not interchangeable because each answers a different evaluation question.

---

## 8. Main Results

| **Metric** | **Value** |
|---|---:|
| Accuracy | **84.48%** |
| Precision | **85.87%** |
| Recall | **87.97%** |
| F1-score | **86.91%** |
| ROC-AUC | **91.82%** |

### Confusion Matrix

- **True Positives:** 2,875
- **False Negatives:** 393
- **False Positives:** 473
- **True Negatives:** 1,840

These values correspond to all **5,581 test observations**.

---

## 9. Additional Evidence — Threshold Analysis

The same trained model and the same predicted probabilities were evaluated at three decision thresholds.

| **Threshold** | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|---|---:|---:|---:|---:|
| 0.3 | 82.76% | 79.75% | **94.58%** | 86.53% |
| 0.5 | **84.48%** | 85.87% | 87.97% | **86.91%** |
| 0.7 | 82.67% | **90.84%** | 78.30% | 84.11% |

### Key Observation

Accuracy at thresholds **0.3 and 0.7** is almost identical:

- 0.3 → **82.76%**
- 0.7 → **82.67%**

However:

- Recall changes from **94.58% → 78.30%**
- Precision changes from **79.75% → 90.84%**

This shows that accuracy alone can hide important differences in model behaviour.

---

## 10. Analysis

The same predictions produce different numerical descriptions because each metric measures a different property.

### Accuracy
Provides an overall measure of correct predictions but does not show how errors are distributed.

### Precision
Focuses on the correctness of positive predictions and is particularly affected by false positives.

### Recall
Focuses on detecting actual positive cases and is particularly affected by false negatives.

### F1-score
Provides a balance between Precision and Recall.

### ROC-AUC
Evaluates how well the model's probability scores discriminate between the two classes across different thresholds.

ROC-AUC should not be interpreted as a higher form of accuracy. It measures a different property.

---

## 11. Supporting Evidence

Additional metrics derived from the confusion matrix:

| **Metric** | **Value** |
|---|---:|
| Specificity | 79.55% |
| Balanced Accuracy | 83.76% |
| Matthews Correlation Coefficient | 0.6790 |

The model's Recall (**87.97%**) is higher than its Specificity (**79.55%**), showing different behaviour for the positive and negative classes.

---

## 12. Critical Evaluation and Limitations

- Only one Logistic Regression model was evaluated.
- Only one train-test split was used.
- The dataset's sampling procedure is not documented.
- Some preprocessing was performed before the train-test split, creating a preprocessing leakage risk.
- The findings may not generalize to other student populations.
- The experiment is a machine-learning prediction study, not a clinical diagnostic system.

### Possible Improvements

A stronger follow-up investigation could:

- Use cross-validation.
- Repeat the experiment across multiple train-test splits.
- Quantify variation in the evaluation metrics.
- Define practical costs for false positives and false negatives.
- Compare how metric conclusions change across additional datasets or models.

---

## 13. Conclusion

The investigation supports the hypothesis within this experimental setting.

The same Logistic Regression model and the same test predictions produced different evaluations:

- Accuracy = **84.48%**
- Precision = **85.87%**
- Recall = **87.97%**
- F1-score = **86.91%**
- ROC-AUC = **91.82%**

Therefore, a model should not be described using a single evaluation metric without considering what that metric actually measures.

The findings are limited to the selected dataset, model and experimental split. Broader conclusions require further validation.

---

## 14. Repository Structure

```text
ML_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/
│   └── Student_Depression_Dataset.csv
│
├── notebooks/
│   └── Team16_Student_Depression.ipynb
│
├── outputs/
│   ├── graphs/
│   ├── confusion_matrix/
│   ├── metrics/
│   └── predictions/
│
├── report/
│   └── Team16_Report_Final.docx
│
└── presentation/
    └── Team16_Presentation.pptx

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


