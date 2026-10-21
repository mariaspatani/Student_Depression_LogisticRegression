"""
scripts/metrics_summary.py
--------------------------
Calculates and displays all primary and derived classification metrics
from confusion matrix components (TP, FN, FP, TN) and summarizes saved model metrics.
"""

import math
import os
import csv

def compute_all_metrics(tp=2875, fn=393, fp=473, tn=1840, roc_auc=0.9182):
    total = tp + fn + fp + tn
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0
    
    # Matthews Correlation Coefficient (MCC)
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "Accuracy": (accuracy, "Proportion of all predictions that are correct"),
        "Precision": (precision, "Positive predictive value (P(Actual + | Pred +))"),
        "Recall (Sensitivity)": (recall, "Detection rate of actual positive cases"),
        "Specificity": (specificity, "True negative rate (P(Pred - | Actual -))"),
        "F1-Score": (f1, "Harmonic mean of Precision and Recall"),
        "Balanced Accuracy": (balanced_acc, "Macro average of recall and specificity"),
        "ROC-AUC": (roc_auc, "Area under the ROC discrimination curve"),
        "Matthews Corr. Coeff.": (mcc, "Balanced measure handling class distribution"),
    }

def print_summary():
    metrics = compute_all_metrics()
    
    print("=" * 78)
    print("Team 16 — Comprehensive Model Evaluation Metrics Summary")
    print("=" * 78)
    print(f"{'Evaluation Metric':<26} | {'Value':<8} | {'Percent':<8} | {'Description'}")
    print("-" * 78)
    
    for metric_name, (val, desc) in metrics.items():
        pct = f"{val * 100:.2f}%" if metric_name != "Matthews Corr. Coeff." else f"{val:.4f}"
        val_str = f"{val:.4f}"
        print(f"{metric_name:<26} | {val_str:<8} | {pct:<8} | {desc}")

    print("-" * 78)
    print("Confusion Matrix Counts (Test Set N=5,581):")
    print("  True Positives (TP): 2,875  |  False Negatives (FN): 393")
    print("  False Positives (FP):  473  |  True Negatives (TN): 1,840")
    print("=" * 78)

if __name__ == "__main__":
    print_summary()
