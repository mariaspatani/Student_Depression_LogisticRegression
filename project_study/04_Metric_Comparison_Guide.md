# 04 — Metric Comparison Guide: Tradeoffs and Contextual Evaluation

## 1. Contextual Importance of Evaluation Metrics

In machine learning, no single metric tells the whole story. The "best" metric depends entirely on the operational domain and the relative costs of errors.

For our project on **Student Depression Prediction**:
- **False Negative (FN):** A student experiencing depression is predicted as not depressed.
  - *Consequence:* The student is denied counseling, early intervention, or academic accommodations, potentially worsening severe mental health risks.
- **False Positive (FP):** A student not experiencing depression is predicted as depressed.
  - *Consequence:* The student is invited for a supportive screening interview or wellness check-in, causing minimal inconvenience or slight resource overhead.

Because the consequence of a False Negative is far more harmful than a False Positive in this context, **Recall (Sensitivity)** is practically more important than Precision.

---

## 2. Accuracy vs. Sensitivity / Specificity

Accuracy treats all errors equally:
- An FP and an FN penalize accuracy by the exact same amount: $\frac{1}{N}$.
- However, when thresholds shift from $0.3$ to $0.7$:
  - At threshold $0.3$: Accuracy = $82.76\%$, Recall = $94.58\%$, Precision = $79.75\%$
  - At threshold $0.7$: Accuracy = $82.67\%$, Recall = $78.30\%$, Precision = $90.84\%$

Notice that the **overall accuracy is virtually identical** ($82.76\%$ vs $82.67\%$), but the clinical utility is radically different! At $0.7$, the model misses over $21\%$ of depressed students, whereas at $0.3$, it catches nearly $95\%$.

Accuracy completely fails to reflect this distinction.

---

## 3. Precision-Recall Tradeoff

As the decision threshold $\tau$ changes:
1. **Lower Threshold ($\tau \downarrow$):**
   - Flags more cases as Positive
   - TP $\uparrow$, FN $\downarrow \implies$ **Recall increases**
   - FP $\uparrow \implies$ **Precision decreases**
2. **Higher Threshold ($\tau \uparrow$):**
   - Requires stronger certainty before flagging Positive
   - FP $\downarrow \implies$ **Precision increases**
   - FN $\uparrow \implies$ **Recall decreases**

The F1-score acts as the harmonic mean, punishing extreme imbalances between Precision and Recall.

---

## 4. ROC-AUC vs. PR-AUC

| Feature | ROC Curve | Precision-Recall (PR) Curve |
|---|---|---|
| **X-axis** | False Positive Rate ($FPR = \frac{FP}{FP + TN}$) | Recall ($TPR = \frac{TP}{TP + FN}$) |
| **Y-axis** | True Positive Rate ($TPR = \frac{TP}{TP + FN}$) | Precision ($\frac{TP}{TP + FP}$) |
| **Threshold independent?** | Yes, evaluates ranking across all possible thresholds | Yes, evaluates across all thresholds |
| **Baseline value** | 0.50 (random chance) | Positive class proportion ($P / (P + N)$) |
| **Sensitivity to imbalance** | Less sensitive; large $TN$ can compress FPR | Highly sensitive to class distribution |

For Team 16's model, **ROC-AUC is 91.82%**, indicating strong discriminative capability regardless of the threshold chosen.

---

## 5. Metric Selection Matrix

| Scenario | Recommended Metric | Primary Objective |
|---|---|---|
| Balanced classes, equal error costs | Accuracy, Balanced Accuracy | Overall correct classification |
| Screening / Medical / Safety critical | Recall, Specificity | Minimize missed cases (False Negatives) |
| Spam detection / Legal prosecution | Precision | Minimize false accusations (False Positives) |
| Threshold-free model ranking | ROC-AUC | Evaluate ranking discrimination |
| Imbalanced positive detection | F1-Score, PR-AUC | Balance precision with recall |
