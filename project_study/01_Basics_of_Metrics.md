# 01 — Basics of Evaluation Metrics

## 1. Why do we need evaluation metrics?

After training a machine learning classification model, we need to know how well the model is performing.

A model may make some predictions correctly and some incorrectly.

Evaluation metrics help us measure the model's performance from different perspectives.

For our project, this is important because our research question is:

> **How does the choice of evaluation metric change the conclusion about a model?**

We are using the **Student Depression Dataset** as the application for our investigation.

---

## 2. Confusion Matrix

For binary classification, the predictions can be organized into a confusion matrix.

There are four possible outcomes:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | TP | FN |
| **Actually Negative** | FP | TN |

### TP — True Positive

The model predicts **Positive**, and the actual class is **Positive**.

Example:

> Student is actually depressed → Model predicts depressed.

---

### TN — True Negative

The model predicts **Negative**, and the actual class is **Negative**.

Example:

> Student is actually not depressed → Model predicts not depressed.

---

### FP — False Positive

The model predicts **Positive**, but the actual class is **Negative**.

Example:

> Student is actually not depressed → Model predicts depressed.

This is also called a **false alarm**.

---

### FN — False Negative

The model predicts **Negative**, but the actual class is **Positive**.

Example:

> Student is actually depressed → Model predicts not depressed.

This means the model **missed a positive case**.

---

# 3. Accuracy

### Formula

Accuracy measures the proportion of all predictions that are correct.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Meaning

> **Out of all predictions, how many were correct?**

### Example

Suppose:

- TP = 80
- TN = 90
- FP = 10
- FN = 20

Then:

$$
Accuracy = \frac{80+90}{80+90+10+20}
$$

$$
Accuracy = \frac{170}{200} = 85\%
$$

### Important limitation

Accuracy can be misleading when the classes are **imbalanced**.

For example, if most students belong to the negative class, a model may get high accuracy simply by predicting the majority class frequently.

---

# 4. Precision

### Formula

$$
Precision = \frac{TP}{TP + FP}
$$

### Meaning

> **When the model predicts Positive, how often is it correct?**

In simple words:

> **"When I say YES, am I usually right?"**

Precision focuses on **false positives**.

### Example

Suppose the model predicts 100 students as positive.

- 80 are actually positive → TP = 80
- 20 are actually negative → FP = 20

Then:

$$
Precision = \frac{80}{80+20} = 80\%
$$

So, among the students predicted as positive, **80% were actually positive**.

---

# 5. Recall

### Formula

$$
Recall = \frac{TP}{TP + FN}
$$

### Meaning

> **Out of all actual Positive cases, how many did the model correctly identify?**

In simple words:

> **"Of all the actual YES cases, how many did I find?"**

Recall focuses on **false negatives**.

### Example

Suppose there are 100 actual positive cases.

The model correctly detects 80 of them.

- TP = 80
- FN = 20

Then:

$$
Recall = \frac{80}{80+20} = 80\%
$$

So the model detected **80% of the actual positive cases**.

---

# 6. Precision vs Recall

This is an important distinction for our project.

| Metric | Main Question | Focus |
|---|---|---|
| Precision | When I predict Positive, am I correct? | False Positives |
| Recall | Of all actual Positives, how many did I find? | False Negatives |

### Easy way to remember

**Precision:**

> "When I say YES, am I right?"

**Recall:**

> "Did I find all the YES cases?"

---

# 7. F1-Score

F1-score combines **Precision and Recall** into a single metric.

### Formula

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

It is the **harmonic mean** of precision and recall.

### Meaning

F1-score is useful when we want a balance between:

- Precision
- Recall

A high F1-score generally requires both precision and recall to be reasonably high.

---

# 8. ROC-AUC

ROC-AUC is another evaluation measure for classification.

### ROC

ROC stands for:

> **Receiver Operating Characteristic**

The ROC curve shows the relationship between:

- True Positive Rate
- False Positive Rate

at different classification thresholds.

### AUC

AUC stands for:

> **Area Under the Curve**

It summarizes the model's ability to distinguish between the two classes across thresholds.

A higher AUC generally indicates better class discrimination.

---

# 9. Why can different metrics give different conclusions?

This is the **main idea behind our project**.

Consider a model that produces:

| Metric | Score |
|---|---:|
| Accuracy | 95% |
| Precision | 70% |
| Recall | 55% |
| F1-score | 61% |
| ROC-AUC | 85% |

If we look only at **accuracy**, the model may appear to perform very well.

But recall tells us that the model is detecting only a smaller proportion of the actual positive cases.

Therefore:

> **The same model can look good according to one metric and less satisfactory according to another metric.**

This is why choosing an appropriate evaluation metric matters.

---

# 10. Class Imbalance

Class imbalance occurs when one class has many more samples than another class.

Example:

```text
Negative → 900 samples
Positive → 100 samples
