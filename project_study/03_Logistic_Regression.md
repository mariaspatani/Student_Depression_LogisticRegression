# 03 — Logistic Regression: Theory and Role in Metric Investigation

## 1. What is Logistic Regression?

Logistic Regression is a foundational supervised machine learning algorithm for binary classification problems.

Despite having "regression" in its name, it is used for **classification**:
- Predicting a discrete outcome: $y \in \{0, 1\}$
- In our project:
  - $y = 0$: No Depression
  - $y = 1$: Depression

Unlike algorithms that only output hard class labels, Logistic Regression models the **probability** that an observation belongs to the positive class:

$$
P(Y = 1 | X)
$$

---

## 2. Mathematical Formulation

### 2.1 Linear Combination

For an observation with feature vector $X = (x_1, x_2, \dots, x_n)$, the model computes a linear combination using learned weights $w$ and bias $b$:

$$
z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n = w^T X + b
$$

### 2.2 Sigmoid Activation Function

To map $z$ from $(-\infty, +\infty)$ into a valid probability range $[0, 1]$, Logistic Regression applies the **Sigmoid (logistic) function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Properties of the Sigmoid function:
- When $z \to +\infty$, $\sigma(z) \to 1$
- When $z \to -\infty$, $\sigma(z) \to 0$
- When $z = 0$, $\sigma(z) = 0.5$

---

## 3. Decision Rule and Decision Threshold

The predicted probability $p = \sigma(z)$ is converted into a binary classification label $\hat{y}$ using a **decision threshold** $\tau$:

$$
\hat{y} = \begin{cases} 
1 & \text{if } P(Y=1|X) \ge \tau \\
0 & \text{if } P(Y=1|X) < \tau 
\end{cases}
$$

### Why is Threshold Selection Critical?
- By default, most machine learning libraries use $\tau = 0.5$.
- However, $\tau = 0.5$ is an arbitrary mathematical midpoint.
- Lowering $\tau$ (e.g. $\tau = 0.3$) makes the model more sensitive:
  - Detects more positive cases $\to$ **Higher Recall**
  - Generates more false alarms $\to$ **Lower Precision**
- Raising $\tau$ (e.g. $\tau = 0.7$) makes the model more conservative:
  - Produces fewer false alarms $\to$ **Higher Precision**
  - Misses positive cases $\to$ **Lower Recall**

This threshold flexibility is the core reason Logistic Regression is ideal for our investigation into **Metric Choice**.

---

## 4. Cost Function: Log-Loss (Binary Cross-Entropy)

Logistic Regression parameters are learned by minimizing the Binary Cross-Entropy loss over $N$ training samples:

$$
J(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

where $\hat{p}_i = \sigma(w^T x_i + b)$.

This penalizes confident wrong predictions heavily.

---

## 5. Relevance to Team 16's Research Question

Our investigation asks:
> **"How does the choice of evaluation metric change the conclusion about a model?"**

Because Logistic Regression generates calibrated continuous probabilities:
1. We train the model once.
2. We freeze model weights, test observations, and predicted probabilities.
3. We examine how different metrics (Accuracy, Precision, Recall, F1, ROC-AUC) describe the performance across decision thresholds.
4. This cleanly isolates the **effect of metric choice** from model variance.
