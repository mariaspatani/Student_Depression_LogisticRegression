# Team 16 — Metric Choice Investigation

## Investigating the Effect of Evaluation Metric Choice on Model Conclusions: Logistic Regression on the Student Depression Dataset

**Course:** 24SJPCCST503 — Machine Learning  
**Team:** 16  
**Institution:** St. Joseph's College of Engineering and Technology, Palai

---

## 1. Project Overview

This project investigates how the choice of evaluation metric changes the conclusion drawn about a machine learning model.

We use the Student Depression Dataset and train a Logistic Regression classifier to predict the dataset's binary `Depression` target. We evaluate the model using five metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

We also investigate how changing the classification threshold affects model performance.

The main objective is not simply to achieve high prediction accuracy. Instead, we investigate whether different evaluation metrics reveal different strengths and weaknesses of the same model.

## 2. Research Question

**How does the choice of evaluation metric change the conclusion about a model?**

This is the central question of our investigation.

## 3. Hypothesis

Different evaluation metrics will provide different interpretations of the same model's performance because each metric measures a different aspect of classification behaviour.

For example, a model may achieve relatively high accuracy but still make a substantial number of false-positive or false-negative predictions. Accuracy alone may not reveal the nature of these errors.

## 4. Project Objectives

The objectives of this project are:

1. To train a Logistic Regression classifier using the Student Depression Dataset.
2. To evaluate the model using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
3. To understand the role of false positives and false negatives using a confusion matrix.
4. To investigate the effect of changing the classification threshold.
5. To compare the conclusions obtained from different evaluation metrics.
6. To identify the limitations of the experiment and explain why metric selection depends on the intended application.

## 5. Dataset Description

**Dataset name:** Student Depression Dataset  
**Source:** [Kaggle — Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

The dataset contains student-related information covering demographic, academic, lifestyle, financial, and family-related factors.

The target variable is `Depression`, represented as a binary classification target.

### Dataset summary

| Property | Description |
|---|---|
| Original records | 27,901 |
| Original columns | 18 |
| Columns after removing ID | 17 |
| Target variable | `Depression` |
| Task | Binary classification |
| Training observations | 22,320 |
| Testing observations | 5,581 |
| Train-test split | 80:20 |
| Input features after encoding | 102 |

These figures describe the dataset and experimental setup documented in the original project report.

### Why did we remove the ID column?

The ID column identifies individual records rather than representing a meaningful student characteristic. We removed it before training to avoid using an irrelevant identifier as a predictive feature.

### Dataset limitations

- The dataset contains self-reported information.
- Its sampling procedure and representativeness are not fully documented in the project materials.
- The results may not generalize to all student populations.
- The dataset's `Depression` label should not be interpreted as a verified clinical diagnosis.

## 6. Methodology

The experiment follows these steps:

1. Load the dataset.
2. Inspect the data and identify missing values and duplicate records.
3. Remove the ID column.
4. Separate the target variable from the input features.
5. Preprocess numerical and categorical features.
6. Split the dataset into training and testing sets using stratification.
7. Train a Logistic Regression classifier.
8. Generate class predictions and predicted probabilities.
9. Calculate evaluation metrics.
10. Analyse the confusion matrix and ROC curve.
11. Change the classification threshold and compare the resulting metrics.
12. Interpret the results in relation to the research question.

### Experimental design

The main metric-comparison experiment evaluates the same trained model on the same test set. Each metric provides a different perspective on its performance.

The threshold experiment uses the same trained model and its predicted probabilities, but changes the cutoff used to convert probabilities into class labels.

## 7. Implementation

### Tools and libraries

The project uses Python and the following libraries:

- Pandas — dataset loading and manipulation
- NumPy — numerical operations
- Matplotlib — visualisation
- Seaborn — statistical visualisation
- Scikit-learn — preprocessing, Logistic Regression, model evaluation, and metrics

### Why Logistic Regression?

Logistic Regression is a supervised learning algorithm commonly used for binary classification. It estimates the probability that an observation belongs to a particular class.

We selected it as a baseline classifier because the main focus of the project is evaluation metric choice rather than developing a complex prediction model.

### What does the model produce?

The model produces predicted probabilities. These probabilities can be converted into class labels using a decision threshold.

For example, with a threshold of 0.5:

- Probability greater than or equal to 0.5 → positive prediction.
- Probability below 0.5 → negative prediction.

Changing this threshold can change the final class predictions even when the trained model remains unchanged.

## 8. Understanding the Confusion Matrix

A confusion matrix summarises the relationship between actual and predicted class labels.

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

### Definitions

**True Positive (TP):** The model correctly predicts a positive case.

**True Negative (TN):** The model correctly predicts a negative case.

**False Positive (FP):** The model predicts positive when the actual class is negative.

**False Negative (FN):** The model predicts negative when the actual class is positive.

### Confusion matrix reported in our original experiment

| | Predicted Positive | Predicted Negative |
|---|---:|---:|
| Actual Positive | 2,875 | 393 |
| Actual Negative | 473 | 1,840 |

The four values add up to 5,581 test observations.

The matrix helps explain why Precision and Recall differ: Precision is affected by false-positive predictions, whereas Recall is affected by false-negative predictions.

## 9. Evaluation Metrics

### 9.1 Accuracy

Accuracy measures the proportion of all predictions that are correct.

Formula:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Question answered:** How many predictions were correct overall?

Accuracy is easy to understand, but it does not describe the distribution of errors in sufficient detail.

### 9.2 Precision

Precision measures the proportion of positive predictions that are actually positive.

Formula:

Precision = TP / (TP + FP)

**Question answered:** When the model predicts positive, how often is it correct?

Precision is particularly relevant when false-positive predictions are costly.

### 9.3 Recall

Recall measures the proportion of actual positive cases that the model correctly identifies.

Formula:

Recall = TP / (TP + FN)

**Question answered:** Of all actual positive cases, how many did the model identify?

Recall is particularly relevant when missing positive cases is costly.

### 9.4 F1-score

F1-score is the harmonic mean of Precision and Recall.

Formula:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Question answered:** How well does the model balance Precision and Recall?

F1-score is useful when both types of performance matter. However, it does not include True Negatives directly in its formula.

### 9.5 ROC-AUC

ROC-AUC is the area under the Receiver Operating Characteristic curve.

The ROC curve plots:

- True Positive Rate (Recall) on the Y-axis.
- False Positive Rate on the X-axis.

ROC-AUC evaluates how well the model's scores distinguish between positive and negative classes across different thresholds.

**Question answered:** How well does the model distinguish between the two classes across thresholds?

An ROC-AUC closer to 1 generally indicates better discrimination. ROC-AUC is not another form of accuracy and should not be interpreted as one.

## 10. Results

The following are the metrics reported in the original project report.

| Metric | Score |
|---|---:|
| Accuracy | 84.48% |
| Precision | 85.87% |
| Recall | 87.97% |
| F1-score | 86.91% |
| ROC-AUC | 91.82% |

### Interpretation

**Accuracy — 84.48%**

The model correctly classified approximately 84.48% of the test observations.

**Precision — 85.87%**

Of the observations predicted as positive, approximately 85.87% actually belonged to the positive class.

**Recall — 87.97%**

The model identified approximately 87.97% of the actual positive cases.

**F1-score — 86.91%**

The F1-score provides a combined summary of Precision and Recall.

**ROC-AUC — 91.82%**

The model demonstrated good discrimination between the two classes across different thresholds.

These scores measure different properties. Their numerical values should not be interpreted as interchangeable measures of the same thing.

## 11. Threshold Analysis

A classification threshold determines how predicted probabilities are converted into class labels.

Our original report compared thresholds of 0.3, 0.5, and 0.7.

| Threshold | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.3 | 82.76% | 79.75% | 94.58% | 86.53% |
| 0.5 | 84.48% | 85.87% | 87.97% | 86.91% |
| 0.7 | 82.67% | 90.84% | 78.30% | 84.11% |

### Interpretation of the threshold graph

**Threshold = 0.3**

The model has high Recall (94.58%) but lower Precision (79.75%). A lower threshold classifies more observations as positive, helping identify more actual positive cases but potentially producing more false positives.

**Threshold = 0.5**

This is the default threshold used in the original experiment. It provides a different balance between Precision and Recall.

**Threshold = 0.7**

Precision increases to 90.84%, while Recall decreases to 78.30%. The model makes positive predictions more selectively but misses more actual positive cases.

### The most important observation

Accuracy is almost identical at thresholds 0.3 and 0.7:

- Threshold 0.3: Accuracy = 82.76%
- Threshold 0.7: Accuracy = 82.67%

However, Recall changes from 94.58% to 78.30%, while Precision increases from 79.75% to 90.84%.

**This is the strongest evidence supporting our research question.**

Two threshold settings can achieve almost identical accuracy while behaving very differently in terms of false positives and false negatives. Therefore, accuracy alone can hide important differences in model performance.

The F1-score also changes relatively little between thresholds 0.3 and 0.5, showing that a combined score may not fully communicate the trade-off between Precision and Recall.

ROC-AUC evaluates discrimination across thresholds and does not change simply because we select a different threshold for the same predicted probabilities.

## 12. Main Findings

The experiment supports the hypothesis within the conditions tested.

1. Different evaluation metrics describe different aspects of the same model's performance.
2. Accuracy alone does not fully describe how classification errors are distributed.
3. Precision and Recall respond differently to changes in the classification threshold.
4. F1-score provides a balance between Precision and Recall but does not replace them in every situation.
5. ROC-AUC provides information about discrimination across thresholds rather than accuracy at one particular threshold.
6. The appropriate metric depends on the purpose of the model and the relative consequences of false-positive and false-negative errors.

## 13. Which Metric Is Best?

There is no universally best evaluation metric.

| Objective | Potentially useful metric |
|---|---|
| Measure overall correctness | Accuracy |
| Reduce incorrect positive predictions | Precision |
| Identify more actual positive cases | Recall |
| Balance Precision and Recall | F1-score |
| Evaluate discrimination across thresholds | ROC-AUC |

For a hypothetical screening application, Recall may be important when missing a positive case is costly. Precision may be more important when unnecessary follow-up is costly.

However, these priorities depend on the real-world context. The Student Depression experiment is a methodological investigation, not a clinical diagnostic system.

## 14. Limitations

### 14.1 Single model

The main experiment uses Logistic Regression. Other algorithms may behave differently.

### 14.2 Single train-test split

The original reported metrics describe one particular split. They should not automatically be treated as universal estimates of model performance.

### 14.3 Dataset limitations

The dataset's sampling procedure and representativeness are not fully documented in the project materials.

### 14.4 Preprocessing leakage risk in the original experiment

The original report notes that some preprocessing was performed before the train-test split. This can allow information from the test set to influence preprocessing.

A stronger experimental pipeline fits imputation, encoding, and scaling using the training data only, then applies the fitted transformations to the test data.

### 14.5 Threshold selection

The three tested thresholds do not establish a universally optimal threshold. The best threshold depends on the application's requirements and error costs.

### 14.6 Clinical interpretation

The dataset's label should not be treated as a confirmed clinical diagnosis. The model's predictions do not replace professional mental-health assessment.

## 15. Conclusion

This project investigated how evaluation metric choice changes the conclusion about a Logistic Regression classifier trained on the Student Depression Dataset.

The original experiment reported Accuracy of 84.48%, Precision of 85.87%, Recall of 87.97%, F1-score of 86.91%, and ROC-AUC of 91.82%.

The threshold experiment provided further evidence: Accuracy was nearly identical at thresholds 0.3 and 0.7, but Recall decreased from 94.58% to 78.30%, while Precision increased from 79.75% to 90.84%.

These results demonstrate that a single evaluation metric cannot provide a complete description of model performance. Different metrics answer different questions, and their usefulness depends on the objective and the consequences of prediction errors.

The findings are limited to the dataset, model, and experimental setup used. Further validation across repeated splits and additional datasets or models would strengthen the reliability and generalizability of the conclusions.

## 16. Presentation Script

### Introduction

"Good morning, everyone. We are Team 16. Our project investigates how the choice of evaluation metric changes the conclusion about a machine learning model. We use the Student Depression Dataset and Logistic Regression to study this question."

### Objectives

"Our objective is to evaluate the same model using different metrics and investigate whether they reveal different aspects of its performance."

### Methodology

"We loaded and preprocessed the dataset, removed the ID column, encoded categorical features, and used a stratified train-test split. We trained a Logistic Regression classifier and evaluated it using Accuracy, Precision, Recall, F1-score, and ROC-AUC. We also conducted a threshold analysis."

### Results

"Our original experiment reported an accuracy of 84.48%, precision of 85.87%, recall of 87.97%, F1-score of 86.91%, and ROC-AUC of 91.82%. These metrics provide different perspectives on the model's performance."

### Graph interpretation

"The threshold experiment is particularly important. Accuracy was nearly identical at thresholds 0.3 and 0.7, but Recall decreased from 94.58% to 78.30%, while Precision increased from 79.75% to 90.84%. This demonstrates that similar accuracy values can hide very different classification behaviours."

### Conclusion

"Our investigation supports the idea that evaluation metric choice affects how we interpret model performance. No single metric is universally best. The appropriate metric depends on the objective and the relative costs of different prediction errors. Thank you."

## 17. Viva Questions and Answers

### Q1. What is the actual topic of your project?

Our topic is Metric Choice. We investigate how different evaluation metrics change the interpretation of the same model's performance.

### Q2. Why did you choose the Student Depression Dataset?

It provides a binary classification task that allows us to investigate how different evaluation metrics describe model performance. Its results must be interpreted with the dataset's limitations in mind.

### Q3. Why did you use Logistic Regression?

We used Logistic Regression as a baseline binary classifier so that the investigation could focus on evaluation metrics rather than model complexity.

### Q4. Why is accuracy not always enough?

Accuracy measures overall correctness but does not separately explain false positives and false negatives. Precision, Recall, and the confusion matrix provide additional information.

### Q5. What is the difference between Precision and Recall?

Precision measures the reliability of positive predictions. Recall measures the proportion of actual positive cases identified by the model.

### Q6. What is F1-score?

F1-score is the harmonic mean of Precision and Recall. It provides a combined summary of these two metrics.

### Q7. What is ROC-AUC?

ROC-AUC measures the model's ability to distinguish between positive and negative classes across different thresholds.

### Q8. What is a classification threshold?

It is the probability cutoff used to convert predicted probabilities into class labels.

### Q9. What is the most important finding?

Accuracy was nearly identical at thresholds 0.3 and 0.7, but Precision and Recall changed substantially. This demonstrates that accuracy alone can hide important differences.

### Q10. Which metric is the best?

There is no universally best metric. The appropriate choice depends on what the model is intended to achieve and the consequences of prediction errors.

### Q11. What are the limitations of your study?

The original experiment uses one dataset, one model, and one train-test split. The dataset's sampling procedure is not fully documented, and preprocessing leakage risk was identified in the original pipeline.

### Q12. Can this model diagnose depression?

No. This is a machine-learning classification experiment using a dataset label. Its predictions should not be treated as clinical diagnoses.

### Q13. What is the difference between a model and an evaluation metric?

A model learns patterns from training data and generates predictions. An evaluation metric measures a particular aspect of those predictions.

### Q14. What is the purpose of the confusion matrix?

It summarises true positives, true negatives, false positives, and false negatives. These values help explain the errors behind evaluation scores.

### Q15. What would you do as future work?

We would repeat the evaluation across multiple stratified splits, use a leakage-safe preprocessing pipeline, investigate additional models or datasets, and quantify uncertainty in the results.

## 18. References

1. Hopesb, "Student Depression Dataset," Kaggle. Available at: https://www.kaggle.com/datasets/hopesb/student-depression-dataset
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
3. Scikit-learn Developers, "Scikit-learn Documentation." Available at: https://scikit-learn.org/stable/
4. M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Information Processing & Management*, vol. 45, no. 4, pp. 427–437, 2009.

---

**Study tip:** Understand the research question, the difference between Precision and Recall, the threshold-comparison table, and the final conclusion. These are the most important concepts to explain confidently during the presentation.
