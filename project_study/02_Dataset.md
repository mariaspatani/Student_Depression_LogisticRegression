# 02 — Dataset: Student Depression

## 1. Dataset Used

For our project, we use the **Student Depression Dataset** as the application/domain for our investigation.

Our assigned topic is **Metric Choice**.

The main purpose of the project is not simply to build a model that predicts student depression. Instead, we use this dataset to investigate:

> **How does the choice of evaluation metric change the conclusion about a model?**

The dataset provides a binary classification problem, which allows us to evaluate the model using different classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 2. Why Did We Choose This Dataset?

The Student Depression Dataset is suitable for our investigation because the target variable represents whether a student belongs to a depression-related class.

This allows us to formulate the problem as a **binary classification problem**.

Our project focuses on comparing different evaluation metrics for the **same machine learning model**.

Therefore, the dataset is used as an application through which we can study how different metrics can produce different interpretations of model performance.

---

## 3. Prediction Task

The machine learning model will use student-related features to predict the target variable representing depression status.

### Input

The input consists of different student-related features available in the dataset.

Examples of feature categories include:

- Academic pressure
- Work/study-related pressure
- Financial stress
- Sleep-related information
- Dietary habits
- Study satisfaction
- Social relationships
- Other student-related characteristics

### Target

The target variable represents the depression status of the student.

The target will be treated as a binary classification target.

Conceptually:

```text
0 → Negative class
1 → Positive class
```

The exact encoding and values will be verified from the actual dataset before model training.

---

# 4. Classification Problem

Our project is a **binary classification problem**.

Binary classification means that the model predicts one of two possible classes.

For our project:

```text
Class 0 → No Depression
Class 1 → Depression
```

The machine learning model receives the student's features and predicts one of these two classes.

The basic workflow is:

```text
Student Features
       ↓
Machine Learning Model
       ↓
Predicted Class
       ↓
0 or 1
```

---

# 5. Features and Target

In machine learning, the dataset is divided into:

### Features — X

Features are the input variables used by the model to make predictions.

Conceptually:

```python
X = df.drop("Depression", axis=1)
```

### Target — y

The target is the variable that we want the model to predict.

```python
y = df["Depression"]
```

Therefore:

```text
X → Input features
y → Target variable
```

The exact target column name will be confirmed from the actual dataset before implementing the final code.

---

# 6. Dataset Inspection

Before building the machine learning model, we need to understand the dataset.

We should inspect:

- Number of rows
- Number of columns
- Column names
- Data types
- Missing values
- Duplicate records
- Target distribution
- Numerical features
- Categorical features

These checks help us understand the structure and quality of the data before preprocessing.

---

## 6.1 Number of Rows and Columns

We can use:

```python
df.shape
```

This returns:

```text
(number of rows, number of columns)
```

For example:

```python
print(df.shape)
```

The exact values will be recorded after loading our actual dataset.

---

## 6.2 Column Names

To see all the columns:

```python
df.columns
```

or:

```python
print(df.columns)
```

This helps us identify:

- Input features
- Target variable
- ID columns
- Columns that may need to be removed

---

## 6.3 Data Types

We can check the data types using:

```python
df.dtypes
```

This helps us identify whether a column contains:

- Integer values
- Floating-point values
- Strings/categorical values
- Other data types

This is important because machine learning models generally require numerical input.
The dataset inspection stage helps us understand the structure, quality, and characteristics of the data before preprocessing. It also helps identify potential issues that could affect model evaluation.
---

# 7. Missing Values

Missing values are values that are not available for some observations.

We can check missing values using:

```python
df.isnull().sum()
```

This shows the number of missing values in each column.

If missing values are present, we need to decide how to handle them.

Possible approaches include:

- Removing rows
- Filling numerical values using an appropriate statistic
- Filling categorical values using an appropriate category
- Using other imputation methods

The method should depend on the actual amount and type of missing data found in the dataset.

---

# 8. Duplicate Records

Duplicate records can affect the analysis and model training.

We can check the number of duplicate rows using:

```python
df.duplicated().sum()
```

If duplicates are found, we need to investigate them before deciding whether they should be removed.

---

# 9. Target Distribution

One of the most important checks for our project is the distribution of the target classes.

We can use:

```python
df["Depression"].value_counts()
```

To calculate percentages:

```python
df["Depression"].value_counts(normalize=True) * 100
```

This tells us how many observations belong to each class.

For example:

```text
Class 0 → XXXXX%
Class 1 → XXXXX%
```

The actual values will be obtained from our dataset.

---

# 10. Why Is Class Distribution Important?

Class distribution is particularly important for our project because our research is about **evaluation metrics**.

If one class is much larger than the other, the dataset is considered **imbalanced**.

For example:

```text
Class 0 → 90%
Class 1 → 10%
```

A model could achieve high accuracy by predicting the majority class frequently.

Therefore, accuracy alone may not provide a complete picture of model performance.

This makes other metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC

important for our investigation.

We should calculate the actual class distribution rather than assuming that the dataset is balanced or imbalanced.

Visualizing the target distribution using a bar chart makes it easier to compare the number of observations in each class. This helps us understand the dataset before selecting appropriate evaluation metrics.

---

# 11. ID Column

If the dataset contains an ID column, it generally serves as an identifier for each record rather than a meaningful characteristic of the student.

For example:

```text
id
```

An identifier should not normally be used as a predictive feature because its numerical value does not represent a meaningful student characteristic.

If the dataset contains such a column, it can be removed before model training:

```python
df = df.drop("id", axis=1)
```

The exact columns to remove will be confirmed after inspecting the actual dataset.

---

# 12. Categorical Features

Some features may contain categories instead of numerical values.

Examples may include:

```text
Gender
City
Dietary Habits
Degree
Sleep Duration
```

Machine learning models such as Logistic Regression require numerical input.

Therefore, categorical variables need to be converted into numerical representations.

One common method is:

> **One-Hot Encoding**

For example, suppose a feature contains:

```text
Dietary Habit

Healthy
Moderate
Unhealthy
```

One-hot encoding can represent these categories using separate binary columns.

Conceptually:

```text
Healthy    → 1 0 0
Moderate   → 0 1 0
Unhealthy  → 0 0 1
```

The exact encoding method will be selected after inspecting the actual dataset.

---

# 13. Numerical Features

Some features are numerical and can be used as numerical inputs after appropriate preprocessing.

Examples may include:

- Age
- Academic pressure
- Financial stress
- Study satisfaction
- Work pressure

The exact numerical columns will be identified during dataset exploration.

---

# 14. Feature Scaling

Some numerical features may have different ranges.

For example:

```text
Feature A → 1 to 5
Feature B → 1 to 100
```

A model such as Logistic Regression can benefit from scaling numerical features so that they are placed on comparable scales.

A common technique is:

> **Standardization**

Standardization transforms a numerical feature approximately to a mean of 0 and a standard deviation of 1.

One commonly used implementation is:

```python
from sklearn.preprocessing import StandardScaler
```

Scaling should be performed correctly to avoid data leakage.

The scaler should be fitted using the training data and then applied to the test data.

---

# 15. Train-Test Split

After preprocessing, the dataset is divided into training and testing data.

### Training Data

The training data is used to train the machine learning model.

### Testing Data

The testing data is used to evaluate the model on previously unseen data.

A common split is:

```text
80% → Training
20% → Testing
```

In Python:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

# 16. Why Do We Use `random_state`?

We use:

```python
random_state=42
```

to make the train-test split reproducible.

Without a fixed random state, the data may be split differently each time the code is executed.

With a fixed random state, we can obtain the same split again.

The value `42` is simply a commonly used fixed seed. It does not have any special mathematical meaning.

---

# 17. Why Do We Use `stratify=y`?

We use:

```python
stratify=y
```

to preserve approximately the same class distribution in both the training and testing sets.

For example, suppose the complete dataset contains:

```text
Class 0 → 80%
Class 1 → 20%
```

A stratified split aims to maintain approximately:

```text
Training → 80% Class 0, 20% Class 1
Testing  → 80% Class 0, 20% Class 1
```

This is especially useful when the classes are not equally distributed.

---

# 18. Data Leakage

Data leakage occurs when information from outside the training data is unintentionally used during model training.

For example, if we calculate preprocessing parameters using the entire dataset before splitting it into training and testing data, information from the test set can influence the training process.

This can produce overly optimistic evaluation results.

Therefore, preprocessing should be performed carefully.

The general idea is:

```text
Dataset
   ↓
Train-Test Split
   ↓
Training Data → Fit preprocessing
Testing Data  → Apply preprocessing
```

This helps keep the test data unseen during training.

To maintain a fair evaluation, preprocessing parameters should be learned from the training data only. The test data should be used to evaluate the trained model without influencing the training process.

---

# 19. Dataset Limitations

A dataset should not automatically be considered a perfect representation of all students.

Possible limitations include:

- The dataset may not represent every student population.
- The data source may introduce sampling bias.
- Some information may be self-reported.
- Some variables may contain missing or inconsistent information.
- The dataset may not represent all geographical or demographic groups.
- The dataset may not capture every factor associated with student depression.

These limitations should be considered when interpreting the results.

---

# 20. Prediction vs Causation

Our project is about **machine learning prediction/classification**.

If the model finds that a particular feature is useful for prediction, this does not automatically mean that the feature causes depression.

For example, if academic pressure is useful for predicting the target, we should not conclude:

> "Academic pressure causes depression."

Instead, we should use careful language such as:

> "Academic pressure is a feature used by the model for prediction."

or:

> "The feature is associated with the model's prediction."

Our project is focused on **evaluation metrics**, not on establishing causal relationships.

---

# 21. Why This Dataset Is Suitable for Our Investigation

The dataset is suitable for our project because:

1. It provides a binary classification task.
2. A classification model can be trained using the available features.
3. The model produces positive and negative predictions.
4. These predictions can be evaluated using multiple classification metrics.
5. Different metrics can emphasize different aspects of model performance.
6. This allows us to investigate our research question.

Our focus is therefore not:

> "How accurate is our depression prediction model?"

Our focus is:

> **"How does the choice of evaluation metric change our conclusion about the model?"**

---

# 22. Dataset → Model Pipeline

Our overall workflow is:

```text
Student Depression Dataset
          ↓
       Inspect
          ↓
   Clean the data
          ↓
 Handle missing values
          ↓
 Identify numerical/categorical features
          ↓
 Remove unnecessary columns
          ↓
   Separate X and y
          ↓
   Train-Test Split
          ↓
    Preprocessing
          ↓
      ML Model
          ↓
     Predictions
          ↓
 Evaluation Metrics
```

---

# 23. Connection to Our Research Question

The dataset is only the application through which we conduct our investigation.

After preprocessing, we will train a classification model and generate predictions.

Then we will evaluate the **same model/predictions** using multiple evaluation metrics.

For a fair comparison, all evaluation metrics should be calculated using the same test dataset and the same model predictions. This ensures that differences in interpretation arise from the metrics being used rather than changes in the model or test data.
The planned metrics are:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. ROC-AUC

The results will then be compared to determine whether the metrics lead to different conclusions.

Conceptually:

```text
Student Depression Dataset
          ↓
   Classification Model
          ↓
       Predictions
          ↓
 ┌────────┼─────────┬────────┬─────────┐
 ↓        ↓         ↓        ↓         ↓
Accuracy Precision Recall   F1      ROC-AUC
          ↓
   Compare Results
          ↓
 Analyze Differences
          ↓
 Final Conclusion
```

---

# 24. Information We Need to Record From the Actual Dataset

Before moving to the model stage, we need to record the following information from the actual dataset:

- Number of rows
- Number of columns
- Column names
- Target column
- Numerical features
- Categorical features
- Missing values
- Duplicate records
- Target class distribution
- Columns removed during preprocessing
- Preprocessing methods used

These values should be obtained directly from the dataset and should not be guessed.

---

# 25. Summary

The Student Depression Dataset provides the data for our binary classification investigation.

The important steps before modelling are:

```text
1. Load the dataset
2. Inspect the dataset
3. Identify the target
4. Check missing values
5. Check duplicate records
6. Check class distribution
7. Identify numerical and categorical features
8. Remove unnecessary columns
9. Split the data into training and testing sets
10. Apply appropriate preprocessing
```

After these steps, we can train our machine learning model.

The main purpose of the project remains:

> **To investigate how different evaluation metrics can change the conclusion about the performance of the same machine learning model.**
