# Student Depression Logistic Regression

## Team 16: Metric Choice

This project investigates student depression using logistic regression. The
objective is to build a reproducible classification workflow and evaluate how
well selected student, lifestyle, and academic indicators identify depression
outcomes. Team 16 focuses on choosing and interpreting appropriate evaluation
metrics rather than relying on accuracy alone, especially when class balance
and the relative cost of errors matter.

## Dataset

The project uses the **Student Depression Dataset**, stored at
`dataset/Student_Depression_Dataset.csv`. It contains student-level
demographic, academic, lifestyle, and mental-health-related attributes, along
with a target indicator for depression. The analysis notebook documents data
loading, cleaning, exploratory analysis, model training, metric selection, and
prediction review.

## Project Structure

```text
Student_Depression_LogisticRegression/
├── dataset/
│   └── Student_Depression_Dataset.csv
├── notebooks/
│   └── Team16_Student_Depression.ipynb
├── outputs/
│   ├── graphs/
│   ├── confusion_matrix/
│   ├── metrics/
│   └── predictions/
├── report/
│   ├── Investigation_Report.docx
│   └── figures/
├── presentation/
│   └── Team16_Presentation.pptx
├── requirements.txt
├── README.md
└── .gitignore
```

## Python Setup

Create and activate a virtual environment from the project root:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch Jupyter Notebook with:

```bash
jupyter notebook
```

Open `notebooks/Team16_Student_Depression.ipynb` and run the cells in order.

## Results

_Results will be added after the analysis is completed, including the chosen
metrics, model performance, confusion matrix findings, and key conclusions._

## Experiments

_Experiment details will be added here, including preprocessing decisions,
feature selection, train-test split configuration, model variants, and metric
comparisons._