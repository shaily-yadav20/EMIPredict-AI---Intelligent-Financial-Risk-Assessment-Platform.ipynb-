# 💰EMI Predict AI-Intelligent Financial Risk Assessment Platform

**EMIPredict AI** is a Machine Learning-based financial risk assessment system that predicts whether a customer is eligible for an EMI/loan and estimates the **maximum safe monthly EMI** they can afford.
LIVE DEMO: https://charm-nexus-hub.lovable.app

The project uses customer financial, employment, credit, expense, and loan information to perform both:

* **Classification** → EMI Eligibility Prediction
* **Regression** → Maximum Monthly EMI Prediction

The final system is integrated with a **Gradio web interface**, allowing users to enter their financial details and receive an instant AI-based assessment.

---

## 📌 Project Overview

When a customer applies for an EMI or loan, financial institutions need to evaluate whether the customer can comfortably repay the amount.

Traditional assessment may require manual verification of:

* Monthly salary
* Existing EMI
* Monthly expenses
* Credit score
* Bank balance
* Emergency fund
* Employment information
* Requested loan amount
* Requested tenure

This project automates this process using Machine Learning.

The system provides:

1. EMI eligibility prediction
2. Maximum safe monthly EMI prediction
3. Financial risk score
4. Risk-level classification
5. Loan/EMI recommendation
6. Interactive Gradio interface
7. MLflow experiment tracking

---

# 🎯 Objectives

The main objectives of the project are:

* Analyze customer financial data.
* Perform data validation and preprocessing.
* Perform Exploratory Data Analysis (EDA).
* Create meaningful financial features.
* Predict EMI eligibility.
* Estimate maximum safe monthly EMI.
* Compare multiple Machine Learning algorithms.
* Select the best-performing models.
* Track experiments using MLflow.
* Build an interactive Gradio application.
* Provide an easy-to-understand financial risk assessment.

---

# 🏗️ Project Architecture

```text
                 ┌──────────────────────────┐
                 │      Customer Dataset    │
                 │      CSV File            │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │    Data Validation       │
                 │    Missing Values        │
                 │    Duplicates            │
                 │    Data Types             │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │          EDA             │
                 │  Salary                  │
                 │  Credit Score            │
                 │  Expenses                │
                 │  EMI                     │
                 │  Loan Amount             │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Feature Engineering    │
                 │  Debt-to-Income Ratio    │
                 │  Expense-to-Income       │
                 │  Disposable Income       │
                 │  Financial Burden         │
                 │  Risk Features            │
                 └─────────────┬────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
      ┌───────────────────┐         ┌────────────────────┐
      │   Classification  │         │     Regression     │
      │                   │         │                    │
      │ Logistic          │         │ Linear Regression  │
      │ Random Forest     │         │ Random Forest      │
      │ XGBoost           │         │ XGBoost            │
      └─────────┬─────────┘         └──────────┬─────────┘
                │                              │
                ▼                              ▼
      ┌───────────────────┐         ┌────────────────────┐
      │ EMI Eligibility   │         │ Maximum Safe EMI   │
      │ Prediction        │         │ Prediction         │
      └─────────┬─────────┘         └──────────┬─────────┘
                │                              │
                └──────────────┬───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   MLflow Tracking    │
                    │   Experiments        │
                    │   Metrics             │
                    │   Models              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Gradio Web App      │
                    │                      │
                    │ Risk Score            │
                    │ Eligibility           │
                    │ Maximum EMI           │
                    │ Recommendation        │
                    └──────────────────────┘
```

---

# 📊 Dataset

The project uses:

```text
emi_prediction_dataset_cleaned.csv
```

The dataset contains customer information related to:

* Personal details
* Employment
* Income
* Household expenses
* Existing loans
* Credit score
* Bank balance
* Emergency fund
* EMI scenario
* Requested loan amount
* Requested tenure

### Target Variables

### 1. `emi_eligibility`

This is the **classification target**.

It represents the customer's EMI eligibility category.

Possible classes:

```text
Eligible
High_Risk
Not_Eligible
```

### 2. `max_monthly_emi`

This is the **regression target**.

It represents the estimated maximum monthly EMI that the customer can safely afford.

---

# 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming               |
| Pandas       | Data manipulation         |
| NumPy        | Numerical operations      |
| Matplotlib   | Visualization             |
| Plotly       | Interactive visualization |
| Scikit-learn | Machine Learning          |
| XGBoost      | Gradient Boosting         |
| MLflow       | Experiment tracking       |
| Gradio       | Web application           |
| Joblib       | Model saving              |
| Google Colab | Development environment   |
| GitHub       | Version control           |

---

# 🚀 How I Created This Project

## Step 1 — Open Google Colab

I created the project using Google Colab because it provides a cloud-based Python environment and makes it easy to develop and test Machine Learning models.

Create a new notebook:

```text
Google Colab
      ↓
New Notebook
      ↓
EMIPredict_AI.ipynb
```

---

# Step 2 — Keep the Dataset in Colab

The cleaned dataset was placed in the Colab environment.

Dataset name:

```text
emi_prediction_dataset_cleaned.csv
```

I used the direct Colab path instead of creating an upload interface.

```python
DATA_PATH = "/content/emi_prediction_dataset_cleaned.csv"
```

---

# Step 3 — Load the Dataset

I used Pandas to read the CSV file.

```python
import pandas as pd

df = pd.read_csv(DATA_PATH)

print(df.shape)
display(df.head())
```

This allowed me to inspect the dataset before building the Machine Learning pipeline.

---

# Step 4 — Validate the Dataset

Before training the models, I checked:

* Number of rows
* Number of columns
* Column names
* Missing values
* Duplicate records
* Data types
* Statistical information

Example:

```python
print(df.shape)

print(df.isnull().sum())

print(df.duplicated().sum())

print(df.dtypes)
```

This step helped ensure that the dataset was suitable for Machine Learning.

---

# Step 5 — Exploratory Data Analysis

I performed EDA to understand the financial patterns in the dataset.

The analysis included:

* EMI eligibility distribution
* Credit score distribution
* Monthly salary distribution
* Salary vs current EMI
* Salary vs maximum EMI
* Credit score vs maximum EMI
* EMI scenario distribution
* Requested loan amount
* Requested tenure
* Maximum EMI by scenario
* Bank balance
* Emergency fund
* Dependents
* Employment type
* Feature correlation

Interactive Plotly visualizations were used to make the analysis easier to understand.

Example:

```python
fig = px.histogram(
    df,
    x="credit_score",
    color="emi_eligibility",
    title="Credit Score Distribution"
)

fig.show()
```

---

# Step 6 — Feature Engineering

Feature engineering was performed to create additional financial indicators from existing columns.

### Total Monthly Expenses

```text
Rent
+ School Fees
+ College Fees
+ Travel Expenses
+ Groceries & Utilities
+ Other Expenses
```

### Total Financial Burden

```text
Current EMI + Total Monthly Expenses
```

### Debt-to-Income Ratio

```text
Current EMI / Monthly Salary
```

### Expense-to-Income Ratio

```text
Total Expenses / Monthly Salary
```

### Total Burden Ratio

```text
Total Financial Burden / Monthly Salary
```

### Disposable Income

```text
Monthly Salary - Total Financial Burden
```

Additional features include:

```text
Emergency Fund Ratio
Bank Balance Ratio
Loan-to-Annual-Income
Employment Stability
Credit Risk Score
Dependent Ratio
```

These features help the models understand the customer's financial condition more effectively.

---

# Step 7 — Define Machine Learning Targets

The project uses two different Machine Learning tasks.

### Classification

```python
CLASS_TARGET = "emi_eligibility"
```

The model predicts:

```text
Eligible
High_Risk
Not_Eligible
```

### Regression

```python
REG_TARGET = "max_monthly_emi"
```

The model predicts a continuous EMI amount.

---

# Step 8 — Split the Dataset

The dataset was divided into:

```text
70% → Training
15% → Validation
15% → Testing
```

The training dataset is used to train the models.

The validation dataset is used during development.

The testing dataset is used to evaluate final model performance.

---

# Step 9 — Data Preprocessing

The Machine Learning pipeline handles both numerical and categorical features.

### Numerical Features

Numerical missing values are handled using:

```text
Median Imputation
```

Then numerical features are standardized using:

```text
StandardScaler
```

### Categorical Features

Categorical missing values are handled using:

```text
Most Frequent Value
```

Categorical variables are converted into numerical form using:

```text
OneHotEncoder
```

A Scikit-learn `ColumnTransformer` was used to combine both preprocessing pipelines.

---

# Step 10 — Classification Models

Three classification algorithms were trained.

## 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

It predicts the customer's EMI eligibility class.

```text
Input Financial Data
        ↓
Logistic Regression
        ↓
Eligibility Class
```

---

## 2. Random Forest Classifier

Random Forest uses multiple decision trees to make predictions.

```text
Financial Features
       ↓
Multiple Decision Trees
       ↓
Voting
       ↓
Final Eligibility
```

It can capture nonlinear relationships between financial variables.

---

## 3. XGBoost Classifier

XGBoost is a gradient boosting algorithm.

It builds trees sequentially and tries to improve errors made by previous trees.

```text
Data
 ↓
Tree 1
 ↓
Correct Errors
 ↓
Tree 2
 ↓
Correct Errors
 ↓
More Trees
 ↓
Final Prediction
```

---

# Step 11 — Classification Evaluation

The classification models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

Example:

```python
accuracy_score(
    y_test,
    predictions
)
```

The model with the strongest overall performance was selected as the final classification model.

---

# Step 12 — Regression Models

Three regression algorithms were trained.

## 1. Linear Regression

Used as a baseline model for predicting maximum monthly EMI.

```text
Financial Features
       ↓
Linear Regression
       ↓
Maximum Safe EMI
```

---

## 2. Random Forest Regressor

Random Forest Regressor combines predictions from multiple decision trees.

It can model nonlinear financial relationships.

---

## 3. XGBoost Regressor

XGBoost Regressor was used to predict the continuous:

```text
max_monthly_emi
```

value.

---

# Step 13 — Regression Evaluation

Regression models were evaluated using:

* RMSE
* MAE
* R²
* MAPE

### RMSE

Measures the average magnitude of prediction errors while giving larger errors more weight.

### MAE

Measures the average absolute difference between actual and predicted EMI.

### R²

Measures how well the model explains variation in the target.

### MAPE

Measures prediction error as a percentage.

The model with the lowest RMSE and strongest overall performance was selected.

---

# Step 14 — Compare Models

The results of all models were stored in DataFrames.

### Classification

```text
Model
Accuracy
Precision
Recall
F1 Score
ROC AUC
```

### Regression

```text
Model
RMSE
MAE
R²
MAPE
```

This makes it easy to compare the algorithms and select the best models.

---

# Step 15 — Confusion Matrix

A confusion matrix was generated for the best classification model.

It shows:

```text
Actual Class
      vs
Predicted Class
```

This helps identify where the model correctly or incorrectly predicts:

```text
Eligible
High_Risk
Not_Eligible
```

---

# Step 16 — Actual vs Predicted EMI

For regression, actual and predicted EMI values were compared using a scatter plot.

```text
Actual EMI
    vs
Predicted EMI
```

A prediction closer to the diagonal indicates better model performance.

---

# Step 17 — MLflow Experiment Tracking

MLflow was integrated to track Machine Learning experiments.

MLflow records:

* Model name
* Parameters
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* RMSE
* MAE
* R²
* MAPE
* Trained models

The project creates separate experiments for:

```text
EMIPredict_AI_Classification
```

and

```text
EMIPredict_AI_Regression
```

This makes model comparison and experiment management easier.

---

# Step 18 — Save the Best Models

After comparing all models, the best models were saved using Joblib.

```text
models/
│
├── best_classification_model.pkl
├── best_regression_model.pkl
└── label_encoder.pkl
```

This allows the trained models to be reused without retraining them every time.

---

# Step 19 — Create Financial Risk Score

A rule-based financial risk score was also created.

The score considers factors such as:

* Credit score
* EMI burden
* Expense burden
* Emergency fund

The final score is represented from:

```text
0 → 100
```

Risk levels:

```text
75–100 → Low Risk
50–74  → Medium Risk
0–49   → High Risk
```

This score is displayed along with the Machine Learning prediction.

---

# Step 20 — Build Gradio Application

After training the models, I created an interactive Gradio web application.

The user enters:

### Personal Information

* Age
* Gender
* Marital Status
* Education

### Employment Information

* Employment Type
* Years of Employment
* Company Type

### Financial Information

* Monthly Salary
* Monthly Rent
* School Fees
* College Fees
* Travel Expenses
* Groceries & Utilities
* Other Expenses
* Existing Loans
* Current EMI

### Financial Strength

* Credit Score
* Bank Balance
* Emergency Fund

### Loan Information

* EMI Scenario
* Requested Amount
* Requested Tenure

---

# Step 21 — Generate Prediction

After clicking:

```text
🔍 Assess Financial Risk
```

the application sends the entered information to the trained models.

The application returns:

```text
EMI Eligibility
        ↓
Maximum Safe Monthly EMI
        ↓
Financial Risk Score
        ↓
Risk Level
        ↓
Recommendation
```

Example output:

```text
EMI Eligibility:
Eligible

Maximum Safe EMI:
₹18,500

Risk Score:
82/100

Risk Level:
Low Risk

Recommendation:
The customer appears financially suitable
for the requested EMI.
```

---

# Step 22 — Gradio Dashboard

The application contains multiple tabs:

### 🏦 Financial Risk Assessment

Used to enter customer information and obtain predictions.

### 📊 Model Performance

Displays classification and regression model results.

### 📁 Dataset Explorer

Allows users to inspect sample dataset records.

### ℹ️ About Project

Provides information about the project and technologies used.

---

# 📁 Project Structure

```text
EMIPredict_AI/
│
├── models/
│   ├── best_classification_model.pkl
│   ├── best_regression_model.pkl
│   └── label_encoder.pkl
│
├── reports/
│   ├── classification_results.csv
│   └── regression_results.csv
│
├── mlruns/
│   └── MLflow experiments
│
├── emi_prediction_dataset_cleaned.csv
│
└── EMIPredict_AI.ipynb
```

---

# 🔄 Complete Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Validation
   ↓
Data Quality Check
   ↓
EDA
   ↓
Feature Engineering
   ↓
Target Definition
   ↓
Train / Validation / Test Split
   ↓
Preprocessing
   ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Classification          Regression
 │                       │
 ▼                       ▼
Logistic Regression     Linear Regression
Random Forest           Random Forest
XGBoost                 XGBoost
 │                       │
 └───────────┬───────────┘
             ▼
      Model Evaluation
             ↓
       Model Comparison
             ↓
       Best Model Selection
             ↓
       MLflow Tracking
             ↓
        Save Models
             ↓
       Risk Score Engine
             ↓
        Gradio Interface
             ↓
      Final EMI Assessment
```

---

# 📈 Key Features

✅ EMI eligibility prediction
✅ Maximum monthly EMI prediction
✅ Financial risk scoring
✅ Credit score analysis
✅ Expense analysis
✅ Debt-to-income analysis
✅ Feature engineering
✅ Multiple classification models
✅ Multiple regression models
✅ Model comparison
✅ Confusion matrix
✅ Actual vs predicted analysis
✅ MLflow experiment tracking
✅ Model persistence
✅ Interactive Gradio UI
✅ Dataset explorer

---

# 🧠 Machine Learning Models

| Task           | Algorithm               |
| -------------- | ----------------------- |
| Classification | Logistic Regression     |
| Classification | Random Forest           |
| Classification | XGBoost                 |
| Regression     | Linear Regression       |
| Regression     | Random Forest Regressor |
| Regression     | XGBoost Regressor       |

---

# 📊 Evaluation Metrics

## Classification

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
```

## Regression

```text
RMSE
MAE
R²
MAPE
```

---

# 💻 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EMIPredict-AI.git
```

```bash
cd EMIPredict-AI
```

---

## 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost mlflow gradio plotly joblib openpyxl
```

---

## 3. Open the Notebook

Open:

```text
EMIPredict_AI.ipynb
```

using:

```text
Google Colab
```

or:

```text
Jupyter Notebook
```

---

## 4. Add the Dataset

Place:

```text
emi_prediction_dataset_cleaned.csv
```

inside the project/Colab environment.

The notebook uses:

```python
DATA_PATH = "/content/emi_prediction_dataset_cleaned.csv"
```

---

## 5. Run the Notebook

Run the cells in order:

```text
1. Dataset Path
2. Imports
3. Load Dataset
4. Validation
5. EDA
6. Feature Engineering
7. Preprocessing
8. Classification
9. Regression
10. Evaluation
11. MLflow
12. Save Models
13. Gradio
```

---

## 6. Launch Gradio

The final application is launched using:

```python
demo.launch(
    share=True,
    debug=True
)
```

Colab generates a public Gradio link that can be opened in a browser.

---

# 📌 Example Use Case

Suppose a customer has:

```text
Monthly Salary       = ₹60,000
Credit Score         = 720
Current EMI          = ₹5,000
Monthly Rent         = ₹10,000
Bank Balance         = ₹1,00,000
Emergency Fund       = ₹1,50,000
Requested Amount     = ₹3,00,000
Requested Tenure     = 36 months
```

The application processes these values and generates:

```text
EMI Eligibility
Maximum Safe EMI
Risk Score
Risk Level
Recommendation
```

This provides a quick AI-assisted financial assessment.

---

# ⚠️ Important Disclaimer

This project is developed for **educational and Machine Learning demonstration purposes**.

The predictions should not be considered a real banking, lending, credit, or financial decision.

Actual loan approval should involve additional verification, financial regulations, customer consent, credit bureau information, fraud checks, and human/financial-institution review.

---

# 🔮 Future Improvements

The project can be further improved by adding:

* Real-time bank transaction analysis
* Credit bureau integration
* Explainable AI using SHAP
* LIME-based prediction explanations
* Automated model retraining
* FastAPI backend
* Cloud deployment
* Docker containerization
* User authentication
* Database integration
* Real-time monitoring
* Model drift detection
* Advanced risk scoring
* PDF financial assessment reports

---

# 👩‍💻 Project Development Summary

This project was developed as an end-to-end Machine Learning solution.

The development process started with a cleaned financial dataset. I first validated and explored the data, then created financial features such as debt-to-income ratio, expense-to-income ratio, disposable income, and financial burden.

I then developed separate Machine Learning pipelines for classification and regression. Three classification algorithms and three regression algorithms were trained and evaluated. Their results were compared using appropriate evaluation metrics, and the best-performing models were saved.

MLflow was used to track experiments, parameters, metrics, and models. Finally, the trained models were integrated into a Gradio application where users can enter their financial information and receive EMI eligibility, maximum safe EMI, financial risk score, and a recommendation.

---

# ⭐ Conclusion

**EMIPredict AI** demonstrates how Machine Learning can be used to automate financial risk assessment and EMI affordability analysis.

The project combines:

```text
Data Science
      +
Machine Learning
      +
Feature Engineering
      +
Experiment Tracking
      +
Risk Analysis
      +
Gradio
```

to create a complete end-to-end AI application.

---

## 👤 Author

**Shaily Yadav**

B.Tech – Computer Science & Engineering
Specialization: Artificial Intelligence & Machine Learning


