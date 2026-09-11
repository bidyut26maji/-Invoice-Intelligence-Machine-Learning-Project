# 📦 Vendor Invoice Intelligence Portal

### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

An end-to-end **Machine Learning project** designed to help finance and procurement teams analyze vendor invoices, predict freight costs, and identify invoices that may require manual approval.

The project combines **Machine Learning, Python, Pandas, Scikit-Learn, SQL, and Streamlit** to provide an interactive invoice intelligence platform.

---

## 🚀 Live Application

🌐 **Streamlit App:**  
https://invoice-intelligence-ml.streamlit.app

> The application provides an interactive interface for freight cost prediction and invoice risk classification.

---

## 📌 Project Overview

Vendor invoices often contain transportation costs, item quantities, invoice amounts, and other financial information that need to be reviewed by finance teams.

Manually reviewing every invoice can be:

- Time-consuming
- Expensive
- Difficult to scale
- Prone to human error
- Inefficient for large volumes of invoices

This project uses Machine Learning to assist with two important finance operations:

### 🚚 1. Freight Cost Prediction

A Machine Learning **regression model** predicts the expected freight/transportation cost based on invoice-related information.

### 🚩 2. Invoice Risk Flagging

A Machine Learning **classification model** predicts whether an invoice may require **manual approval/review** based on invoice, freight, and item-level patterns.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Predict expected freight costs
- Identify potentially abnormal invoices
- Reduce manual invoice review workload
- Support finance and procurement teams
- Improve cost forecasting
- Assist in identifying potentially risky invoice patterns
- Provide an easy-to-use ML interface
- Demonstrate an end-to-end Machine Learning workflow

---

# 💼 Business Impact

The system is designed to support organizations in improving financial operations.

### 💰 Improved Cost Forecasting

Predict expected transportation costs before or during invoice processing.

### 🚩 Reduced Invoice Anomalies

Identify invoice patterns that may require additional investigation.

### ⚡ Faster Finance Operations

Reduce the amount of manual work required to review invoices.

### 📊 Data-Driven Decisions

Use historical invoice data and Machine Learning predictions to support financial decision-making.

### 🤝 Vendor Management

Predicted freight costs can help organizations compare expected and reported transportation costs during vendor evaluation and negotiations.

---

# 🧠 Machine Learning Modules

The project contains two primary Machine Learning modules.

---

## 🚚 Module 1 — Freight Cost Prediction

### Problem Type

**Regression**

The model predicts a continuous numerical value representing the expected freight cost.

### Example Input
The exact prediction depends on the trained model and historical data.

Business Use Case

The prediction can be used to:

Estimate transportation expenses
Improve budgeting
Compare expected and actual freight charges
Identify unusually high transportation costs
Support vendor negotiations
🚩 Module 2 — Invoice Risk Flagging
Problem Type

Binary Classification

The classification model predicts whether an invoice may require additional manual review.

Input Features

The current application uses:

invoice_quantity
invoice_dollars
Freight
total_item_quantity
total_item_dollars
Prediction

The model produces a binary prediction:

0 → Lower Risk
1 → Higher Risk

The Streamlit application converts this prediction into a user-friendly recommendation.

Low-Risk Example
Invoice Quantity: 50
Invoice Dollars: $352.95
Freight: $1.73
Total Item Quantity: 162
Total Item Dollars: $2,476

Possible application result:

🟢 LOW RISK — AUTO-APPROVAL RECOMMENDED
High-Risk Example

An invoice containing highly unusual values or patterns may produce:

🔴 HIGH RISK — MANUAL REVIEW RECOMMENDED

Important: A Machine Learning prediction is not a guarantee that an invoice is fraudulent or safe. The system is designed to support human review and financial decision-making.

🏗️ Project Architecture
                    ┌───────────────────────┐
                    │     Invoice Data      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Data Preprocessing    │
                    │ Cleaning & Features   │
                    └───────────┬───────────┘
                                │
                     ┌──────────┴──────────┐
                     │                     │
                     ▼                     ▼
          ┌───────────────────┐   ┌────────────────────┐
          │ Freight Regression│   │ Invoice Classifier │
          │      Model        │   │       Model        │
          └─────────┬─────────┘   └──────────┬─────────┘
                    │                        │
                    ▼                        ▼
          ┌───────────────────┐   ┌────────────────────┐
          │ Freight Prediction│   │ Risk Prediction    │
          └─────────┬─────────┘   └──────────┬─────────┘
                    │                        │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      Streamlit        │
                    │   Interactive Portal  │
                    └───────────────────────┘
📂 Project Structure
Machine Learning Project/
│
├── 📁 data/
│   └── Dataset files
│
├── 📁 freight_cost_prediction/
│   └── Freight prediction related files
│
├── 📁 invoice_flagging/
│   └── Invoice classification related files
│
├── 📁 inference/
│   ├── __init__.py
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── 📁 notebooks/
│   └── Jupyter notebooks for analysis and experimentation
│
├── 📄 app.py
├── 📄 requirements.txt
├── 📄 train.py
├── 📄 data_preprocessing.py
└── 📄 modeling_evaluation.py

Model file locations may vary depending on the final project structure.

🛠️ Technology Stack
Technology	Purpose
🐍 Python	Core programming language
🐼 Pandas	Data manipulation and analysis
🔢 NumPy	Numerical computation
🤖 Scikit-Learn	Machine Learning
🗄️ SQL	Data querying and processing
📊 Plotly	Interactive visualization
🎨 Streamlit	Web application and deployment
💾 Joblib	Saving and loading ML models
📓 Jupyter Notebook	Data analysis and experimentation
🔧 Git	Version control
🐙 GitHub	Source code hosting
🔄 Machine Learning Workflow

The project follows a standard Machine Learning pipeline.

1. Data Collection
       ↓
2. Data Cleaning
       ↓
3. Exploratory Data Analysis
       ↓
4. Feature Selection
       ↓
5. Data Preprocessing
       ↓
6. Train/Test Split
       ↓
7. Model Training
       ↓
8. Model Evaluation
       ↓
9. Model Serialization
       ↓
10. Inference
       ↓
11. Streamlit Deployment
🧹 Data Preprocessing

The data preprocessing stage prepares the invoice dataset for Machine Learning.

Typical preprocessing operations include:

Handling missing values
Removing unnecessary columns
Checking duplicate records
Converting data types
Selecting relevant features
Preparing target variables
Splitting training and testing data
Feature scaling where required

The same preprocessing logic used during training should be applied consistently during inference.

🔍 Feature Engineering

The invoice classification module uses invoice-level and item-level information.

Invoice Features
invoice_quantity
invoice_dollars
Freight Feature
Freight
Item-Level Features
total_item_quantity
total_item_dollars

These features allow the model to learn relationships between invoice size, item quantities, total values, and freight charges.

🤖 Model Training

The project uses Scikit-Learn Machine Learning algorithms.

Freight Prediction

A regression model is used because freight cost is a continuous numerical value.

Input Features
      ↓
Regression Model
      ↓
Predicted Freight Cost
Invoice Flagging

A classification model is used because the output represents two possible classes.

Input Features
      ↓
Classification Model
      ↓
0 → Lower Risk
1 → Higher Risk
📊 Model Evaluation

Machine Learning models should be evaluated using a separate test dataset.

Classification Metrics

The invoice classification model can be evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC

Example:

Accuracy  : XX.XX%
Precision : XX.XX%
Recall    : XX.XX%
F1 Score  : XX.XX%
Why Recall Matters

For invoice risk detection, recall can be particularly important.

A model that misses many genuinely risky invoices may not be useful even if its overall accuracy is high.

Therefore, the model should be evaluated using multiple metrics rather than accuracy alone.

🚚 Freight Regression Evaluation

The freight prediction model can be evaluated using regression metrics such as:

MAE
MSE
RMSE
R² Score

Example:

MAE  : XX.XX
RMSE : XX.XX
R²   : XX.XX

These metrics help determine how closely the predicted freight cost matches the actual freight cost.

🖥️ Streamlit Application

The Streamlit application provides two prediction modules.

Sidebar

Users can select:

🔹 Freight Cost Prediction
🔹 Invoice Manual Approval Prediction
🚚 Freight Cost Prediction Interface

Users provide:

📦 Quantity
💰 Invoice Dollars

The application returns:

📊 Estimated Freight Cost
🚩 Invoice Risk Prediction Interface

Users provide:

📦 Invoice Quantity
🚚 Freight Cost
💰 Invoice Dollars
📦 Total Item Quantity
💵 Total Item Dollars

The model returns a risk prediction.

Example:

🟢 LOW RISK
AUTO-APPROVAL RECOMMENDED

or:

🔴 HIGH RISK
MANUAL REVIEW RECOMMENDED
🎨 Dashboard Features

The Streamlit dashboard includes:

Modern finance-oriented interface
Sidebar navigation
Model selection
Interactive input forms
Prediction results
Risk indicators
Business impact information
Invoice evaluation details
Interactive visualizations
Responsive layout
💻 Installation
1. Clone the Repository
git clone https://github.com/bidyut26maji/-Invoice-Intelligence-Machine-Learning-Project.git

Navigate into the project:

cd -Invoice-Intelligence-Machine-Learning-Project
2. Create a Virtual Environment
Windows
python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process Bypass

Then:

.\.venv\Scripts\Activate.ps1
3. Install Dependencies
pip install -r requirements.txt

If requirements.txt is not available or incomplete:

pip install streamlit pandas numpy scikit-learn joblib plotly
▶️ Run the Application Locally

From the project root:

python -m streamlit run app.py

Or:

streamlit run app.py

The application will normally open at:

http://localhost:8501

Jupyter Notebook normally runs on port 8888, while Streamlit commonly runs on port 8501.

🧪 Running the Machine Learning Pipeline

Depending on the project structure, the training workflow can be executed using:

python train.py

The training process should:

Load the dataset
Clean the data
Prepare features
Create training and testing datasets
Train the model
Evaluate the model
Save the trained model
💾 Model Inference

The inference/ directory contains prediction functions.

Freight
from inference.predict_freight import predict_freight_cost
Invoice Risk
from inference.predict_invoice_flag import predict_invoice_flag

These functions load the trained models and generate predictions for new invoice data.

📋 Example Invoice Prediction
input_data = {
    "invoice_quantity": [50],
    "invoice_dollars": [352.95],
    "Freight": [1.73],
    "total_item_quantity": [162],
    "total_item_dollars": [2476.0]
}

The model produces a classification result:

Predicted_Flag
       ↓
0 → Lower Risk
1 → Higher Risk
🚨 Important Model Considerations

The system is a Machine Learning decision-support tool.

A prediction such as:

LOW RISK

does not guarantee that an invoice is legitimate.

Similarly:

HIGH RISK

does not prove that an invoice is fraudulent.

The predictions should be used to prioritize invoices for review.

🔐 Security & Data Privacy

If real financial or vendor data is used:

Do not commit sensitive invoice information to GitHub
Do not commit passwords or API keys
Do not expose database credentials
Use environment variables for secrets
Add sensitive files to .gitignore
Anonymize confidential business information before sharing the repository

Example .gitignore:

# Virtual environment
.venv/
venv/

# Python cache
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env

# Sensitive datasets
*.xlsx
*.xls
*.db

# Logs
*.log

# OS files
.DS_Store
Thumbs.db
📈 Future Improvements

Possible future improvements include:

🔹 Advanced Anomaly Detection

Add unsupervised Machine Learning models such as:

Isolation Forest
Local Outlier Factor
One-Class SVM
🔹 Explainable AI

Add explanations showing why an invoice was flagged.

For example:

Risk Factors:
✓ Freight unusually high
✓ Invoice value significantly above historical range
✓ Quantity/value relationship is unusual
🔹 Probability Scores

Display classification probabilities:

Low Risk Probability  : 12%
High Risk Probability : 88%
🔹 Model Monitoring

Track:

Prediction performance
Data drift
Feature drift
Model accuracy over time
🔹 Database Integration

Connect the application directly to a production SQL database.

🔹 User Authentication

Add secure login functionality for finance teams.

🔹 Batch Invoice Processing

Allow users to upload CSV/Excel files containing multiple invoices.

🔹 Automated Alerts

Send notifications when high-risk invoices are detected.

🧑‍💻 Development Workflow

The project follows a typical Git/GitHub workflow:

Local Development
       ↓
Git Add
       ↓
Git Commit
       ↓
Git Push
       ↓
GitHub
       ↓
Streamlit Deployment

Useful commands:

git status
git add .
git commit -m "Update invoice intelligence portal"
git push origin main
📊 Example Business Scenario

Imagine a company receives thousands of vendor invoices every month.

Without automation:

1000 Invoices
      ↓
Manual Review
      ↓
Large Finance Workload
      ↓
Slow Processing

With the Invoice Intelligence Portal:

1000 Invoices
      ↓
Machine Learning
      ↓
Risk Classification
      ↓
┌──────────────────────┐
│ Lower Risk           │ → Automated processing
│ Higher Risk          │ → Manual review
└──────────────────────┘

The system therefore helps finance teams focus their attention on invoices that deserve closer examination.

🎯 Skills Demonstrated

This project demonstrates practical knowledge of:

Python
Pandas
NumPy
SQL
Exploratory Data Analysis
Data Cleaning
Feature Engineering
Feature Selection
Machine Learning
Regression
Classification
Model Evaluation
Scikit-Learn
Model Serialization
Joblib
Streamlit
Data Visualization
Git
GitHub
ML Deployment
👨‍💻 Author

Bidyut Maji

Machine Learning / Data Analytics Project

GitHub:

https://github.com/bidyut26maji

⭐ Project Highlights
🚚 Freight Cost Prediction
🚩 Invoice Risk Classification
📊 Interactive Analytics Dashboard
🤖 Machine Learning
🐍 Python
🐼 Pandas
🧠 Scikit-Learn
🗄️ SQL
🎨 Streamlit
🐙 GitHub
📜 Disclaimer

This project is intended for educational, analytical, and decision-support purposes.

Machine Learning predictions should not be considered definitive financial, fraud, compliance, or legal decisions. Real-world deployment should include appropriate validation, business rules, human review, security controls, and model monitoring.

⭐ If you find this project useful

Feel free to star ⭐ the repository, explore the code, and provide feedback.


### One important recommendation before you upload this

I intentionally used **`XX.XX%`** for the model metrics rather than inventing accuracy/precision/recall numbers. Replace those with the **actual values from your model evaluation**.

Also, your README currently says:

> `0 → Lower Risk`  
> `1 → Higher Risk`

Make sure this is actually how your training dataset defines the target. Earlier you had an issue where a norm
```text
Quantity: 1200
Invoice Dollars: $18,500
