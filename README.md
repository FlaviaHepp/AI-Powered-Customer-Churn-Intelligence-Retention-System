# 🧠AI-Powered Customer Churn Intelligence & Retention System

## 💰From Prediction to Action: A Business-Driven Churn Strategy

---

## 📌Business Problem

Customer churn is one of the most critical challenges in banking.

However, most models fail because:
- They focus only on prediction accuracy  
- They ignore **economic impact**  
- They do not provide **actionable decisions**

👉 This project solves that gap.

---

## 🎯Objective

Build an **end-to-end churn intelligence system** that:

- Predicts customer churn
- Maximizes retention profitability
- Generates business-driven actions
- Delivers real-time predictions via API

---

## 🚀Key Features

### 🤖 Machine Learning
- Multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost (best performer)
- Model comparison using ROC AUC

---

### 💰 Profit Optimization Engine
- Custom profit function
- Threshold optimization
- Business-driven decision making

``python
Profit = (True Positives * Benefit) - (False Positives * Cost)

## 🧠Advanced Feature Engineering

Business-oriented features:

Age segmentation (customer lifecycle)
High-value customer detection
Engagement score
Balance-to-salary ratio
Inactive high-balance customers

👉 These features simulate real banking segmentation logic

## 🔍Explainability (SHAP)

Model interpretability using SHAP:

Global feature importance
Customer-level explanations
Identification of churn drivers
⚙️ MLOps with MLflow
Experiment tracking
Metrics logging (AUC, profit, threshold)
Model versioning
Reproducibility
🌐 Production-Ready API (FastAPI)

Endpoints:

/predict → churn prediction
/predict_pro → advanced decision engine

Returns:
- churn probability
- prediction
- explanation
- recommended action
- retention message

## 🧠Decision Engine (Business Logic)

Transforms predictions into actions:
- Premium retention (high-value clients)
- Reactivation campaigns
- Standard retention strategies
  
## 💬Personalized Retention Strategy
- Automated explanations
- Customer-specific retention messages
- Optional LLM-based explanation generation

## 📊Campaign Simulation
- Profit simulation
- Conversion rate estimation
- A/B testing (model vs random strategy)

## 🔁Model Monitoring
- Drift detection
- Logging predictions
- Performance tracking

## 📈Results & Impact
- Improved targeting vs random strategy
- Optimized retention costs
- Increased ROI through threshold tuning
- Business-aligned decision making

## 🛠Tech Stack
Python
Pandas / NumPy
Scikit-learn
XGBoost
SHAP
Matplotlib / Seaborn
MLflow
FastAPI
Joblib

## 📂Project Structure
│── churn_bank.py
│── churn.csv
│── model.pkl
│── scaler.pkl
│── features.pkl
│── README.md
│── requirements.txt

## ▶️How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train model
python churn_bank.py
3. Run API
uvicorn churn_bank:app --reload
4. Test endpoint
POST /predict_pro
{
  "CreditScore": 600,
  "Age": 45,
  "Balance": 120000,
  "EstimatedSalary": 50000,
  "IsActiveMember": 0,
  "NumOfProducts": 1
}

## 🔥Key Insights
Prediction alone is not enough → decisions matter
Profit-based thresholds outperform default models
Feature engineering drives business value
Explainability is critical for adoption

## 🧠Business Takeaways

✔ Data Science must be aligned with ROI
✔ Not all churn is equally important
✔ Decision systems outperform prediction models
✔ Personalization increases retention effectiveness

## 🚀What Makes This Project Different

This is not just a churn model.

👉 It is a Customer Intelligence System that:
- Predicts
- Explains
- Decides
- Acts
## 📬About Me

Data Scientist specialized in:

💰 Revenue Optimization
📊 Business Analytics
🧠 Customer Intelligence Systems

I build data products that drive measurable business impact.


---
