# Customer Churn Diagnosis & Retention Tool

**Predicting customer churn and identifying which customers are worth retaining.**

This project uses hypothesis testing and machine learning to analyze why telecom customers churn, predict customer risk, and convert predictions into a business-focused retention strategy. The project also includes an interactive Streamlit dashboard for live churn prediction and customer analysis.

---

## Business Problem

Telecom companies lose revenue when customers churn, but retention budgets are limited. This project helps identify:

* Why customers churn
* Which customers are most likely to leave
* Which high-risk customers are actually worth retaining

---

## Project Workflow

| Stage               | Notebook/File                 | Description                                        |
| ------------------- | ----------------------------- | -------------------------------------------------- |
| Data Cleaning & EDA | `01_data_cleaning&EDA.ipynb`  | Cleaned data issues and analyzed churn patterns    |
| Hypothesis Testing  | `02_hypothesis_testing.ipynb` | Used statistical tests to validate churn drivers   |
| Modeling            | `03_modelling.ipynb`          | Built Logistic Regression and Random Forest models |
| Business Impact     | `04_business_impact.ipynb`    | Converted churn predictions into business actions  |
| Streamlit App       | `app.py`                      | Interactive customer churn prediction dashboard    |

---

## Key Insights

* Customers with month-to-month contracts showed significantly higher churn.
* Customers with lower tenure were more likely to leave.
* Higher monthly charges increased churn probability.
* Fiber optic internet customers had the highest churn rate.
* Random Forest achieved better churn recall than Logistic Regression.

---

## Business Impact

Customers were segmented using:

* **Churn Risk**
* **Customer Lifetime Value (CLV)**

Based on these factors, customers were grouped into:

* 🔴 Call Immediately
* 🟠 Low-Cost Offer
* 🟢 Monitor
* ⚪ No Action

This helps prioritize retention efforts and estimate potential business value saved.


---

## Tech Stack

* Python
* pandas, numpy
* matplotlib, seaborn
* scipy.stats
* scikit-learn
* Streamlit

---

## Skills Demonstrated

* Data Cleaning & EDA
* Hypothesis Testing
* Machine Learning
* Customer Churn Analysis
* Business Analytics
* Data Visualization
* Streamlit App Development

---

## Project Structure

```bash
churn-customer-diagnosis/
│
├── data/
├── notebooks/
├── reports/
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/divyanipatil093/churn-customer-diagnosis
cd churn-customer-diagnosis
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

IBM Telco Customer Churn Dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Author

Divyani Patil
BSc Information Technology — SIES College, Mumbai
