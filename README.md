# Customer Churn Diagnosis & Retention Tool

**Predicting *why* customers churn — and calculating who's actually worth saving.**

Most churn projects stop at "here's a model that predicts churn." This project goes further: it statistically proves *which* factors drive churn (not just correlates with it), builds a model to rank customers by risk, and translates that risk into a business decision — who should a retention team actually call, and is it worth the cost?

🔗 **Live App:** _[add your Streamlit Cloud link here once deployed]_

---

## The Question

> Telecom companies lose revenue every time a customer churns. But retention budgets aren't infinite — you can't call everyone. **Which customers are actually worth calling, and why are they leaving in the first place?**

---

## Project Workflow

| Stage | Notebook | What it does |
|---|---|---|
| 1. Data Cleaning | `01_data_cleaning.ipynb` | Fixes a hidden data quality issue in `TotalCharges`, standardizes inconsistent encoding |
| 2. EDA | `02_eda.ipynb` | Visualizes churn patterns across contract type, tenure, charges, and internet service |
| 3. Hypothesis Testing | `03_hypothesis_testing.ipynb` | Proves 4 churn drivers statistically significant using Chi-square and t-tests (not just eyeballed) |
| 4. Modeling | `04_modelling.ipynb` | Logistic Regression vs. Random Forest, compared on recall/F1 — not just accuracy |
| 5. Business Impact | `05_business_impact.ipynb` | Converts churn probability into ₹ value, segments customers into an action matrix |
| App | `app.py` | Interactive Streamlit tool — enter a customer's details, get a live prediction + recommendation |

---

## Key Finding #1: The Data Had a Hidden Problem

`TotalCharges` looked clean at first glance — pandas reported **zero nulls**. But 11 rows contained blank strings, not real numbers, which silently forced the entire column to text type. Investigating *why* revealed all 11 belonged to customers with `tenure = 0` — brand-new customers who hadn't been billed yet. Rather than dropping them, they were correctly imputed as `0`, preserving all 7,043 customers.

---

## Key Finding #2: Four Drivers, Proven Statistically

| Hypothesis | Test | Result |
|---|---|---|
| Contract type → Churn | Chi-square | χ² = 1184.6, **p < 0.001** — significant |
| Tenure → Churn | t-test | Churned avg = 18.0 mo vs Retained avg = 37.6 mo, **p < 0.001** |
| Monthly Charges → Churn | t-test | Churned avg = $74.44 vs Retained avg = $61.27, **p < 0.001** |
| Internet Service type → Churn | Chi-square | χ² = 732.3, **p < 0.001** — Fiber optic churns ~42% vs ~7% for no-internet |

All four were later **independently confirmed** by Random Forest feature importance — two completely different methods (statistical tests vs. tree-based splits) converged on the same top drivers.

---

## Key Finding #3: Accuracy Lies When Classes Are Imbalanced

Churn is imbalanced (~73.5% No / 26.5% Yes). A model that predicts "No" for everyone scores 73.5% accuracy while catching zero actual churners. So both models were judged on **recall for the churn class**, not accuracy alone:

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 80% | 77% |
| Churn Precision | 0.64 | 0.55 |
| **Churn Recall** | 0.53 | **0.70** |
| Churn F1-score | 0.58 | **0.61** |
| ROC-AUC | 0.839 | 0.835 |

**Random Forest was chosen** despite lower accuracy — it catches 70% of actual churners vs. 53% for Logistic Regression. In a retention context, a missed churner is a lost customer; a false alarm is just one extra call. That asymmetry makes recall the metric that matters here.

---

## Key Finding #4: Turning Predictions Into a Business Decision

A churn probability alone isn't a decision. Each customer was scored on two axes — **Risk** (churn probability) and **Value** (estimated CLV = Monthly Charges × 24 months) — and sorted into an action matrix:

- 🔴 **Call Immediately** — high risk, high value
- 🟠 **Low-Cost Offer** — high risk, low value (not worth a phone call, but worth an email/SMS offer)
- 🟢 **Monitor** — low risk, high value
- ⚪ **No Action** — low risk, low value

**Result on the test set (1,409 customers):** 346 customers flagged as "Call Immediately." Targeting them costs an estimated ₹173,000 (at ₹500/call) but protects ₹714,746 in customer value — a **net expected gain of ₹348,128**.

*(Retention cost and CLV-period are stated assumptions, not given by the dataset — see notebook 05 for full reasoning.)*

---

## Tech Stack

- **Python** — pandas, numpy
- **Statistics** — scipy.stats (Chi-square, t-tests)
- **ML** — scikit-learn (Logistic Regression, Random Forest)
- **Visualization** — matplotlib, seaborn
- **App** — Streamlit

---

## Run It Yourself

```bash
git clone <your-repo-url>
cd churn-customer-diagnosis
pip install -r requirements.txt
streamlit run app.py
```

The app trains the model fresh on first launch (cached after that) — no pickle files needed.

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features, sourced via Kaggle.

---

## Author

Divyani — BSc Information Technology, SIES College, Mumbai