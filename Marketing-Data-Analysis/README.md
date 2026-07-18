# 🏦 Bank Marketing Campaign Analysis — Term Deposit Subscription Prediction

Predicting which bank customers are likely to subscribe to a term deposit, using the
UCI Bank Marketing dataset and a tuned Decision Tree classifier.

## 📌 Project Overview

Banks run large-scale telemarketing campaigns to sell term deposits, but most calls
don't convert. This project analyzes **45,211 customer records** from a Portuguese
bank's direct marketing campaign to:

- Understand **which customer segments** are most likely to subscribe
- Identify the **best times of year** to run campaigns
- Build a **machine learning model** to predict subscription likelihood, so marketing
  teams can prioritize high-value leads instead of cold-calling everyone

## 📊 Dataset

- **Source:** UCI Machine Learning Repository — Bank Marketing Dataset
- **Records:** 45,211
- **Features:** 17 (demographic, financial, and campaign-related attributes)
- **Target:** `y` — did the customer subscribe to a term deposit? (`yes` / `no`)
- **Class balance:** ~88.3% "no" vs ~11.7% "yes" (imbalanced classification problem)

| Feature | Description |
|---|---|
| age, job, marital, education | Customer demographics |
| default, balance, housing, loan | Financial profile |
| contact, day, month, duration | Contact/campaign details |
| campaign, pdays, previous, poutcome | Prior campaign history |

## 🔧 Methodology

1. **Data Wrangling**
   - Converted `yes`/`no` text columns to booleans
   - Bucketed `balance` into 5 classes (A–E) based on quantiles
   - Flagged whether a customer was contacted in a previous campaign
   - Dropped low-signal / leaky columns: `pdays`, `poutcome`, `previous`, `day`

2. **Exploratory Data Analysis**
   - Visualized subscription rate by age group, job, month, and account balance class
   - Identified seasonal and demographic patterns driving conversion

3. **Modeling**
   - Encoded categorical features with `OrdinalEncoder`
   - Split data 85/15 (stratified) into train/test sets
   - Trained a baseline `DecisionTreeClassifier`
   - Tuned hyperparameters (`max_depth`, `criterion`, `min_samples_split`,
     `min_samples_leaf`) via `GridSearchCV` with 10-fold cross-validation, optimizing
     for **recall** on the minority ("yes") class

## 📈 Key Insights

| Insight | Finding |
|---|---|
| **Best age segments** | Customers ≤25 and 56+ convert at the highest rates |
| **Best occupations** | Students and retirees are the most likely to subscribe |
| **Best campaign months** | March, September, October, and December significantly outperform other months |
| **Balance effect** | Higher account balances correlate with higher subscription likelihood |
| **Top model drivers** | Age, contact month, job, and number of campaign contacts |

## 🎯 Model Performance (Tuned Decision Tree)

| Metric | Class: No | Class: Yes | Overall |
|---|---|---|---|
| Precision | 0.91 | 0.28 | — |
| Recall | 0.90 | 0.29 | — |
| F1-score | 0.90 | 0.28 | — |
| Accuracy | — | — | **0.83** |

The model was intentionally tuned to **maximize recall on subscribers** rather than raw
accuracy — in a marketing context, missing a genuine lead (false negative) is costlier
than an occasional wasted call (false positive).

## 🖼️ Visualizations Included

1. Subscription class distribution
2. Subscription rate by age group
3. Subscription rate by occupation
4. Subscription rate by contact month (seasonality)
5. Subscription rate by account balance class
6. Confusion matrix (tuned model)
7. Feature importance ranking

## 💡 Business Recommendations

1. **Time campaigns strategically** — concentrate outreach in March, September,
   October, and December.
2. **Prioritize high-converting segments** — students, retirees, and customers aged
   56+ show the strongest response rates.
3. **Cap contact frequency** — repeated calls to the same customer show diminishing
   returns; the model flags `campaign` count as a top driver of outcome.
4. **Use the model as a lead-scoring layer** — rank customers by predicted
   subscription probability before dialing, to improve team efficiency.

## 🛠️ Tech Stack

- **Python** — pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Machine Learning** — scikit-learn (Decision Tree, GridSearchCV, OrdinalEncoder)
- **Environment** — Jupyter Notebook

## 📁 Repository Structure

```
├── Code/
      └── Marketing-Analysis.ip
├── data/
│   └── bank.csv                    # Full analysis notebook with visualizations
├── images/                            # Dataset (UCI Bank Marketing)       # Exported chart images
└── README.md                             # Project documentation
```

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Marketing_Analysis_Visualized.ipynb
```

## 📎 Dataset Citation

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the
Success of Bank Telemarketing.* Decision Support Systems, Elsevier.
UCI Machine Learning Repository: Bank Marketing Data Set.

---

*This project is for educational/portfolio purposes, demonstrating an end-to-end
data analysis and classification workflow: EDA → feature engineering → model tuning →
business insight generation.*
