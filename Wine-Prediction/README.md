# 🍷 Wine Quality Prediction

A machine learning project that predicts whether a wine is **"good"** or **"bad"** quality based on its physicochemical properties, using the classic Red Wine Quality dataset (1,599 samples).

## 📌 Project Overview

Wine quality is traditionally rated on a scale of 0–10 by human tasters. This project reframes it as a **binary classification problem** — good vs. bad — and trains and compares two classic ML algorithms to see which predicts quality most accurately from measurable chemical properties.

## 📊 Dataset

- **Source:** Red Wine Quality dataset (`data.csv`)
- **Samples:** 1,599 wines
- **Features (11 numeric):**
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target:** `quality` (originally scored 3–8, re-binned into `bad` (≤6) and `good` (7–8), then label-encoded to 0/1)

## ⚙️ What Was Implemented

1. **Exploratory Data Analysis (EDA)**
   - Boxplot and point plot of `fixed acidity` vs. `quality` to inspect relationships between chemical properties and wine quality.
2. **Feature Engineering**
   - Converted the 0–10 quality score into two classes (`bad`, `good`) using `pd.cut`.
   - Encoded classes numerically with `LabelEncoder`.
3. **Preprocessing**
   - Train/test split (80/20, `random_state=42`).
   - Feature scaling with `StandardScaler`.
4. **Model Training & Hyperparameter Tuning**
   - **Random Forest Classifier** tuned via `GridSearchCV` (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) with 5-fold cross-validation.
   - **Support Vector Classifier (SVC)** tuned via `GridSearchCV` (`C`, `kernel`, `gamma`) with 10-fold cross-validation.
5. **Model Evaluation**
   - Accuracy score, precision/recall/F1-score (`classification_report`).
   - Confusion matrix heatmap for the tuned Random Forest model.
   - Feature importance plot to identify which chemical properties matter most for predicting quality.

## 🧠 Tech Stack / Libraries Used

- `pandas` — data loading & manipulation
- `seaborn` / `matplotlib` — visualization
- `scikit-learn` — modeling (`RandomForestClassifier`, `SVC`, `GridSearchCV`, `StandardScaler`, `LabelEncoder`, `train_test_split`, `classification_report`, `confusion_matrix`)

## 📈 Results

| Model | Accuracy |
|---|---|
| Random Forest (tuned) | ~89–90% |
| SVC (tuned) | **89.7%** |

**Best SVC classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Bad (0) | 0.90 | 0.99 | 0.94 | 273 |
| Good (1) | 0.89 | 0.34 | 0.49 | 47 |
| **Accuracy** | | | **0.90** | 320 |

> ⚠️ Note: The dataset is imbalanced (far more "bad" wines than "good" ones), which is why recall on the "good" class is noticeably lower. This is an important limitation to call out rather than hide.

## 🗂️ Project Structure

```
Wine-Prediction/
│
├── data/
│   └── data.csv                # Raw dataset
├── Wine_Prediction.ipynb        # Main notebook (EDA, modeling, evaluation)
├── images/
│   └── accuracy_report.png      # Classification report screenshot
└── README.md
```

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/Wine-Prediction.git
cd Wine-Prediction
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook Wine_Prediction.ipynb
```

## 🔮 Future Improvements

- Address class imbalance (e.g., SMOTE, class weighting) to improve recall on "good" wines.
- Try additional models (XGBoost, Logistic Regression) for comparison.
- Add ROC-AUC and precision-recall curves for a more complete evaluation, especially given the imbalance.
- Deploy as a simple web app (Streamlit/Flask) for interactive predictions.

## 🙋 Author

Built as a hands-on machine learning / data analysis project applying classification techniques to a real-world dataset.
