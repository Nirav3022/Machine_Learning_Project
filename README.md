# 🌦️ Seattle Weather Prediction

A machine learning project that predicts the **weather category** (drizzle, rain, sun, snow, or fog) for a given day in Seattle, using historical daily precipitation, temperature, and wind data.

<img width="1884" height="1610" alt="04_confusion_matrix" src="https://github.com/user-attachments/assets/d3e3a371-7c0c-46e9-80e5-2a45b72b4057" />


## 📊 Dataset

[seattle-weather.csv](https://github.com/user-attachments/files/29942477/seattle-weather.csv)

| Column | Description |
|---|---|
| `date` | Observation date |
| `precipitation` | Precipitation (mm) |
| `temp_max` | Max temperature (°C) |
| `temp_min` | Min temperature (°C) |
| `wind` | Wind speed |
| `weather` | Target label: drizzle / rain / sun / snow / fog |

## 🛠️ What was implemented

- **Data cleaning & preprocessing**: checked for nulls, encoded the categorical `weather` label with `LabelEncoder`, and normalized numeric features (precipitation, temp_max, temp_min, wind) to a 0–1 scale.
- **Train/test split**: 80/20 split, **stratified** on the weather label so rare classes (snow, drizzle) are proportionally represented in both sets.
- **Model**: `GradientBoostingClassifier` (scikit-learn) trained on 4 numeric features.
- **Evaluation**: accuracy, per-class precision/recall/F1 (`classification_report`), and a confusion matrix — evaluated on a **held-out test set** (see "Fix" note below).
- **Model interpretability**: feature importance ranking to see which weather signals matter most.
- **Visualizations**: class distribution, correlation heatmap, feature distributions per weather type, confusion matrix, feature importance, and actual-vs-predicted comparison over time.

### 🐛 Fix vs. the original version
The original notebook evaluated `accuracy_score` on the **training data** the model had already seen, which inflates performance and hides overfitting. This version evaluates on a proper held-out **test set**, giving an honest read on real-world performance:

| Metric | Score |
|---|---|
| Train accuracy | ~91.5% |
| **Test accuracy** | **~83.6%** |

## 🚀 Features / Tech Used

- Python 3
- pandas, numpy — data handling
- scikit-learn — `LabelEncoder`, `train_test_split`, `GradientBoostingClassifier`, metrics
- matplotlib, seaborn — visualization
- Jupyter Notebook

## 📁 Project structure

```
weather-prediction/
├── Weather_Prediction.ipynb   # main notebook: EDA, preprocessing, model, evaluation
├── generate_charts.py         # standalone script that regenerates all chart images
├── data/
│   └── seattle-weather.csv
├── images/                    # exported charts (used in README + LinkedIn post)
├── metrics.txt                # saved accuracy / classification report
├── requirements.txt
└── README.md
```

## ▶️ How to run

```bash
git clone https://github.com/<your-username>/weather-prediction.git
cd weather-prediction
pip install -r requirements.txt
jupyter notebook Weather_Prediction.ipynb
```

## 📈 Key visuals

| | |
|---|---|
(images/01_we<img width="2060" height="1459" alt="01_weather_distribution" src="https://github.com/user-attachments/assets/ab73958f-928a-43ae-b10e-816f83bb9979" />
ather_distribution.png)
<img width="1744" height="1437" alt="02_correlation_heatmap" src="https://github.com/user-attachments/assets/1d4c1e7e-6e94-4f1a-808b-481244d80996" />
(images/05_feat<img width="2061" height="1310" alt="05_feature_importance" src="https://github.com/user-attachments/assets/61e03b11-034f-4f2c-8d3a-fc77537f6e8a" />
ure_importance.png)
<img width="3860" height="1459" alt="06_actual_vs_predicted" src="https://github.com/user-attachments/assets/d96c2918-f8a1-4cae-b2b3-4119a1479e9a" />

## 🔭 Next steps

- Try `XGBClassifier` and compare against `GradientBoostingClassifier`
- Handle class imbalance (very few snow/drizzle days) with class weights or SMOTE
- Add k-fold cross-validation
- Engineer a seasonal feature (month / day-of-year)

## 📄 License

MIT — feel free to use and adapt.
