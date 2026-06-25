# 🏥 Medical Insurance Cost Prediction

A Machine Learning regression project that predicts annual medical insurance charges based on demographic and health-related attributes. The app displays results in both **USD and INR**.

🔗 **Live App:** [Click here to open the app](https://geetha8247-medical-insurance-prediction-app.streamlit.app)

---

## 📋 Problem Statement

Given personal attributes such as age, sex, BMI, smoking status, region, and number of children — predict the **annual medical insurance charges** for an individual.

This is a **Regression** problem because the target variable `charges` is a continuous numeric value with no fixed upper bound.

---

## 📁 Repository Structure

```
medical-insurance-prediction/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

> Note: No pkl files needed — the model trains automatically on first app load using live data.

---

## 📊 Dataset Description

- **Source:** [Kaggle — Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Mirror Used:** GitHub raw CSV (auto-downloaded in app)
- **Rows:** 1,338
- **Columns:** 7 (6 features + 1 target)

| Column   | Type        | Description                              |
|----------|-------------|------------------------------------------|
| age      | Numerical   | Age of primary beneficiary               |
| sex      | Categorical | Gender (male / female)                   |
| bmi      | Numerical   | Body Mass Index                          |
| children | Numerical   | Number of dependents covered             |
| smoker   | Categorical | Smoking status (yes / no)                |
| region   | Categorical | US residential region (NE/NW/SE/SW)      |
| charges  | **Target**  | Annual insurance cost in USD (continuous)|

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Checked shape, dtypes, missing values, duplicates
- Target `charges` is right-skewed — smokers drive high-cost outliers
- `smoker` has strongest correlation with charges (0.79)
- Visualized distributions, scatter plots, boxplots, heatmap

### 2. Data Cleaning
- No missing values found in dataset
- Removed duplicate rows
- Capped `charges` at 99th percentile to handle extreme outliers

### 3. Feature Engineering (5 New Features)

| Feature       | Method      | Justification                                  |
|---------------|-------------|------------------------------------------------|
| bmi_category  | Binning     | Clinical risk groups (Obese/Overweight/Normal) |
| age_group     | Binning     | Non-linear age-risk relationship               |
| smoker_obese  | Interaction | Smoker + Obese = compounded cost risk          |
| family_size   | Binning     | Children count grouped into 3 tiers            |
| age_bmi       | Interaction | Age x BMI captures compound aging effect      |

### 4. Preprocessing Pipeline
- `StandardScaler` on numerical features
- `OneHotEncoder` on categorical features
- `ColumnTransformer` combining both
- 80/20 train-test split with `random_state=42`

### 5. Models Trained

| Model              | Justification                                       |
|--------------------|-----------------------------------------------------|
| Linear Regression  | Baseline — simple and interpretable                 |
| Random Forest      | Handles non-linearity; robust to outliers           |
| **XGBoost**        | Best performance — gradient boosting ✅ Selected   |

---

## 📈 Results

| Model             | MAE       | RMSE      | R² Score |
|-------------------|-----------|-----------|----------|
| Linear Regression | ~4,200    | ~6,000    | ~0.78    |
| Random Forest     | ~2,500    | ~4,500    | ~0.87    |
| **XGBoost**       | **~2,200**| **~4,100**| **~0.89**|

> ✅ **XGBoost** selected as the best model with R² ≈ 0.89

### Key Findings
- 🚬 **Smoking** is the #1 predictor — increases charges by ~4x
- ⚖️ **BMI** and **Age** positively correlate with charges
- 🔥 **Smoker + Obese** combination has the highest risk
- 👨‍👩‍👧 Region and number of children have lower impact

---

## 🖥️ Streamlit App Features

- Input form with sliders and dropdowns
- Predict button with instant results
- Cost displayed in both **USD** and **INR** (1 USD = 83.5 INR)
- Monthly breakdown shown
- Risk factor warnings (smoking, obesity)
- Error handling built-in
- Model trains automatically — no pkl files needed

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Geetha8247/medical-insurance-prediction.git
cd medical-insurance-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 📸 Screenshots

### App Interface
> Input form with age, BMI, smoker status, region inputs

### Prediction Output
> Displays estimated cost in USD and INR with monthly breakdown

---

## 🎁 Bonus Completed
- ✅ Deployed on Streamlit Cloud (free hosting)
- ✅ Public URL shared
- ✅ No local setup needed — runs in browser

---

## 👤 Author

- **Name:** Geetha
- **GitHub:** [@Geetha8247](https://github.com/Geetha8247)
- **Dataset:** Medical Insurance Cost Prediction
- **Task Type:** Regression

---

## 📄 License
This project is for academic purposes only.
