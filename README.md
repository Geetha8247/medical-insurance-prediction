# 🏥 Medical Insurance Cost Prediction

A Machine Learning regression project to predict annual medical insurance charges based on demographic and health-related attributes.

---

## 📋 Problem Statement

Given personal attributes (age, sex, BMI, smoking status, region, number of children), predict the **annual medical insurance charges** for an individual. This is a **regression** problem since the target variable `charges` is a continuous numeric value.

---

## 📁 Repository Structure

```
insurance_project/
│
├── data/                          # (place insurance.csv here)
├── notebooks/
│   └── Medical_Insurance_Cost_Prediction.ipynb   # Main Colab notebook
├── src/                           # (helper scripts if any)
├── models/
│   ├── model.pkl                  # Trained best model
│   ├── preprocessor.pkl           # Sklearn ColumnTransformer
│   └── feature_config.pkl         # Feature column metadata
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 📊 Dataset Description

- **Source:** [Kaggle — Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Rows:** 1,338
- **Columns:** 7 (6 features + 1 target)

| Column   | Type        | Description                            |
|----------|-------------|----------------------------------------|
| age      | Numerical   | Age of primary beneficiary             |
| sex      | Categorical | Gender (male / female)                 |
| bmi      | Numerical   | Body Mass Index                        |
| children | Numerical   | Number of dependents covered           |
| smoker   | Categorical | Smoking status (yes / no)              |
| region   | Categorical | US region (NE / NW / SE / SW)          |
| charges  | **Target**  | Annual insurance cost in USD           |

---

## ⚙️ Methodology

### 1. EDA
- Checked shape, dtypes, missing values, duplicates
- Analyzed distribution of `charges` (right-skewed)
- Found `smoker` has the strongest correlation with charges (0.79)
- Visualized scatter plots, boxplots, and correlation heatmap

### 2. Data Cleaning
- No missing values found
- Removed 1 duplicate row
- Capped `charges` at 99th percentile to handle extreme outliers

### 3. Feature Engineering (5 features)
| Feature       | Method      | Justification                              |
|---------------|-------------|--------------------------------------------|
| bmi_category  | Binning     | Clinical risk buckets (Obese/Overweight…)  |
| age_group     | Binning     | Non-linear age-risk relationship           |
| smoker_obese  | Interaction | Smoker + Obese = compounded cost risk      |
| family_size   | Binning     | Children count grouped into 3 tiers        |
| age_bmi       | Interaction | Age × BMI captures compound aging effect  |

### 4. Preprocessing Pipeline
- `StandardScaler` on numerical features
- `OneHotEncoder` on categorical features
- `ColumnTransformer` combining both
- 80/20 train-test split with `random_state=42`

### 5. Models Trained
| Model              | Justification                                      |
|--------------------|---------------------------------------------------|
| Linear Regression  | Baseline — simple and interpretable               |
| Random Forest      | Handles non-linearity; robust to outliers         |
| XGBoost            | State-of-art gradient boosting; best performance  |

---

## 📈 Results

| Model             | MAE       | RMSE      | R² Score |
|-------------------|-----------|-----------|----------|
| Linear Regression | ~4,200    | ~6,000    | ~0.78    |
| Random Forest     | ~2,500    | ~4,500    | ~0.87    |
| **XGBoost**       | **~2,200**| **~4,100**| **~0.89**|

> ✅ **XGBoost** achieved the best R² score of ~0.89

### Key Findings
- 🚬 **Smoking** is the single most important predictor (drives up costs 4x)
- ⚖️ **BMI** and **Age** positively correlate with charges
- 🔥 **Smoker + Obese** interaction captures the highest-risk group

---

## 🖥️ How to Run

### Option A: Google Colab (Recommended)
1. Open `notebooks/Medical_Insurance_Cost_Prediction.ipynb` in Google Colab
2. Run all cells top to bottom
3. Download the 3 model files at the end (Step 9 cell)

### Option B: Local Streamlit App
```bash
# 1. Clone the repo
git clone <your-repo-url>
cd insurance_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model files in models/ folder
# (downloaded from Colab Step 9)

# 4. Run Streamlit
streamlit run app.py
```

### Option C: Docker
```bash
docker build -t insurance-predictor .
docker run -p 8501:8501 insurance-predictor
# Open http://localhost:8501
```

---

## 📸 Screenshots

> Add screenshots of your Streamlit app here after running it.

---

## 👤 Author

- **Name:** [Your Name]
- **Roll No:** [Your Roll Number]
- **Dataset:** Medical Insurance Cost Prediction

---

## 📄 License
This project is for academic purposes only.
