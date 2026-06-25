import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥", layout="centered")

@st.cache_resource
def train_and_load():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from xgboost import XGBRegressor

    url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
    df = pd.read_csv(url)

    df['bmi_category'] = pd.cut(df['bmi'], bins=[0,18.5,24.9,29.9,100],
                                 labels=['Underweight','Normal','Overweight','Obese']).astype(str)
    df['age_group'] = pd.cut(df['age'], bins=[0,25,40,55,100],
                              labels=['Young','Middle','Senior','Elderly']).astype(str)
    df['family_size'] = pd.cut(df['children'], bins=[-1,0,2,10],
                                labels=['No_children','Small_family','Large_family']).astype(str)
    df['smoker_obese'] = ((df['smoker']=='yes') & (df['bmi']>=30)).astype(int)
    df['age_bmi'] = df['age'] * df['bmi']

    X = df.drop(columns=['charges'])
    y = df['charges']

    numerical_features   = ['age','bmi','children','age_bmi','smoker_obese']
    categorical_features = ['sex','smoker','region','bmi_category','age_group','family_size']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_proc = preprocessor.fit_transform(X_train)

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X_train_proc, y_train)

    return model, preprocessor, numerical_features, categorical_features


st.markdown("## 🏥 Medical Insurance Cost Predictor")
st.markdown("Estimate your annual insurance charges in USD and INR")

with st.spinner("Loading model... (first load takes ~30 seconds)"):
    model, preprocessor, numerical_features, categorical_features = train_and_load()

st.success("Model ready!")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age      = st.slider("Age", 18, 65, 30)
    bmi      = st.number_input("BMI", 10.0, 55.0, 27.0, 0.1)
    children = st.selectbox("Children", [0,1,2,3,4,5])
with col2:
    sex    = st.selectbox("Sex", ["male","female"])
    smoker = st.selectbox("Smoker?", ["no","yes"])
    region = st.selectbox("Region", ["northeast","northwest","southeast","southwest"])

bmi_label = "Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese"
st.caption(f"BMI Category: **{bmi_label}**")

if st.button("Predict Insurance Cost"):
    bmi_category = "Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese"
    age_group    = "Young" if age<=25 else "Middle" if age<=40 else "Senior" if age<=55 else "Elderly"
    family_size  = "No_children" if children==0 else "Small_family" if children<=2 else "Large_family"
    smoker_obese = int(smoker=="yes" and bmi>=30)
    age_bmi      = age * bmi

    input_df = pd.DataFrame([{
        "age":age, "bmi":bmi, "children":children,
        "age_bmi":age_bmi, "smoker_obese":smoker_obese,
        "sex":sex, "smoker":smoker, "region":region,
        "bmi_category":bmi_category, "age_group":age_group, "family_size":family_size
    }])

    processed  = preprocessor.transform(input_df)
    prediction = model.predict(processed)[0]
    inr        = prediction * 83.5

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1e3a5f,#2e6da4);border-radius:16px;
    padding:2rem;text-align:center;color:white;margin-top:1rem'>
    <div style='font-size:0.9rem;opacity:0.8;text-transform:uppercase'>Estimated Annual Cost</div>
    <div style='font-size:3rem;font-weight:800'>${prediction:,.2f} USD</div>
    <div style='font-size:1.6rem;font-weight:600;margin-top:0.3rem'>&#8377;{inr:,.0f} INR</div>
    <div style='opacity:0.7;margin-top:0.5rem'>Per month: ${prediction/12:,.2f} USD &nbsp;|&nbsp; &#8377;{inr/12:,.0f} INR</div>
    </div>
    """, unsafe_allow_html=True)

    if smoker == "yes":
        st.warning("Smoking is the biggest cost driver — 4x higher charges!")
    if bmi >= 30:
        st.info("Obese BMI range detected — contributes to higher premiums.")

st.markdown("---")
st.caption("This is an ML prediction, not a real insurance quote.")
