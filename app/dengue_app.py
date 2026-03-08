import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Dengue Prediction", page_icon="🦟", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('../models/xgboost_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/dengue_features_final.csv')
    df['month'] = pd.to_datetime(df['month'])
    return df

model = load_model()
df = load_data()

st.sidebar.title("🦟 Dengue Prediction")
page = st.sidebar.radio("Navigation", ["Overview", "Predict", "Data", "About"])

if page == "Overview":
    st.title("Singapore Dengue Prediction System")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Cases/Month", f"{df['cases'].mean():.0f}")
    col2.metric("Peak Month", "June")
    col3.metric("Model R²", "0.725")
    
    st.subheader("📈 Dengue Cases Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['month'], y=df['cases'], mode='lines', name='Cases'))
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Cases")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📅 Seasonal Pattern")
    monthly = df.groupby('month_num')['cases'].mean()
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=months, y=monthly.values))
    fig2.update_layout(height=400, xaxis_title="Month", yaxis_title="Avg Cases")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Predict":
    st.title("🔮 Predict Dengue Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        month_num = st.slider("Month (1=Jan, 12=Dec)", 1, 12, 6)
        rainfall = st.slider("Rainfall (mm)", 0, 700, 200)
        temperature = st.slider("Temperature (°C)", 24.0, 27.0, 26.0, 0.1)
        cases_lag_1 = st.number_input("Cases Last Month", 0, 7000, 1000)
    
    with col2:
        cases_lag_2 = st.number_input("Cases 2 Months Ago", 0, 7000, 900)
        cases_lag_3 = st.number_input("Cases 3 Months Ago", 0, 7000, 800)
        rainfall_lag_1 = st.slider("Rainfall Last Month", 0, 700, 180)
        temp_lag_1 = st.slider("Temp Last Month", 24.0, 27.0, 25.8, 0.1)
    
    quarter = (month_num - 1) // 3 + 1
    cases_rolling_3 = (cases_lag_1 + cases_lag_2 + cases_lag_3) / 3
    temp_rain_product = temp_lag_1 * rainfall_lag_1
    
    if st.button("🔍 Predict", type="primary"):
        input_data = pd.DataFrame({
            'cases_lag_1': [cases_lag_1],
            'cases_lag_2': [cases_lag_2],
            'cases_lag_3': [cases_lag_3],
            'cases_rolling_3': [cases_rolling_3],
            'rainfall': [rainfall],
            'rainfall_lag_1': [rainfall_lag_1],
            'temperature': [temperature],
            'temp_lag_1': [temp_lag_1],
            'temp_rain_product': [temp_rain_product],
            'month_num': [month_num],
            'quarter': [quarter]
        })
        
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        if prediction < 1000:
            st.success(f"🟢 LOW RISK: {prediction:.0f} cases predicted")
        elif prediction < 2000:
            st.warning(f"🟡 MODERATE RISK: {prediction:.0f} cases predicted")
        else:
            st.error(f"🔴 HIGH RISK: {prediction:.0f} cases predicted")

elif page == "Data":
    st.title("📊 Historical Data")
    
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start Date", df['month'].min())
    with col2:
        end = st.date_input("End Date", df['month'].max())
    
    mask = (df['month'] >= pd.to_datetime(start)) & (df['month'] <= pd.to_datetime(end))
    filtered = df[mask]
    
    st.metric("Total Months", len(filtered))
    
    display = filtered[['month', 'cases', 'rainfall', 'temperature']].copy()
    display['month'] = display['month'].dt.strftime('%Y-%m')
    st.dataframe(display, use_container_width=True, height=400)
    
    csv = display.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "dengue_data.csv", "text/csv")

else:
    st.title("ℹ️ About")
    st.markdown("""
    ### Singapore Dengue Prediction System
    
    **Model:** XGBoost Regressor
    - Test R²: 0.725
    - Test MAE: 521 cases
    - Beats baseline by 2.6%
    
    **Data Sources:**
    - Dengue cases: data.gov.sg
    - Weather: data.gov.sg
    - Period: 2012-2023
    
    **Top Features:**
    1. Cases from last month (26%)
    2. 3-month rolling average (20%)
    3. Cases from 3 months ago (14%)
    4. Rainfall lag (9%)
    5. Temperature lag (9%)
    
    ---
    *Educational project - March 2025*
    """)
