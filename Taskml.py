import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Generate Dummy Data
num_days = 365
date_range = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
np.random.seed(42)

data = {
    'date': date_range,
    'electricity_usage': np.random.normal(loc=1000, scale=200, size=num_days), # kWh
    'water_consumption': np.random.normal(loc=500, scale=100, size=num_days), # Cubic meters
    'waste_collected': np.random.normal(loc=50, scale=10, size=num_days), # Tons
    'traffic_flow': np.random.normal(loc=2000, scale=400, size=num_days), # Number of vehicles
    'emergency_calls': np.random.poisson(lam=5, size=num_days) # Number of calls
}

df = pd.DataFrame(data)

# Introduce anomalies
anomalies = np.random.choice(df.index, size=10, replace=False)
df.loc[anomalies, 'electricity_usage'] *= np.random.uniform(1.5, 2.0, size=10)
df.loc[anomalies, 'water_consumption'] *= np.random.uniform(1.5, 2.0, size=10)
df.loc[anomalies, 'waste_collected'] *= np.random.uniform(1.5, 2.0, size=10)
df.loc[anomalies, 'traffic_flow'] *= np.random.uniform(1.5, 2.0, size=10)
df.loc[anomalies, 'emergency_calls'] *= np.random.uniform(1.5, 2.0, size=10)

# Step 2: Data Preprocessing
df.fillna(method='ffill', inplace=True)

# Step 3: Feature Engineering
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['day_of_year'] = df['date'].dt.dayofyear

for col in ['electricity_usage', 'water_consumption', 'waste_collected', 'traffic_flow', 'emergency_calls']:
    for lag in range(1, 8):  # 1 to 7 days lag
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
    df[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()

df.dropna(inplace=True)

# Step 4: Build Predictive Models
features = df.drop(columns=['date', 'electricity_usage', 'water_consumption', 'waste_collected', 'traffic_flow', 'emergency_calls'])
target = df[['electricity_usage', 'water_consumption', 'waste_collected', 'traffic_flow', 'emergency_calls']]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions for evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Anomaly Detection
iso_forest = IsolationForest(contamination=0.01, random_state=42)
anomalies = iso_forest.fit_predict(features)

df['anomaly'] = anomalies
df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Streamlit App
st.title("City Resource Needs Forecasting and Anomaly Detection")

st.write(f"Model Evaluation Metrics: Mean Absolute Error = {mae:.2f}, R2 Score = {r2:.2f}")

# User inputs for prediction
st.header("Enter Values for Prediction")
day_of_week = st.slider("Day of Week", 0, 6, 0)
month = st.slider("Month", 1, 12, 1)
quarter = (month - 1) // 3 + 1
day_of_year = st.slider("Day of Year", 1, 365, 1)

input_data = {
    'day_of_week': day_of_week,
    'month': month,
    'quarter': quarter,
    'day_of_year': day_of_year
}

for col in ['electricity_usage', 'water_consumption', 'waste_collected', 'traffic_flow', 'emergency_calls']:
    for lag in range(1, 8):
        input_data[f'{col}_lag_{lag}'] = st.number_input(f'{col} lag {lag}', value=0.0)
    input_data[f'{col}_rolling_mean_7'] = st.number_input(f'{col} rolling mean 7', value=0.0)
    input_data[f'{col}_rolling_std_7'] = st.number_input(f'{col} rolling std 7', value=0.0)

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

# Display prediction
st.header("Prediction Results")
st.write("Electricity Usage (kWh): ", prediction[0][0])
st.write("Water Consumption (Cubic meters): ", prediction[0][1])
st.write("Waste Collected (Tons): ", prediction[0][2])
st.write("Traffic Flow (Number of vehicles): ", prediction[0][3])
st.write("Emergency Calls: ", prediction[0][4])
