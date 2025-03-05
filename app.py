import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import xgboost as xgb  # Ensure xgboost is imported before loading the model

# Ensure Render-assigned PORT is used
PORT = int(os.environ.get("PORT", 8501))

# Load the trained model
model_path = "xgboost_model.pkl"  # Use relative path
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    st.error("Model file not found!")
    st.stop()

# Load the dataset
data_path = "banglore_price_prediction.csv"  # Relative path for Render compatibility
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.error("Dataset file not found!")
    st.stop()

# Drop target variable and unwanted columns
columns_to_drop = ["price", "Unnamed: 0"]  # Removed duplicate column
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Extract categorical and numerical columns
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove unwanted input columns
numerical_columns = [col for col in numerical_columns if col not in ["Unnamed: 0", "price_per_sqft"]]

# Streamlit App
st.title("Bangalore House Price Prediction")

# Show dataset preview (without one-hot encoding)
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

# User input fields
st.subheader("Enter Details")

# Location selection (assuming 'location' is a categorical column)
if "location" in categorical_columns:
    locations = df["location"].unique().tolist()
    selected_location = st.selectbox("Select Location", locations)
else:
    st.error("Location column not found in dataset!")
    st.stop()

# Manual entry for numerical features (excluding target variable)
user_input = {}
for feature in numerical_columns:
    user_input[feature] = st.number_input(f"Enter {feature}", min_value=0.0, format="%.2f")

# Prepare final input dataframe
input_df = pd.DataFrame([user_input])
input_df.insert(0, "location", selected_location)

# Convert categorical data using same encoding as training
input_df = pd.get_dummies(input_df)

# Align columns with model training (fill missing columns with 0)
model_features = model.get_booster().feature_names or []  # Fix potential NoneType issue
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure column order matches model training
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹{prediction:,.2f}")
