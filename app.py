import streamlit as st
import pandas as pd
import joblib
import numpy as np
from src.evaluation import evaluate_clustering, print_clustering_summary  # Import evaluation functions

# --- Load Pre-trained KMeans Model and Scaler ---
# Correct path for pre-trained scaler
model_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\src\kmeans_model.pkl"
scaler_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\src\kmeans_model.pkl"  # Fixed path
required_features_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\processed\required_features.pkl"

# Load the KMeans model
model = joblib.load(model_path)

# Load the StandardScaler
scaler = joblib.load(scaler_path)

# Load required features from training
required_features = joblib.load(required_features_path)

# --- Streamlit Interface ---
st.title("Customer Segmentation with KMeans Clustering")
st.write("Enter the customer's details below and click **Predict Cluster** to find their segment.")

# --- User Input Section ---
# Descriptions for input features
feature_descriptions = {
    "BALANCE": "Current outstanding balance on the customer's account.",
    "PURCHASES": "Total amount of purchases made by the customer.",
    "ONEOFF_PURCHASES": "Total value of one-time purchases.",
    "TENURE": "Number of months the customer has been with the company."
}

# Define important input features for user input
important_features = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "TENURE"]

# Create input fields for the user
with st.form(key='user_input_form'):
    user_data = {}
    for feature in important_features:
        user_data[feature] = st.number_input(
            f"{feature} ({feature_descriptions[feature]})", 
            min_value=0.0, 
            step=0.1
        )
    
    # Create the Predict button
    submit_button = st.form_submit_button(label='Predict Cluster')

# --- Ensure Required Features ---
if submit_button:
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_data])

    # Add missing features with default value 0 and ensure correct column order
    for feature in required_features:
        if feature not in user_df.columns:
            user_df[feature] = 0  # Default value for missing features

    # Reorder columns to match the training order
    user_df = user_df[required_features]

    # --- Debug: Display Data ---
    st.write("Data being passed to the model:")
    st.write(user_df)

    # --- Preprocess Input Data ---
    user_data_scaled = scaler.transform(user_df)

    # --- Predict Customer Segment ---
    cluster = model.predict(user_data_scaled)

    # --- Display Results ---
    st.subheader("Predicted Customer Segment")
    st.success(f"The predicted cluster for this customer is: **Cluster {cluster[0]}**")

    # --- Optional: Cluster Descriptions ---
    st.subheader("Cluster Descriptions")
    st.info("""
    - **Cluster 0**: High-value customers with large purchases and balanced spending.
    - **Cluster 1**: Moderate spenders with steady activity.
    - **Cluster 2**: Customers with low purchases and short tenure.
    - **Cluster 3**: Customers primarily making one-off large purchases.
    """)

    # --- Evaluate Clustering Performance ---
    # Assuming 'df_full_data' contains the dataset with clusters and necessary features
    # For demonstration purposes, we'll simulate the data with just this user's input
    df_full_data = user_df.copy()
    df_full_data['Cluster'] = cluster[0]  # Only set the predicted cluster for the single input

    # --- Evaluate clustering performance ---
    silhouette_score_value = evaluate_clustering(df_full_data)
    st.subheader("Clustering Evaluation")
    st.write(f"Silhouette Score for clustering: **{silhouette_score_value}**")

    # --- Print clustering summary ---
    st.subheader("Clustering Summary")
    print_clustering_summary(df_full_data)
