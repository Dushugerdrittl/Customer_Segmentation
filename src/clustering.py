import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Paths for cleaned data and the trained model
input_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\processed\customer_data_cleaned.csv"
output_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\processed\clustered_data.csv"
model_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\src\kmeans_model.pkl"
required_features_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\processed\required_features.pkl"

def perform_clustering(input_path, output_path, model_path, required_features_path, n_clusters=4):
    """
    Performs clustering using KMeans algorithm:
    - Loads the preprocessed data
    - Trains the KMeans model
    - Saves the model
    - Adds cluster labels to the data
    - Saves the clustered data
    """
    # Load preprocessed data
    data = pd.read_csv(input_path)

    # Specify only the features that are used as input for the clustering
    important_features = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "TENURE"]  # These should match the input fields in Streamlit

    # Exclude non-numerical features that should not be part of clustering
    numerical_cols = data[important_features].select_dtypes(include='number').columns.tolist()

    # Initialize KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Perform KMeans clustering
    data['Cluster'] = kmeans.fit_predict(data[numerical_cols])

    # Save the trained model
    joblib.dump(kmeans, model_path)

    # Save the clustered data to a CSV file (include 'Cluster' and numerical features)
    clustered_data = data[numerical_cols + ['Cluster']]  # Ensure 'Cluster' is included
    clustered_data.to_csv(output_path, index=False)

    # Save the list of required features (numerical columns)
    joblib.dump(important_features, required_features_path)
    
    # Evaluate clustering performance (optional)
    score = silhouette_score(data[numerical_cols], data['Cluster'])
    print(f"Silhouette Score: {score}")
    return score

# Call the function to perform clustering
if __name__ == "__main__":
    perform_clustering(input_path, output_path, model_path, required_features_path)
