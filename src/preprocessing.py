import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    print("Loading raw data...")
    try:
        data = pd.read_csv(input_path)
        print(f"Data loaded successfully! Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Handle missing values (only for numeric columns)
    print("Handling missing values...")
    numeric_cols = data.select_dtypes(include='number').columns
    for col in numeric_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # Feature engineering (same as before)
    print("Feature engineering...")
    data['AVG_PURCHASE_VALUE'] = data['PURCHASES'] / (data['PURCHASES_TRX'] + 1e-6)
    data['LIMIT_USAGE'] = data['BALANCE'] / data['CREDIT_LIMIT']

    # Scale numeric columns
    print("Scaling numeric columns...")
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include='number').columns
    data_scaled = scaler.fit_transform(data[numeric_cols])

    # Create a new dataframe with the scaled data
    data_scaled_df = pd.DataFrame(data_scaled, columns=numeric_cols)

    # Add the non-numeric columns back (e.g., Customer ID)
    non_numeric_cols = data.select_dtypes(exclude='number').columns
    data_scaled_df[non_numeric_cols] = data[non_numeric_cols]

    # Save the cleaned data
    print(f"Saving cleaned data to {output_path}...")
    try:
        data_scaled_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

# Example usage
if __name__ == "__main__":
    input_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\raw\customer_data.csv"
    output_path = r"C:\Users\praveen\OneDrive\Attachments\Desktop\FinalProj\data\processed\customer_data_cleaned.csv"
    
    preprocess_data(input_path, output_path)
