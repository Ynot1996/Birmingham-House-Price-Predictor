import pandas as pd
import numpy as np


def clean_and_feature_engineering(file_path):
    # Load the filtered real data
    df = pd.read_csv(file_path)

    # 1. Extract Postcode Area (e.g., B1, B29)
    # Some postcodes might be missing, we drop them for accuracy
    df = df.dropna(subset=['Postcode'])
    df['Postcode_Area'] = df['Postcode'].apply(lambda x: str(x).split(' ')[0])

    # 2. Convert Date to Month (Seasonality factor)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    # 3. Select relevant features for the model
    # Price is our target, others are features
    required_cols = ['Price', 'Postcode_Area', 'Type', 'Old_New', 'Duration', 'Month']
    df_clean = df[required_cols]

    # 4. One-Hot Encoding for categorical text data
    # This turns 'Type' into multiple columns of 0s and 1s
    df_final = pd.get_dummies(df_clean, columns=['Postcode_Area', 'Type', 'Old_New', 'Duration'])

    return df_final


if __name__ == "__main__":
    print("Processing 2024 training data...")
    train_data = clean_and_feature_engineering('birmingham_prices_real_2024.csv')
    train_data.to_csv('train_features_2024.csv', index=False)

    print("Processing 2025 testing data...")
    test_data = clean_and_feature_engineering('birmingham_prices_real_2025.csv')
    # Important: Ensure 2025 data has the same columns as 2024
    test_data.to_csv('test_features_2025.csv', index=False)

    print("Feature Engineering Complete!")