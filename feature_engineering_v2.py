import pandas as pd
import numpy as np


def clean_and_feature_engineering(train_path, test_path):
    """
    V2 Feature Engineering:
    - Resolves SettingWithCopyWarning using .copy()
    - Implements Target Encoding (Area_Avg_Price) after outlier filtering
    - Strengthens geographic signals for the Birmingham mainstream market
    """
    # 1. Load raw datasets
    train_df_raw = pd.read_csv(train_path)
    test_df_raw = pd.read_csv(test_path)

    # 2. Filter Outliers (Mainstream market <= £1M) and use .copy()
    # This ensures we work on an independent DataFrame and avoid warnings
    train_df = train_df_raw[train_df_raw['Price'] <= 1000000].copy()
    test_df = test_df_raw[test_df_raw['Price'] <= 1000000].copy()

    def preprocess_base(df):
        # Extract Postcode Area (e.g., B15, B29)
        df['Postcode_Area'] = df['Postcode'].apply(lambda x: str(x).split(' ')[0])

        # Convert Date to Month for seasonality factors
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        return df

    # Apply base preprocessing to both sets
    train_df = preprocess_base(train_df)
    test_df = preprocess_base(test_df)

    # ---------------------------------------------------------
    # 3. TARGET ENCODING: Area Average Price
    # ---------------------------------------------------------
    # Calculate means based ONLY on 2024 training data to prevent data leakage
    # We calculate this after filtering to reflect the true mainstream market average
    area_lookup = train_df.groupby('Postcode_Area')['Price'].mean()

    # Map the 2024 averages back to both 2024 and 2025 datasets
    train_df['Area_Avg_Price'] = train_df['Postcode_Area'].map(area_lookup)
    test_df['Area_Avg_Price'] = test_df['Postcode_Area'].map(area_lookup)

    # Handle new postcodes in 2025 not found in 2024 (fill with global mean)
    global_mean = train_df['Price'].mean()
    test_df['Area_Avg_Price'] = test_df['Area_Avg_Price'].fillna(global_mean)

    # 4. Feature Selection and One-Hot Encoding
    required_cols = ['Price', 'Area_Avg_Price', 'Postcode_Area', 'Type', 'Old_New', 'Duration', 'Month']
    train_clean = train_df[required_cols]
    test_clean = test_df[required_cols]

    # Convert categorical text into numerical binary columns
    categorical_features = ['Postcode_Area', 'Type', 'Old_New', 'Duration']
    train_final = pd.get_dummies(train_clean, columns=categorical_features)
    test_final = pd.get_dummies(test_clean, columns=categorical_features)

    return train_final, test_final


if __name__ == "__main__":
    print("Starting Feature Engineering V2...")

    # Process both datasets to ensure consistency
    train_features, test_features = clean_and_feature_engineering('birmingham_prices_real_2024.csv',
        'birmingham_prices_real_2025.csv')

    # Save to V2 CSV files
    train_features.to_csv('train_features_v2.csv', index=False)
    test_features.to_csv('test_features_v2.csv', index=False)

    print("-" * 30)
    print("Success! V2 Features saved to 'train_features_v2.csv' and 'test_features_v2.csv'.")
    print("New feature 'Area_Avg_Price' added and Warning 'SettingWithCopy' resolved.")