import pandas as pd
import numpy as np


def clean_and_feature_engineering_v2_1(train_path, test_path):
    """
    Refined Feature Engineering:
    - Removes 'Month' to reduce seasonal noise.
    - Creates 'Area_Type_Avg' as a high-precision proxy for house size/value.
    - Resolves SettingWithCopyWarning using .copy().
    """
    # 1. Load raw datasets
    train_df_raw = pd.read_csv(train_path)
    test_df_raw = pd.read_csv(test_path)

    # 2. Filter Outliers (Mainstream market <= £1M)
    train_df = train_df_raw[train_df_raw['Price'] <= 1000000].copy()
    test_df = test_df_raw[test_df_raw['Price'] <= 1000000].copy()

    def preprocess_base(df):
        # Extract Postcode Area
        df['Postcode_Area'] = df['Postcode'].apply(lambda x: str(x).split(' ')[0])
        return df

    train_df = preprocess_base(train_df)
    test_df = preprocess_base(test_df)

    # ---------------------------------------------------------
    # 3. COMPOSITE TARGET ENCODING: Area + Property Type
    # This acts as a proxy for size (e.g., B15 Detached vs B15 Flat)
    # ---------------------------------------------------------
    # Calculate means based ONLY on 2024 training data
    lookup = train_df.groupby(['Postcode_Area', 'Type'])['Price'].mean()

    # Map the composite averages back using a multi-index join logic
    train_df['Area_Type_Avg'] = train_df.set_index(['Postcode_Area', 'Type']).index.map(lookup)
    test_df['Area_Type_Avg'] = test_df.set_index(['Postcode_Area', 'Type']).index.map(lookup)

    # Fill missing values for combinations not seen in 2024
    global_mean = train_df['Price'].mean()
    train_df['Area_Type_Avg'] = train_df['Area_Type_Avg'].fillna(global_mean)
    test_df['Area_Type_Avg'] = test_df['Area_Type_Avg'].fillna(global_mean)

    # 4. Feature Selection (REMOVING 'Month')
    required_cols = ['Price', 'Area_Type_Avg', 'Postcode_Area', 'Type', 'Old_New', 'Duration']
    train_clean = train_df[required_cols]
    test_clean = test_df[required_cols]

    # Categorical encoding
    categorical_features = ['Postcode_Area', 'Type', 'Old_New', 'Duration']
    train_final = pd.get_dummies(train_clean, columns=categorical_features)
    test_final = pd.get_dummies(test_clean, columns=categorical_features)

    return train_final, test_final


if __name__ == "__main__":
    print("Starting Feature Engineering V2.1...")
    train_features, test_features = clean_and_feature_engineering_v2_1('birmingham_prices_real_2024.csv',
        'birmingham_prices_real_2025.csv')

    train_features.to_csv('train_features_v2_1.csv', index=False)
    test_features.to_csv('test_features_v2_1.csv', index=False)

    print("-" * 30)
    print("Success! V2.1 Features saved (Month removed, Area_Type_Avg added).")