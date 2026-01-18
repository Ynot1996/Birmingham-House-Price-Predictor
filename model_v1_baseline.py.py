import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the processed feature files
train_df = pd.read_csv('train_features_2024.csv')
test_df = pd.read_csv('test_features_2025.csv')

# 2. Alignment: Ensure both datasets have the exact same columns
# If a postcode exists in 2024 but not 2025 (or vice versa), the model will fail
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

# 3. Define Features (X) and Target (y)
X_train = train_df.drop('Price', axis=1)
y_train = train_df['Price']
X_test = test_df.drop('Price', axis=1)
y_test = test_df['Price']

print(f"Training on {len(X_train)} records (2024)...")
print(f"Testing on {len(X_test)} records (2025)...")

# 4. Train Random Forest Model
# This model handles non-linear relationships well (e.g., location vs price)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make Predictions
predictions = model.predict(X_test)

# 6. Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"Model Performance on 2025 Data:")
print(f"Average Error (MAE): £{mae:.2f}")
print(f"Model Reliability (R2): {r2:.4f}")

# 7. Visualization: Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (£)')
plt.ylabel('Predicted Price (£)')
plt.title('Birmingham House Price Prediction: Actual vs Predicted (2025)')
plt.savefig('prediction_results.png')
plt.show()