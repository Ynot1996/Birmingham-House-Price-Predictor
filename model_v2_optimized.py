import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
train_df = pd.read_csv('train_features_2024.csv')
test_df = pd.read_csv('test_features_2025.csv')

# 2. IMPORTANT: Filter Outliers (Only focus on properties <= £1,000,000)
# We do this to prevent luxury estates from skewing our mainstream model
train_df = train_df[train_df['Price'] <= 1000000]
test_df = test_df[test_df['Price'] <= 1000000]

# 3. Alignment
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

# 4. Feature/Target Split & Log Transformation
# We predict log(Price) to normalize the distribution
X_train = train_df.drop('Price', axis=1)
y_train_log = np.log1p(train_df['Price'])

X_test = test_df.drop('Price', axis=1)
y_test_real = test_df['Price'] # Keep real price for final evaluation

# 5. Train Model
print("Training Optimized Model (V2)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train_log)

# 6. Predict and Convert Back from Log
predictions_log = model.predict(X_test)
predictions_real = np.expm1(predictions_log) # Convert log back to GBP

# 7. Evaluation
mae = mean_absolute_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)

print("-" * 30)
print(f"V2 Performance (Under £1M Market):")
print(f"Average Error (MAE): £{mae:.2f}")
print(f"Model Reliability (R2): {r2:.4f}")

# 8. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test_real, predictions_real, alpha=0.3, color='green')
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
plt.xlabel('Actual Price (£)')
plt.ylabel('Predicted Price (£)')
plt.title('V2 Optimized: Actual vs Predicted (Mainstream Market)')
plt.savefig('results_v2_optimized.png')
plt.show()