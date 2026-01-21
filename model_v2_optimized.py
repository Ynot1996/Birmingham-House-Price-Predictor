import matplotlib

# Force the backend to TkAgg for macOS compatibility to ensure the window pops up
matplotlib.use('TkAgg')

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
# This handles the skewness issue you identified in V1
train_df = train_df[train_df['Price'] <= 1000000]
test_df = test_df[test_df['Price'] <= 1000000]

# 3. Alignment
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

# 4. Feature/Target Split & Log Transformation
# Normalizing the distribution to improve R2 score
X_train = train_df.drop('Price', axis=1)
y_train_log = np.log1p(train_df['Price'])

X_test = test_df.drop('Price', axis=1)
y_test_real = test_df['Price']

# 5. Train Model
print("Training Optimized Model (V2) with Log Transformation...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train_log)

# 6. Predict and Convert Back from Log
predictions_log = model.predict(X_test)
predictions_real = np.expm1(predictions_log)  # Convert log back to GBP

# 7. Evaluation
mae = mean_absolute_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)

print("-" * 30)
print(f"V2 Performance (Under £1M Market):")
print(f"Average Error (MAE): £{mae:.2f}")
print(f"Model Reliability (R2): {r2:.4f}")


# ---------------------------------------------------------
# 8. NEW: Feature Importance Visualization
# ---------------------------------------------------------
def plot_feature_importance(model, X_train):
    importances = model.feature_importances_
    feature_names = X_train.columns

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(
        by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='magma', hue='Feature',
        legend=False)
    plt.title('Top 15 Influential Factors - Birmingham Optimized Model', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance_v2.png')
    plt.show(block=True)


# Execute the new visualization
plot_feature_importance(model, X_train)

# ---------------------------------------------------------
# 9. Visualization: Actual vs Predicted
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(y_test_real, predictions_real, alpha=0.3, color='green')
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
plt.xlabel('Actual Price (£)')
plt.ylabel('Predicted Price (£)')
plt.title('V2 Optimized: Actual vs Predicted (Mainstream Market)')
plt.savefig('results_v2_optimized.png')
plt.show(block=True)