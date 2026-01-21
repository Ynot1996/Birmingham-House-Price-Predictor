import matplotlib

# Force the backend to TkAgg for macOS compatibility
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load V2.1 Datasets (Month removed, Area_Type_Avg added)
print("Loading V2.1 feature datasets for Model V3.1...")
train_df = pd.read_csv('train_features_v2_1.csv')
test_df = pd.read_csv('test_features_v2_1.csv')

# 2. Alignment: Ensure both datasets have the exact same columns
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

# 3. Define Features and Target (Log Transformation)
# Price is our target; all other columns in X are predictors
X_train = train_df.drop('Price', axis=1)
y_train_log = np.log1p(train_df['Price'])

X_test = test_df.drop('Price', axis=1)
y_test_real = test_df['Price']

# 4. Train Model V3.1 (Strict Hyperparameter Control)
# We limit max_depth to 10 to force the model to learn general patterns,
# not specific outliers.
print("Training Model V3.1 (Random Forest with Composite Encoding)...")
model = RandomForestRegressor(n_estimators=100, max_depth=10,  # Prevents the trees from growing too deep/overfitting
    min_samples_leaf=15,  # Ensures each leaf has enough data for a stable average
    random_state=42)
model.fit(X_train, y_train_log)

# 5. Make Predictions and Revert Log
predictions_log = model.predict(X_test)
predictions_real = np.expm1(predictions_log)

# 6. Evaluation
mae = mean_absolute_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)

print("-" * 30)
print(f"Model V3.1 Performance Statistics:")
print(f"Average Error (MAE): £{mae:.2f}")
print(f"Model Reliability (R2): {r2:.4f}")


# 7. Visualization: Feature Importance
def plot_importance_v3_1(model, X_train):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance',
        ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='rocket', hue='Feature', legend=False)
    plt.title('V3.1 Feature Importance: Area + Type Proxy Dominance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance_v3_1.png')
    print("Success: Saved feature_importance_v3_1.png")
    plt.show(block=True)


# 8. Visualization: Actual vs Predicted
def plot_results_v3_1(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (£)')
    plt.ylabel('Predicted Price (£)')
    plt.title('Model V3.1: Actual vs Predicted Comparison')
    plt.tight_layout()
    plt.savefig('results_v3_1.png')
    print("Success: Saved results_v3_1.png")
    plt.show(block=True)


# Run Visualizations
plot_importance_v3_1(model, X_train)
plot_results_v3_1(y_test_real, predictions_real)