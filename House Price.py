import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
data = pd.read_csv('house_data.csv')
print("✅ Dataset loaded. Columns:")
print(data.columns.tolist(), "\n")

# Step 2: Preprocessing
# Drop rows where any of our selected features or target is null
features = ['Lot Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Gr Liv Area']
target = 'SalePrice'
data = data.dropna(subset=features + [target])

# Step 3: Feature Matrix & Target Vector
X = data[features]
y = data[target]

# (Optional) Quick sanity‑check of first five rows:
print("First five rows of X:")
print(X.head(), "\n")
print("First five SalePrice values:")
print(y.head(), "\n")

# Step 4: Train‑Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Step 5: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Evaluation Metrics
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("📊 Model Evaluation:")
print(f"  • MAE:  {mae:,.2f}")
print(f"  • MSE:  {mse:,.2f}")
print(f"  • RMSE: {rmse:,.2f}")
print(f"  • R²:   {r2:.4f}\n")

# Step 8: Visualization (Actual vs Predicted)
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
# draw y=x line for reference
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, '--', color='gray')
plt.tight_layout()
plt.show()

# (Bonus) Save processed dataset for future phases
processed_path = 'house_data_processed.csv'
data.to_csv(processed_path, index=False)
print(f"✅ Processed data saved to '{processed_path}'")
