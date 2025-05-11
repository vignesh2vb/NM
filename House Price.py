
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
data = pd.read_csv('/content/house_data.csv')
print("✅ Dataset loaded.\n")

# Step 2: Feature Engineering + Handling Missing Values
data = data.copy()

# Derived features
data['TotalBaths'] = data['Full Bath'] + 0.5 * data['Half Bath']
data['HouseAge'] = data['Yr Sold'] - data['Year Built']
data['RemodAge'] = data['Yr Sold'] - data['Year Remod/Add']
data['PricePerSqft'] = data['SalePrice'] / data['Gr Liv Area']

# Selected features
features = [
    'Overall Qual', 'Gr Liv Area', 'Total Bsmt SF', 'Garage Area',
    'Bedroom AbvGr', 'TotalBaths', 'HouseAge', 'RemodAge', 'PricePerSqft'
]
target = 'SalePrice'

# Drop rows with missing values in any selected column
data = data.dropna(subset=features + [target])

# Step 3: Define X and y
X = data[features]
y = data[target]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation Metrics:")
print(f"  • MAE :  {mae:,.2f}")
print(f"  • RMSE:  {rmse:,.2f}")
print(f"  • R²   :  {r2:.4f}\n")

# Step 9: Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, '--', color='gray')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 10: Coefficients
coeffs = model.coef_
importance_df = pd.DataFrame({'Feature': features, 'Coefficient': coeffs})
print("📌 Feature Coefficients:")
print(importance_df.sort_values(by='Coefficient', key=abs, ascending=False))

# Step 11: Save processed dataset
data.to_csv('/content/house_data.csv', index=False)
print("✅ Processed data saved to '/content/house_data.csv'")
