import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

print("Project Started")

# Load Dataset
df = pd.read_csv("sales_data.csv")

print(df.head())

# Features and Target
X = df[['Month']]
y = df['Sales']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)
print("MAE:", mae)

# Future Prediction
future_months = pd.DataFrame({
    'Month':[13,14,15,16]
})

future_sales = model.predict(future_months)

future_df = pd.DataFrame({
    'Month':[13,14,15,16],
    'Predicted Sales':future_sales
})

print("\nFuture Predictions")
print(future_df)

# Save Predictions
future_df.to_csv("predictions.csv", index=False)

# Visualization
plt.figure(figsize=(10,6))

plt.scatter(
    df['Month'],
    df['Sales'],
    label="Actual Sales"
)

plt.plot(
    future_df['Month'],
    future_df['Predicted Sales'],
    marker='o',
    label='Future Prediction'
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting Using Linear Regression")
plt.legend()

plt.show()

print("Project Completed")