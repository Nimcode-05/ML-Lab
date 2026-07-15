import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------
# Step 1: Load the California Housing Dataset and preprocess

housing = fetch_california_housing(as_frame=True)

# Create DataFrame
data = housing.frame

# Display first five rows
print(data.head())


print("\nDataset Shape:", data.shape)

print("\nData Types")
print(data.dtypes)

print("\nMissing Values")
print(data.isnull().sum())

print("\nDuplicate Rows:", data.duplicated().sum())

# Remove duplicates if any
data = data.drop_duplicates()

print("\nStatistical Summary")
print(data.describe())

# ------------------------------------------------
# Detect Outliers using IQR Method

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)

IQR = Q3 - Q1

outliers = ((data < (Q1 - 1.5 * IQR)) |
            (data > (Q3 + 1.5 * IQR)))

print("\nNumber of Outliers")
print(outliers.sum())

# ---------------------------------------------------
# Step 2: Select Feature and Target
# Feature: Average number of rooms

X = data[['AveRooms']]

# Target: Median house value
y = data['MedHouseVal']

# ---------------------------------------------------
# Step 3: Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# ---------------------------------------------------
# Step 4: Create and Train the Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)

# ---------------------------------------------------
# Step 5: Predict on Test Data
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# Step 6: Evaluate the Model
# ---------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("Intercept :", model.intercept_)
print("Coefficient :", model.coef_[0])
print("Mean Squared Error :", mse)
print("R² Score :", r2)

# ---------------------------------------------------
# Step 7: Plot the Regression Line
# ---------------------------------------------------
plt.figure(figsize=(8,6))

# Scatter plot of actual data
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')

# Sort data for a smooth regression line
sorted_index = X_test['AveRooms'].argsort()

plt.plot(
    X_test.iloc[sorted_index],
    y_pred[sorted_index],
    color='red',
    linewidth=2,
    label='Regression Line'
)

plt.xlim(0, 15),

plt.xlabel("Average Number of Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.title("Linear Regression using One Feature")
plt.legend()
plt.grid(True)

plt.show()
