import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame

print(data.head(5))

# Check missing values
print(data.isnull().sum())

# Remove duplicates
data = data.drop_duplicates()

# Select one feature
X = data[['AveRooms']].values

# Target
y = data['MedHouseVal'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Missing value handling
imputer = SimpleImputer(strategy='mean')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize parameters

w = 0
b = 0

learning_rate = 0.01

epochs = 1000

n = len(X_train)

loss_history = []

for epoch in range(epochs):

    y_pred = w * X_train.flatten() + b

    dw = (-2/n) * np.sum(X_train.flatten() * (y_train - y_pred))
    db = (-2/n) * np.sum(y_train - y_pred)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    mse = np.mean((y_train - y_pred)**2)
    loss_history.append(mse)

print("Gradient Descent Parameters")
print("Weight:", w)
print("Bias:", b)

y_pred_gd = w * X_test.flatten() + b

print("Gradient Descent Results")

print("MSE :", mean_squared_error(y_test, y_pred_gd))
print("R2 :", r2_score(y_test, y_pred_gd))

plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.title("Gradient Descent Loss Curve")

plt.xlabel("Epoch")

plt.ylabel("MSE")

plt.grid(True)

plt.show()

