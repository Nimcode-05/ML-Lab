
# Multiple Linear Regression on California Housing Dataset


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


housing = fetch_california_housing(as_frame=True)
data = housing.frame


# 3. Select Multiple Features

X = data[[
    'MedInc',
    'HouseAge',
    'AveRooms',
    'AveBedrms',
    'Population',
    'AveOccup',
    'Latitude',
    'Longitude'
]]

y = data['MedHouseVal']


# 4. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# ---------------------------------------------------
# 5. Missing Value Imputation
# ---------------------------------------------------

imputer = SimpleImputer(strategy='mean')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ---------------------------------------------------
# 6. Feature Scaling
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# 7. Train Model
# ---------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# ---------------------------------------------------
# 8. Predictions
# ---------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------

print("\nIntercept")
print(model.intercept_)

print("\nCoefficients")

feature_names = X.columns

for name, coef in zip(feature_names, model.coef_):
    print(f"{name:12s}: {coef:.4f}")

print("\nModel Performance")

print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²  :", r2_score(y_test, y_pred))

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
    color='blue'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")

plt.title("Actual vs Predicted House Prices")

plt.grid(True)




coef = pd.Series(model.coef_, index=feature_names)

coef.sort_values().plot(
    kind='barh',
    figsize=(8,5),
    color='skyblue'
)

plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.title("Linear Regression Coefficients")

plt.grid(axis='x')

plt.show()
