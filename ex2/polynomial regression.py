import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

df = pd.read_csv(url)

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

X = df[['displacement']]
y = df['mpg']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Linear Regression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

print("\nLinear Regression")
print("-------------------")
print("MSE :", linear_mse)
print("R2 Score :", linear_r2)


# Polynomial Regression
degrees = [2, 3, 4]

results = []

for degree in degrees:
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])

    poly_model.fit(X_train, y_train)

    y_pred = poly_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([degree, mse, r2])

    print(f"\nPolynomial Degree {degree}")
    print("-------------------------")
    print("MSE :", mse)
    print("R2 Score :", r2)



 # Sort values for smooth plotting
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1,1)

plt.figure(figsize=(10,6))

# Scatter Plot
plt.scatter(X, y, color='gray', alpha=0.6, label='Data')

# Linear Regression Line
y_line = linear_model.predict(X_range)
plt.plot(X_range, y_line,
         color='blue',
         linewidth=2,
         label='Linear Regression') 


#Polynomial Curves
colors = ['red', 'green', 'orange']

for degree, color in zip(degrees, colors):

    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])

    poly_model.fit(X, y)

    y_poly = poly_model.predict(X_range)

    plt.plot(X_range,
             y_poly,
             color=color,
             linewidth=2,
             label=f'Polynomial Degree {degree}')

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear vs Polynomial Regression on Auto MPG Dataset")
plt.legend()
plt.grid(True)

plt.show()





