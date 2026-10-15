import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = datasets.load_iris()

X = iris.data
y = iris.target

print("Feature names:", iris.feature_names)
print("Class names:", iris.target_names)
print("Dataset shape:", X.shape)
# Setosa = 1
# Non-Setosa = 0
y_binary = np.where(y == 0, 1, 0)


# Feature 0 = Sepal Length
# Feature 1 = Sepal Width
X_2d = X[:, [0, 1]]

X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

vm_model = SVC(
    kernel="linear",
    C=1.0
)

svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Setosa", "Setosa"]
    )
)

# Create a mesh over the feature space
x_min = X_train_scaled[:, 0].min() - 1
x_max = X_train_scaled[:, 0].max() + 1
y_min = X_train_scaled[:, 1].min() - 1
y_max = X_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# Predict decision function values
grid = np.c_[xx.ravel(), yy.ravel()]
Z = svm_model.decision_function(grid)
Z = Z.reshape(xx.shape)

# Plot training data
plt.figure(figsize=(10, 7))

plt.scatter(
    X_train_scaled[y_train == 1, 0],
    X_train_scaled[y_train == 1, 1],
    color="blue",
    edgecolor="black",
    s=60,
    label="Setosa"
)

plt.scatter(
    X_train_scaled[y_train == 0, 0],
    X_train_scaled[y_train == 0, 1],
    color="red",
    edgecolor="black",
    s=60,
    label="Non-Setosa"
)

# Decision boundary: w.x + b = 0
plt.contour(
    xx,
    yy,
    Z,
    levels=[0],
    colors="black",
    linewidths=2
)

# Margins: w.x + b = -1 and +1
plt.contour(
    xx,
    yy,
    Z,
    levels=[-1, 1],
    colors="gray",
    linestyles="--",
    linewidths=1.5
)

# Highlight support vectors
plt.scatter(
    svm_model.support_vectors_[:, 0],
    svm_model.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="green",
    linewidths=2,
    label="Support Vectors"
)

plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Sepal Width")
plt.title("Linear SVM: Setosa vs Non-Setosa")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#petal length and width as features

y_binary = np.where(y == 0, 1, 0)

X = iris.data
# For visualization, use:
# Feature 0 = Petal Length
# Feature 1 = Petal  Width
X_2d = X[:, [2, 3]]


X_train, X_test, y_train, y_test = train_test_split(
    X_2d,
    y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svm_model = SVC(
    kernel="linear",
    C=1.0
)

svm_model.fit(X_train_scaled, y_train)


y_pred = svm_model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Setosa", "Setosa"]
    )
)


# Create a mesh over the feature space
x_min = X_train_scaled[:, 0].min() - 1
x_max = X_train_scaled[:, 0].max() + 1
y_min = X_train_scaled[:, 1].min() - 1
y_max = X_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# Predict decision function values
grid = np.c_[xx.ravel(), yy.ravel()]
Z = svm_model.decision_function(grid)
Z = Z.reshape(xx.shape)

# Plot training data
plt.figure(figsize=(10, 7))

plt.scatter(
    X_train_scaled[y_train == 1, 0],
    X_train_scaled[y_train == 1, 1],
    color="blue",
    edgecolor="black",
    s=60,
    label="Setosa"
)

plt.scatter(
    X_train_scaled[y_train == 0, 0],
    X_train_scaled[y_train == 0, 1],
    color="red",
    edgecolor="black",
    s=60,
    label="Non-Setosa"
)

# Decision boundary: w.x + b = 0
plt.contour(
    xx,
    yy,
    Z,
    levels=[0],
    colors="black",
    linewidths=2
)




# Margins: w.x + b = -1 and +1
plt.contour(
    xx,
    yy,
    Z,
    levels=[-1, 1],
    colors="gray",
    linestyles="--",
    linewidths=1.5
)



# Highlight support vectors
plt.scatter(
    svm_model.support_vectors_[:, 0],
    svm_model.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="green",
    linewidths=2,
    label="Support Vectors"
)

plt.xlabel("Standardized Petal Length")
plt.ylabel("Standardized Petal Width")
plt.title("Linear SVM: Setosa vs Non-Setosa")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
