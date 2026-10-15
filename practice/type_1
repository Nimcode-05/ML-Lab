import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

#malignant is cosidered as 0 and benign as 1

data = load_breast_cancer()
X = data.data
y = data.target
print("Feature shape:", X.shape)
print("Target names:", data.target_names)
print("Target values:", np.unique(y))



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Train the three base classifiers

# Decision Tree
dt = DecisionTreeClassifier(
    random_state=42
)

# k-Nearest Neighbors
knn = KNeighborsClassifier(
    n_neighbors=9
)

# Gaussian Naive Bayes
gnb = GaussianNB()


# Train all three models
dt.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
gnb.fit(X_train_scaled, y_train)

# predict from the 3 base classifiers

dt_pred = dt.predict(X_test_scaled)
knn_pred = knn.predict(X_test_scaled)
gnb_pred = gnb.predict(X_test_scaled)


# test accuracy for all the model

dt_acc = accuracy_score(y_test, dt_pred)
knn_acc = accuracy_score(y_test, knn_pred)
gnb_acc = accuracy_score(y_test, gnb_pred)

print("BASE MODEL TEST ACCURACIES")

print(f"Decision Tree       : {dt_acc:.4f}")
print(f"k-NN                : {knn_acc:.4f}")
print(f"Gaussian Naive Bayes: {gnb_acc:.4f}")




#Generate probability features
# We need P(malignant).
# Therefore, [:, 0] gives probability of malignant.

dt_train_prob = dt.predict_proba(X_train_scaled)[:, 0]
knn_train_prob = knn.predict_proba(X_train_scaled)[:, 0]
gnb_train_prob = gnb.predict_proba(X_train_scaled)[:, 0]

# Generate probabilities for TEST data as well
dt_test_prob = dt.predict_proba(X_test_scaled)[:, 0]
knn_test_prob = knn.predict_proba(X_test_scaled)[:, 0]
gnb_test_prob = gnb.predict_proba(X_test_scaled)[:, 0]

#Append the three probability features
#Original features + 3 probability features

X_train_augmented = np.column_stack([
    X_train_scaled,
    dt_train_prob,
    knn_train_prob,
    gnb_train_prob
])

X_test_augmented = np.column_stack([
    X_test_scaled,
    dt_test_prob,
    knn_test_prob,
    gnb_test_prob
])


print("FEATURE DIMENSIONS")

print("Original features :", X_train_scaled.shape[1])
print("Augmented features:", X_train_augmented.shape[1])





# Logistic Regression on ORIGINAL features

lr_original = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_original.fit(X_train_scaled, y_train)

original_pred = lr_original.predict(X_test_scaled)

original_acc = accuracy_score(
    y_test,
    original_pred
)

original_cm = confusion_matrix(
    y_test,
    original_pred
)




# Logistic Regression on AUGMENTED features

lr_augmented = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_augmented.fit(
    X_train_augmented,
    y_train
)

augmented_pred = lr_augmented.predict(
    X_test_augmented
)

augmented_acc = accuracy_score(
    y_test,
    augmented_pred
)

augmented_cm = confusion_matrix(
    y_test,
    augmented_pred
)



#Compare Logistic Regression results

print("LOGISTIC REGRESSION COMPARISON")

print(f"Original features accuracy : {original_acc:.4f}")
print(f"Augmented features accuracy: {augmented_acc:.4f}")

print("\nConfusion Matrix - Original Features")
print(original_cm)

print("\nConfusion Matrix - Augmented Features")
print(augmented_cm)


#Display confusion matrices

fig, ax = plt.subplots()

ConfusionMatrixDisplay(
    confusion_matrix=original_cm,
    display_labels=["Malignant", "Benign"]
).plot(ax=ax)

ax.set_title("Logistic Regression - Original Features")
plt.show()


fig, ax = plt.subplots()

ConfusionMatrixDisplay(
    confusion_matrix=augmented_cm,
    display_labels=["Malignant", "Benign"]
).plot(ax=ax)

ax.set_title("Logistic Regression - Augmented Features")
plt.show()





# HOSPITAL SCREENING ANALYSIS
# Final model = Logistic Regression on ORIGINAL features
# Probability of malignant class (class 0)

malignant_prob = lr_original.predict_proba(
    X_test_scaled
)[:, 0]


# Default threshold = 0.5


default_pred = np.where(
    malignant_prob >= 0.5,
    0,      # malignant
    1       # benign
)
default_cm = confusion_matrix(
    y_test,
    default_pred,
    labels=[0, 1]
)

print("DEFAULT THRESHOLD = 0.5")


print(default_cm)

default_TP = default_cm[0, 0]
default_FN = default_cm[0, 1]
default_FP = default_cm[1, 0]
default_TN = default_cm[1, 1]

print("Malignant correctly detected (TP):", default_TP)
print("Malignant missed (FN):", default_FN)
print("Benign sent for biopsy (FP):", default_FP)
print("Benign correctly cleared (TN):", default_TN)

thresholds = np.sort(
np.unique(malignant_prob)
)

valid_results = []

for threshold in thresholds:

pred = np.where(
    malignant_prob >= threshold,
    0,      # malignant
    1       # benign
)

cm = confusion_matrix(
    y_test,
    pred,
    labels=[0, 1]
)

FN = cm[0, 1]
FP = cm[1, 0]

# Hospital requirement
if FN <= 2:
    valid_results.append(
        (threshold, FN, FP)
    )


# Choose the threshold with the minimum FP
# while maintaining FN <= 2

best_threshold, best_FN, best_FP = min(
valid_results,
key=lambda x: x[2]
)

print("Selected threshold:", best_threshold)
print("FN:", best_FN)
print("FP:", best_FP)


print("THRESHOLD ANALYSIS")


print(f"Default threshold       : 0.50")
print(f"Selected threshold      : {best_threshold:.4f}")


print("\nMalignant cases missed:")
print("At 0.5 :", default_FN)
print("Selected:", best_FN)

print("\nBenign patients sent for biopsy:")
print("At 0.5 :", default_FP)
print("Selected:", best_FP)

print("\nChange in unnecessary biopsies:",
      best_FP - default_FP)

threshold_pred = np.where(
    malignant_prob >= best_threshold,
    0,
    1
)

threshold_cm = confusion_matrix(
    y_test,
    threshold_pred,
    labels=[0, 1]
)

threshold_TP = threshold_cm[0, 0]
threshold_FN = threshold_cm[0, 1]
threshold_FP = threshold_cm[1, 0]
threshold_TN = threshold_cm[1, 1]

fig, ax = plt.subplots()

ConfusionMatrixDisplay(
    confusion_matrix=threshold_cm,
    display_labels=["Malignant", "Benign"]
).plot(ax=ax)

ax.set_title(
    f"Final Model - Threshold = {best_threshold:.4f}"
)

plt.show()
