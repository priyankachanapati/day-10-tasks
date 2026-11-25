
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Step 2: Define 5-fold cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Step 3: Create Logistic Regression pipeline
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=10000))
])

# Step 4: Create Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 5: Evaluate Logistic Regression
logistic_scores = cross_val_score(
    logistic_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# Step 6: Evaluate Random Forest
rf_scores = cross_val_score(
    random_forest_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# Step 7: Calculate mean and standard deviation
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Mean Accuracy": [
        np.mean(logistic_scores),
        np.mean(rf_scores)
    ],
    "Standard Deviation": [
        np.std(logistic_scores),
        np.std(rf_scores)
    ]
})

# Step 8: Display results
print("\nIndividual Fold Scores")
print("--------------------------------")

print("Logistic Regression:", logistic_scores)
print("Random Forest:", rf_scores)

print("\nModel Comparison")
print("--------------------------------")
print(results.to_string(index=False))

# Step 9: Identify the better model
if np.mean(logistic_scores) > np.mean(rf_scores):
    print("\nBetter Model: Logistic Regression")
elif np.mean(rf_scores) > np.mean(logistic_scores):
    print("\nBetter Model: Random Forest")
else:
    print("\nBoth models have equal mean accuracy.")
