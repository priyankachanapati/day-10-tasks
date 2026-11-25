
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Step 2: Make malignant class the positive class
# Original: 0 = Malignant, 1 = Benign
y = (y == 0).astype(int)

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=10000))
])

# Step 5: Define hyperparameter grid
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

# Step 6: Configure multiple scoring metrics
scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

# Step 7: Configure GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit="recall",
    cv=5,
    n_jobs=-1
)

# Step 8: Train and tune the model
grid_search.fit(X_train, y_train)

# Step 9: Display best model
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Recall:", grid_search.best_score_)

# Step 10: Predict on test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Step 11: Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"]
    )
)

# Step 12: Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 13: Display all cross-validation metrics
print("\nGrid Search Metric Comparison:")

results = grid_search.cv_results_

for i, params in enumerate(results["params"]):
    print("\nParameters:", params)
    print("Mean Precision:", round(results["mean_test_precision"][i], 4))
    print("Mean Recall:", round(results["mean_test_recall"][i], 4))
    print("Mean F1:", round(results["mean_test_f1"][i], 4))
