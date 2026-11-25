
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# In this dataset:
# 0 = Malignant
# 1 = Benign
# We treat Malignant (0) as the positive class.
y_malignant = (y == 0).astype(int)

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_malignant,
    test_size=0.2,
    random_state=42,
    stratify=y_malignant
)

# Step 3: Create Logistic Regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=10000))
])

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Get prediction probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Step 6: Calculate ROC-AUC
auc = roc_auc_score(y_test, y_scores)

print("ROC-AUC Score:", round(auc, 4))

# Step 7: Plot ROC Curve
RocCurveDisplay.from_predictions(
    y_test,
    y_scores,
    name="Logistic Regression"
)

plt.title("ROC Curve - Logistic Regression")
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# Step 8: Plot Precision-Recall Curve
PrecisionRecallDisplay.from_predictions(
    y_test,
    y_scores,
    name="Logistic Regression"
)

plt.title("Precision-Recall Curve - Logistic Regression")
plt.savefig("precision_recall_curve.png", dpi=300,
            bbox_inches="tight")
plt.show()

# Step 9: Find a threshold that maintains recall >= 0.90
precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_scores
)

eligible = np.where(recall[:-1] >= 0.90)[0]

if len(eligible) == 0:
    raise ValueError("No threshold satisfies recall >= 0.90")

# Select the highest eligible threshold
best_index = eligible[-1]
best_threshold = thresholds[best_index]

# Step 10: Generate predictions at selected threshold
y_pred = (y_scores >= best_threshold).astype(int)

# Step 11: Evaluate the selected threshold
print("\nSelected Threshold:", round(best_threshold, 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 12: Save threshold-specific confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")

ax.set_title("Confusion Matrix - Selected Threshold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Benign", "Malignant"])
ax.set_yticklabels(["Benign", "Malignant"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
