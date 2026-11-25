
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Train decision trees with different depths
results = []

depths = range(1, 16)

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on training and testing data
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(
        y_train, train_predictions
    )

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    results.append({
        "Max Depth": depth,
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy
    })

# Step 4: Display results
df = pd.DataFrame(results)

print("\nModel Evaluation Results:\n")
print(df.to_string(index=False))

# Step 5: Plot training and testing accuracy
plt.figure(figsize=(10, 6))

plt.plot(
    df["Max Depth"],
    df["Training Accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    df["Max Depth"],
    df["Testing Accuracy"],
    marker="s",
    label="Testing Accuracy"
)

plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy - Overfitting Analysis")

plt.xticks(range(1, 16))
plt.ylim(0.75, 1.05)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("overfitting_analysis.png", dpi=300)
plt.show()

# Step 6: Identify the generalization gap
df["Accuracy Gap"] = (
    df["Training Accuracy"] - df["Testing Accuracy"]
)

print("\nAccuracy Gap Analysis:\n")
print(df[["Max Depth", "Accuracy Gap"]].to_string(index=False))

print("\nGraph saved as overfitting_analysis.png")
