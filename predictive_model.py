from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Load Dataset
# ==========================
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target

# Display Dataset Preview
print("========== DATASET PREVIEW ==========")
print(df.head())

# ==========================
# Features and Target
# ==========================
X = iris.data
y = iris.target

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Random Forest Model
# ==========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Make Predictions
# ==========================
predictions = model.predict(X_test)

# ==========================
# Accuracy Score
# ==========================
accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

# ==========================
# Confusion Matrix
# ==========================
cm = confusion_matrix(y_test, predictions)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# Graph 1 - Confusion Matrix
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.show()

# ==========================
# Feature Importance Graph
# ==========================
importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(iris.feature_names, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
