import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 75)
print("              ARTIFICIAL INTELLIGENCE PROJECT")
print("               IRIS FLOWER CLASSIFICATION")
print("=" * 75)

# Load Iris Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target

# Dataset Overview
print("\n[1] DATASET OVERVIEW")
print("-" * 50)
print("Dataset Name   : Iris Flower Dataset")
print("Total Records  :", len(df))
print("Total Features :", len(iris.feature_names))
print("Target Classes :", len(iris.target_names))

print("\nFeature Names:")
for feature in iris.feature_names:
    print("•", feature)

# Flower Categories
print("\n[2] FLOWER SPECIES CATEGORIES")
print("-" * 50)
for i, flower in enumerate(iris.target_names):
    print(f"Class {i} : {flower.capitalize()}")

# Dataset Sample
print("\n[3] DATASET SAMPLE")
print("-" * 50)
print(df.head())

# Dataset Statistics
print("\n[4] DATASET STATISTICS")
print("-" * 50)
print(df.describe())

# Feature Averages
print("\n[5] FEATURE AVERAGES")
print("-" * 50)

for feature in iris.feature_names:
    print(f"{feature:<20} : {df[feature].mean():.2f}")

# Features and Target
X = iris.data
y = iris.target

# Class Distribution
print("\n[6] CLASS DISTRIBUTION")
print("-" * 50)

for i, flower in enumerate(iris.target_names):
    count = sum(y == i)
    percentage = (count / len(y)) * 100
    print(f"{flower.capitalize():<12} : {count} samples ({percentage:.2f}%)")

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Classification Model
model = DecisionTreeClassifier(random_state=42)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Model Training Information
print("\n[7] MODEL TRAINING INFORMATION")
print("-" * 50)
print("Training Records :", len(X_train))
print("Testing Records  :", len(X_test))

# Classification Results
print("\n[8] CLASSIFICATION RESULTS")
print("-" * 50)
print("Model Accuracy :", round(accuracy * 100, 2), "%")

# Model Evaluation
print("\n[9] MODEL EVALUATION")
print("-" * 50)

if accuracy >= 0.95:
    print("Excellent Classification Performance")
elif accuracy >= 0.80:
    print("Good Classification Performance")
else:
    print("Model Needs Improvement")

# Confusion Matrix
print("\n[10] CONFUSION MATRIX")
print("-" * 50)
print(confusion_matrix(y_test, y_pred))

# Prediction Distribution
print("\n[11] PREDICTION DISTRIBUTION")
print("-" * 50)

for i, flower in enumerate(iris.target_names):
    count = sum(y_pred == i)
    print(f"{flower.capitalize():<12} : {count} predictions")

# Sample Predictions
print("\n[12] SAMPLE PREDICTIONS")
print("-" * 50)

for i in range(5):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]
    print(f"Actual: {actual:<12} Predicted: {predicted}")

# Performance Summary
print("\n[13] PERFORMANCE SUMMARY")
print("-" * 50)
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# Final Result
print("=" * 75)
print("FINAL RESULT")
print("=" * 75)

print(f"""
Project Objective
-----------------
To build an Artificial Intelligence classification model that
can identify Iris flower species based on flower measurements.

Project Achievements
--------------------
✓ Dataset Loaded Successfully
✓ Data Analysis Performed
✓ Dataset Split into Training and Testing Sets
✓ Classification Model Trained Successfully
✓ Flower Species Predicted Accurately
✓ Model Performance Evaluated

Classification Categories
-------------------------
• Setosa
• Versicolor
• Virginica

Overall Accuracy : {accuracy * 100:.2f}%

Conclusion
----------
The Artificial Intelligence model successfully classified
Iris flowers into their respective species with high accuracy.
The model demonstrated effective learning and prediction
capabilities using supervised machine learning techniques.

Project Status : SUCCESSFUL
""")

print("=" * 75)