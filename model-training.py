# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1a. Load dataset using pandas
data = load_breast_cancer()

# Convert to pandas DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)

# 1b. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# 1c. Train a Logistic Regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 2. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 3. Save the trained model using pickle
with open("logistic_regression_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as 'logistic_regression_model.pkl'")