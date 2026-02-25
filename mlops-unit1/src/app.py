import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = data.target

print("Dataset Shape:", df.shape)
print("\nStatistical Summary:")
print(df.describe())

# Train simple model
model = LogisticRegression(max_iter=200)
model.fit(df, target)

# Save model inside models folder
joblib.dump(model, "../models/iris_model.joblib")

print("\nModel saved successfully inside models/ folder")