# File: train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("migraine_symptom_classification.csv")

# Assume last column is the target
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Model evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature list
with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)
