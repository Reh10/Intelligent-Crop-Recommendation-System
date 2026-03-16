import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Show first 5 rows
print("Dataset Preview:")
print(data.head())

# Separate input and output
X = data[['N','P','K','temperature','humidity','ph','rainfall']]
y = data['label']

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Example Prediction
sample = [[90, 42, 43, 20, 82, 6.5, 200]]
result = model.predict(sample)

print("\nஉங்கள் நிலத்திற்கு பொருத்தமான பயிர்:", result[0])
