import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

np.random.seed(42)

num_samples = 5000

current_distance = np.random.uniform(0.5, 1000, num_samples)
relative_velocity = np.random.uniform(0.1, 15, num_samples)
predicted_min_distance = np.random.uniform(0.1, 500, num_samples)
time_to_tca = np.random.uniform(1, 120, num_samples)

collision_radius = 200

# Smooth probability curve
labels = 1 - (predicted_min_distance / collision_radius)
labels = np.clip(labels, 0, 1)

X = np.column_stack((
    current_distance,
    relative_velocity,
    predicted_min_distance,
    time_to_tca
))

y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")

joblib.dump(model, "collision_risk_model.pkl")
print("Model saved as collision_risk_model.pkl")