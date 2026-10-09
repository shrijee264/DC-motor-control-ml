
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def generate_data(samples=2000):
    np.random.seed(42)

    data = []
    labels = []

    for _ in range(samples):
        fault = np.random.choice([
            "Normal",
            "High Friction",
            "Overload",
            "Sensor Fault"
        ])

        if fault == "Normal":
            mean_speed = np.random.normal(1000, 20)
            speed_std = np.random.normal(10, 2)
            mean_current = np.random.normal(2, 0.2)

        elif fault == "High Friction":
            mean_speed = np.random.normal(800, 40)
            speed_std = np.random.normal(20, 4)
            mean_current = np.random.normal(3.5, 0.3)

        elif fault == "Overload":
            mean_speed = np.random.normal(700, 50)
            speed_std = np.random.normal(25, 5)
            mean_current = np.random.normal(4, 0.4)

        else:
            mean_speed = np.random.normal(1000, 20)
            speed_std = np.random.normal(100, 15)
            mean_current = np.random.normal(2, 0.2)

        data.append([mean_speed, speed_std, mean_current])
        labels.append(fault)

    return np.array(data), np.array(labels)


X, y = generate_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("ML Fault Detection Results")
print("--------------------------")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions))

print("\nExample prediction:")
example = [[800, 20, 3.5]]
print("Predicted condition:", model.predict(example)[0])
