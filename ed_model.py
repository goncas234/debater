import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


data = pd.read_csv("data.csv")

x = data.drop(columns=["Class"], axis=1)
y = data["Class"]


train_x, test_x, train_y, test_y = train_test_split(
    x, y, stratify=y, test_size=0.2, random_state=42
)


model = Pipeline([
    ("standard_scaler", StandardScaler()),
    ("random_forest", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=20,
        min_samples_split=5
    ))
])


print("Performing 5-fold cross-validation...")
cv_scores = cross_val_score(model, train_x, train_y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

model.fit(train_x, train_y)

pred_y = model.predict(test_x)
test_accuracy = accuracy_score(test_y, pred_y)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(test_y, pred_y))
print("\nConfusion Matrix:")
print(confusion_matrix(test_y, pred_y))

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")