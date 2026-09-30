import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create student dataset
data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Attendance": [60, 65, 70, 72, 75, 80, 82, 85, 90, 95],
    "Previous_Marks": [45, 50, 55, 60, 65, 70, 72, 78, 85, 90],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["Hours_Studied", "Attendance", "Previous_Marks"]]
y = df["Result"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Machine Learning model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display model information
print("Student Performance Prediction")
print("--------------------------------")
print("Model Accuracy:", accuracy * 100, "%")

# Get student input
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
marks = float(input("Enter previous marks: "))

# Create DataFrame for new student
new_student = pd.DataFrame(
    [[hours, attendance, marks]],
    columns=["Hours_Studied", "Attendance", "Previous_Marks"]
)

# Make prediction
prediction = model.predict(new_student)

# Display result
if prediction[0] == 1:
    print("Prediction: Good Performance")
else:
    print("Prediction: Needs Improvement")