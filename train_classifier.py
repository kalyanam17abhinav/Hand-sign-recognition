import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset from pickle file
with open("./ASL.pickle", "rb") as f:
    dataset = pickle.load(f)

# Count occurrences of each label
label_counts = Counter(dataset["labels"])

# Find the labels with only one sample
labels_to_remove = [label for label, count in label_counts.items() if count == 1]

# Filter out these labels from the dataset
filtered_dataset = []
filtered_labels = []

for data, label in zip(dataset["dataset"], dataset["labels"]):
    if label not in labels_to_remove:
        filtered_dataset.append(data)
        filtered_labels.append(label)

# Convert filtered dataset to numpy arrays
data = np.asarray(filtered_dataset)
labels = np.asarray(filtered_labels)

# Split the data into train and test sets, stratifying by labels
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
score = accuracy_score(y_pred, y_test)
print(f"Accuracy: {score}")

# Save the trained model to a pickle file
with open("./ASL_model.p", "wb") as f:
    pickle.dump({"model": model}, f)
