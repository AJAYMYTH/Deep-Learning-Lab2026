import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Build the MLP model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        10,
        activation='relu',
        input_shape=(4,)
    ),
    tf.keras.layers.Dense(
        3,
        activation='softmax'
    )
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=5,
    verbose=0
)

# ---------------- Evaluation ----------------

# Predict class probabilities
y_pred_prob = model.predict(X_test, verbose=0)

# Convert one-hot encoded labels to class indices
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate metrics
print("Accuracy:", accuracy_score(y_true, y_pred))

print("Precision:", precision_score(
    y_true,
    y_pred,
    average='macro'
))

print("Recall:", recall_score(
    y_true,
    y_pred,
    average='macro'
))

print("F1 Score:", f1_score(
    y_true,
    y_pred,
    average='macro'
))

# Detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=iris.target_names
))