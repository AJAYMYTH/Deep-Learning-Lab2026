import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# 1. Load Iris Dataset
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)
print("Classes:", iris.target_names)


# ============================================================
# 2. Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 4. Create MLP Model
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),

    tf.keras.layers.Dense(3, activation='softmax')
])


# ============================================================
# 5. Compile Model
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# 6. Display Model
# ============================================================

model.summary()


# ============================================================
# 7. Train Model
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# 8. Evaluate Model
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# 9. Predictions
# ============================================================

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

print("\nActual Labels:")
print(y_test)

print("\nPredicted Labels:")
print(y_pred)


# ============================================================
# 10. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 11. Accuracy Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 12. Loss Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 13. Actual vs Predicted Plot
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(len(y_test)),
    y_test,
    'o-',
    label='Actual'
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    'x--',
    label='Predicted'
)

plt.xlabel("Test Sample")
plt.ylabel("Class")

plt.yticks(
    [0, 1, 2],
    iris.target_names
)

plt.title("Actual vs Predicted Classes")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 14. Predict New Flower
# ============================================================

new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

predicted_class = np.argmax(prediction, axis=1)[0]

print("\n==============================")
print("NEW FLOWER PREDICTION")
print("==============================")

print("Predicted Class :", predicted_class)
print("Species         :", iris.target_names[predicted_class])
print("Probability     :", prediction[0])