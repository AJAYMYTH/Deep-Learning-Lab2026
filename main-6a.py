import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target.reshape(-1, 1)


# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42
)


# Build MLP model
def build_model(hidden_layers, neurons, activation='relu'):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=(4,)))

    for _ in range(hidden_layers):
        model.add(
            tf.keras.layers.Dense(
                neurons,
                activation=activation
            )
        )

    model.add(
        tf.keras.layers.Dense(
            3,
            activation='softmax'
        )
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# --------------------------------------------------
# Underfitting Model
# --------------------------------------------------

under_model = build_model(
    hidden_layers=1,
    neurons=2
)

under_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=0
)

under_pred = np.argmax(
    under_model.predict(X_test, verbose=0),
    axis=1
)

y_true = np.argmax(y_test, axis=1)

print(
    "\nUnderfitting Model Accuracy:",
    accuracy_score(y_true, under_pred)
)


# --------------------------------------------------
# Good Model
# --------------------------------------------------

good_model = build_model(
    hidden_layers=5,
    neurons=64
)

good_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=0
)

good_pred = np.argmax(
    good_model.predict(X_test, verbose=0),
    axis=1
)

print(
    "\nGood Model Accuracy:",
    accuracy_score(y_true, good_pred)
)


# --------------------------------------------------
# Overfitting Model
# --------------------------------------------------

over_model = build_model(
    hidden_layers=5,
    neurons=64
)

over_model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=16,
    verbose=0
)

over_pred = np.argmax(
    over_model.predict(X_test, verbose=0),
    axis=1
)

print(
    "\nOverfitting Model Accuracy:",
    accuracy_score(y_true, over_pred)
)