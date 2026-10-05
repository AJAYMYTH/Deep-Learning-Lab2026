import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

# 1. Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0  # shape -> (60000, 28, 28, 1)
x_test = x_test[..., np.newaxis] / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. Build a CNN model with 2 conv layers
input_layer = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', name='conv1')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
output_layer = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train the model (just 1-2 epochs for quick test)
model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.1)

# 4. Function to visualize feature maps
def visualize_feature_maps(model, layer_names, input_image):
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=outputs)

    # Get the feature maps
    activations = activation_model.predict(input_image)
    for layer_name, activation in zip(layer_names, activations):
        num_filters = activation.shape[-1]
        plt.figure(figsize=(15, 5))
        for i in range(min(num_filters, 8)):  # Display up to 8 feature maps
            plt.subplot(1, 8, i + 1)
            plt.imshow(activation[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f"Feature Maps from Layer: {layer_name}")
        plt.show()

# 5. Choose an image and visualize
img = x_test[0]  # or any other test image
img = np.expand_dims(img, axis=0)  # shape -> (1, 28, 28, 1)
visualize_feature_maps(model, ['conv1', 'conv2'], img)
