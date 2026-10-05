from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
project_dir = Path(__file__).resolve().parent
dataset_dir = project_dir / 'dataset'
if not dataset_dir.is_dir():
   raise FileNotFoundError(
      f"Training dataset not found: {dataset_dir}. "
      "Place the class folders and images in a dataset directory first."
   )

image_size = (128, 128)
batch_size = 32
val_split = 0.2
datagen = ImageDataGenerator(
   rescale=1./255,
   validation_split=val_split,
   rotation_range=20,
   zoom_range=0.2,
   horizontal_flip=True
)
train_generator = datagen.flow_from_directory(
   dataset_dir,
   target_size=image_size,
   batch_size=batch_size,
   class_mode='categorical',
   subset='training',
   shuffle=True
)
val_generator = datagen.flow_from_directory(
   dataset_dir,
   target_size=image_size,
   batch_size=batch_size,
   class_mode='categorical',
   subset='validation',
   shuffle=False
)
num_classes = len(train_generator.class_indices)
print("Class labels:", train_generator.class_indices)
print("Training samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)
print("Number of classes:", num_classes)
model = Sequential([
   Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
   MaxPooling2D(2,2),
   Conv2D(64, (3,3), activation='relu'),
   MaxPooling2D(2,2),
   Conv2D(128, (3,3), activation='relu'),
   MaxPooling2D(2,2),
   Flatten(),
   Dense(256, activation='relu'),
   Dropout(0.5),
   Dense(num_classes, activation='softmax')
 ])
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit(
   train_generator,
   epochs=20, 
   validation_data=val_generator
 )
model.save(str(project_dir / 'plant_disease_cnn_model.h5'))
print("Model saved as plant_disease_cnn_model.h5")
val_loss, val_acc = model.evaluate(val_generator)
print(f"Final Validation Accuracy: {val_acc:.4f}")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()