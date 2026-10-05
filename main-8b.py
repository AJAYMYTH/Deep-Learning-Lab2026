import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from pathlib import Path

project_dir = Path(__file__).resolve().parent
model_path = project_dir / 'plant_disease_cnn_model.h5'
dataset_dir = project_dir / 'dataset'
if not model_path.is_file():
    raise FileNotFoundError(
        f"Model file not found: {model_path}. "
        "Train the model first by running main-8a.py."
    )

model = load_model(str(model_path))
print("✅Model loaded successfully.")
image_paths = sorted(
    path
    for class_dir in dataset_dir.iterdir()
    if class_dir.is_dir()
    for path in class_dir.iterdir()
    if path.is_file()
)
if not image_paths:
    raise FileNotFoundError(f"No images found in dataset directory: {dataset_dir}")
img_path = image_paths[0]
img_size = (128, 128)
img = image.load_img(str(img_path), target_size=img_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
class_names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)
plt.imshow(img)
plt.title(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
plt.axis("off")
plt.show()