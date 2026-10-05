# Deep Learning Lab 2026

A collection of small Python experiments covering NumPy, activation functions,
neural-network training, Iris classification, MNIST convolutional networks, and
tomato-leaf disease classification.

## Contents

- [Requirements and setup](#requirements-and-setup)
- [Run the experiments](#run-the-experiments)
- [Experiments 8a and 8b: tomato disease CNN](#experiments-8a-and-8b-tomato-disease-cnn)
- [Dataset details](#dataset-details)
- [Troubleshooting](#troubleshooting)

## Requirements and setup

Use Python 3.11 (64-bit recommended). TensorFlow runs on the CPU in this
Windows setup; GPU support on native Windows depends on the TensorFlow version
and is not required for these scripts.

In Windows PowerShell:

```powershell
git clone https://github.com/AJAYMYTH/Deep-Learning-Lab2026.git
Set-Location .\Deep-Learning-Lab2026

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can run the project
interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\main-8a.py
```

The image-classification dataset and trained `plant_disease_cnn_model.h5` are
included in the repository, so experiment 8b can run without first downloading
or training. Training again with 8a replaces the model file.

## Run the experiments

Run a script from the repository root. For example:

```powershell
python .\main-1a.py
```

Some scripts open Matplotlib windows; close each plot window to let the script
continue or exit. Scripts 1c, 2b, 3c, 5a, 6c, 7a, 7b, 7c, and 8a display
plots. Script 8b displays the prediction image.

| Script | What it demonstrates |
| --- | --- |
| `main-1a.py` | NumPy arrays, reshaping, and slicing |
| `main-1b.py` | A weighted sum using a vector dot product |
| `main-1c.py` | Displaying a random matrix as a heatmap |
| `main-2b.py` | Sigmoid, tanh, and ReLU activation curves |
| `main-3a.py` | Training a small XOR neural network with a manual forward/backward pass |
| `main-3b.py` | One manual forward pass, gradient calculation, and weight update for XOR |
| `main-3c.py` | XOR training over multiple iterations and a loss curve |
| `main-5a.py` | Iris MLP with early stopping, confusion matrix, accuracy/loss plots, and a new-flower prediction |
| `main-5b.py` | Iris MLP with accuracy, precision, recall, F1 score, and a classification report |
| `main-6a.py` | Comparing Iris models with different capacities (underfitting and overfitting) |
| `main-6b.py` | Iris classification with L2 regularization and dropout |
| `main-6c.py` | Iris MLP with early stopping, test accuracy, and a loss curve |
| `main-7a.py` | MNIST digit classification with a CNN and train/validation loss plot |
| `main-7b.py` | MNIST CNN training and test accuracy |
| `main-7c.py` | Visualizing intermediate CNN feature maps for MNIST |
| `main-8a.py` | Training and evaluating the tomato-leaf disease CNN |
| `main-8b.py` | Loading the saved tomato model and predicting a dataset image |

The experiments are standalone scripts; run them one at a time. MNIST is
downloaded by Keras on the first run of experiments 7a–7c. Experiments 5a–6c
use scikit-learn's built-in Iris dataset.

## Experiments 8a and 8b: tomato disease CNN

### 1. Check the dataset

The expected folder structure is:

```text
dataset/
  Bacterial Spot/
  Early Blight/
  Healthy/
  Late Blight/
  Leaf Mold/
  Septoria Leaf Spot/
  Spider Mites Two-spotted Spider Mite/
  Target Spot/
  Tomato Mosaic Virus/
  Tomato Yellow Leaf Curl Virus/
```

The included dataset has 100 images per class (1,000 images total). If you
remove it and need to download it again, run:

```powershell
python .\create_dataset.py
```

The downloader defaults to 100 images per class. To request a different count
(up to 1,100 per class):

```powershell
python .\create_dataset.py --samples-per-class 200
```

It will not overwrite a non-empty `dataset` directory. Move or rename that
directory before regenerating it. Dataset source, revision, and license are
recorded in `dataset/DATASET_INFO.json`.

### 2. Train with `main-8a.py`

```powershell
python .\main-8a.py
```

The script reads class-named folders under `dataset/`, resizes images to
128 × 128, scales pixel values to 0–1, and uses an 80/20 training/validation
split. It trains a CNN for 20 epochs and prints the class mapping and sample
counts. At the end it saves or replaces `plant_disease_cnn_model.h5` in the
repository root and prints final validation accuracy.

Training can take several minutes on CPU. Keep the terminal open until the
final validation result and model-save message appear. The model file is about
75 MiB.

### 3. Predict with `main-8b.py`

After training completes (or when using the included trained model), run:

```powershell
python .\main-8b.py
```

The script loads `plant_disease_cnn_model.h5`, selects the first image in the
dataset's sorted class/file order, preprocesses it at the model's input size,
prints a prediction, and displays the image with the predicted class and
confidence. The model must correspond to the same set and ordering of classes
as the dataset.

The checked-in model was trained on this dataset and achieved 80.0% validation
accuracy in one run. Results can vary between training runs because model
initialization and training order are not fixed.

## Dataset details

The tomato-leaf images are from
[Project-AgML's tomato leaf disease dataset](https://huggingface.co/datasets/Project-AgML/tomato_leaf_disease),
which identifies its license as **CC0-1.0**. This repository contains a
balanced 1,000-image subset with 10 classes. This is a tomato-only dataset,
not the full PlantVillage collection. See `dataset/DATASET_INFO.json` for the
pinned source revision and class names.

## Troubleshooting

- **Model file not found:** Run `main-8a.py` to train the model, or make sure
  `plant_disease_cnn_model.h5` is in the repository root.
- **Dataset not found:** Restore the checked-in `dataset/` folder or run
  `create_dataset.py`.
- **TensorFlow oneDNN or CPU/GPU messages:** These are informational/warning
  messages. They do not by themselves indicate that training or prediction
  failed.
- **A plot window appears to stop the program:** Close the window to continue.
- **`ModuleNotFoundError`:** Activate the virtual environment and install
  `requirements.txt`.
