# MNIST Image Classification Pipeline

A complete machine learning pipeline for MNIST image classification, covering both binary and multi-class problems with models implemented from scratch.

## Project Overview

This project recognizes handwritten digits from the MNIST dataset using a two-phase machine learning pipeline.

In Phase 1, the problem is treated as binary classification: detecting whether an image is the digit `0` or not. Several classic machine learning models are implemented from scratch, including KNN, Naive Bayes, Logistic Regression, and Perceptron.

In Phase 2, the project expands to full multiclass classification for digits `0` through `9`. A CNN is used as a feature extractor, then the extracted feature vectors are classified using custom multiclass models such as SVM, Perceptron, Multinomial Regression, KNN, and Naive Bayes.

The project also includes a Streamlit web app where users can draw a digit on a canvas and get a prediction from the trained CNN feature extractor and SVM classifier.

## Requirements

- Python 3.9 or newer
- pip

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment.

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```bash
.\.venv\Scripts\activate.bat
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install streamlit streamlit-drawable-canvas numpy pillow opencv-python matplotlib tensorflow scikit-learn h5py
```

## Required Packages

The project uses these external Python packages:

- `streamlit` - web app interface
- `streamlit-drawable-canvas` - drawing canvas for handwritten digit input
- `numpy` - numerical operations and saved model weights
- `pillow` - image loading and processing
- `opencv-python` - image filtering and preprocessing
- `matplotlib` - plotting processed images and learning curves
- `tensorflow` - MNIST dataset loading and CNN feature extractor
- `scikit-learn` - train/test split, metrics, and cross-validation helpers
- `h5py` - reading TensorFlow/Keras `.h5` weight files

## Running the App

```bash
streamlit run app.py
```

The app loads the saved CNN and SVM weights from the `saved_models/` folder:

- `saved_models/cnn_weights.weights.h5`
- `saved_models/svm_weights.npy`

## Running the Training Scripts

Run the Phase 1 binary classification models:

```bash
python phase1/main.py
```

Run the Phase 2 multiclass models:

```bash
python phase2/models/run_all.py
```

Generate CNN features and save CNN weights:

```bash
python phase2/cnn_features.py
```

Save the trained SVM weights:

```bash
python phase2/models/save_svm.py
```
