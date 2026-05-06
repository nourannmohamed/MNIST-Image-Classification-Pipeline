import os
import sys
from pathlib import Path
from PIL import ImageFilter
import cv2

ROOT_DIR = Path(r"C:\Users\B2B\Documents\GitHub\MNIST-Image-Classification-Pipeline")

sys.path.insert(0, str(ROOT_DIR / "phase2" / "models"))
sys.path.insert(0, str(ROOT_DIR / "phase2"))
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

from svm import MulticlassSVM


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="MNIST AI Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================
# CUSTOM CSS
# =====================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .title {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        color: #1E293B;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #bbbbbb;
        margin-bottom: 30px;
    }

    .prediction-box {
        background-color: #1c1f26;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #333;
    }

    .prediction-number {
        font-size: 80px;
        font-weight: bold;
        color: #4CAF50;
    }

    .section-header {
        font-size: 30px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================
# HEADER
# =====================================
st.markdown('<div class="title">🧠 MNIST Digit Recognition</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">CNN Feature Extraction + Multiclass SVM</div>',
    unsafe_allow_html=True
)


# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "✍ Predict Digit",
        "📊 Model Performance",
        "⚙ Pipeline Overview"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This system uses a trained CNN feature extractor combined with a custom multiclass SVM model."
)


# =====================================
# BUILD CNN FEATURE EXTRACTOR
# =====================================
@st.cache_resource
def build_feature_extractor():

    input_layer = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    feature_layer = Dense(128, activation='relu')(x)

    output_layer = Dense(10, activation='softmax')(feature_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.load_weights(
        ROOT_DIR / "saved_models" / "cnn_weights.weights.h5"
    )

    feature_model = Model(inputs=model.input, outputs=feature_layer)

    return feature_model


# =====================================
# LOAD SVM
# =====================================
@st.cache_resource
def load_svm():

    svm = MulticlassSVM()

    svm.W = np.load(ROOT_DIR / "saved_models" / "svm_weights.npy")

    return svm


feature_model = build_feature_extractor()
svm_model = load_svm()


# =====================================
# PREPROCESS IMAGE
# =====================================
def preprocess_image(img):

    img = img.convert("L")

    img = np.array(img)

    img = img.astype("float32") / 255.0
    img = cv2.bilateralFilter(
        (img * 255).astype(np.uint8),
        5,
        75,
        75
    )

    img = img.astype("float32") / 255.0

    coords = np.argwhere(img > 0.1)

    if len(coords) == 0:
        return np.zeros((1, 28, 28, 1))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img = img[y_min:y_max+1, x_min:x_max+1]

    pil_img = Image.fromarray((img * 255).astype(np.uint8))


    pil_img = pil_img.resize((20, 20))

    img = np.array(pil_img).astype("float32") / 255.0

    final_img = np.zeros((28, 28))

    final_img[4:24, 4:24] = img

    final_img = final_img.reshape(1, 28, 28, 1)

    return final_img


# =====================================
# HOME PAGE
# =====================================
if page == "🏠 Home":

    st.markdown(
        '<div class="section-header">📘 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This project focuses on handwritten digit recognition using the MNIST dataset.
        The system was implemented in two main phases to compare different machine
        learning approaches for binary and multiclass classification.
        """
    )

    st.markdown("---")

    # =====================================
    # PHASE 1
    # =====================================
    st.markdown(
        """
        <div style="
            background-color:#EEF2FF;
            padding:25px;
            border-radius:15px;
            border-left:8px solid #4F46E5;
            margin-bottom:25px;
        ">
        <h2 style="color:#1E293B;">🔹 Phase 1 — Binary Classification</h2>

        <p style="font-size:18px; color:#334155;">
        In Phase 1, the task was to classify digits into:
        </p>

        <ul style="font-size:18px; color:#334155;">
            <li><b>Class 1:</b> Digit 0</li>
            <li><b>Class 2:</b> Any digit other than 0</li>
        </ul>

        <p style="font-size:18px; color:#334155;">
        The following machine learning models were implemented from scratch:
        </p>

        <ul style="font-size:18px; color:#0F172A;">
            <li>✔ K-Nearest Neighbors (KNN)</li>
            <li>✔ Naive Bayes</li>
            <li>✔ Logistic Regression</li>
            <li>✔ Perceptron</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================
    # PHASE 2
    # =====================================
    st.markdown(
        """
        <div style="
            background-color:#ECFDF5;
            padding:25px;
            border-radius:15px;
            border-left:8px solid #10B981;
            margin-bottom:25px;
        ">
        <h2 style="color:#1E293B;">🔹 Phase 2 — Multiclass Classification</h2>

        <p style="font-size:18px; color:#334155;">
        In Phase 2, the system was extended to perform full multiclass
        handwritten digit recognition for all digits from 0 to 9.
        </p>

        <p style="font-size:18px; color:#334155;">
        CNN feature extraction was used before classification.
        The following models were implemented:
        </p>

        <ul style="font-size:18px; color:#0F172A;">
            <li>✔ K-Nearest Neighbors (KNN)</li>
            <li>✔ Naive Bayes</li>
            <li>✔ Multinomial Regression</li>
            <li>✔ Support Vector Machine (SVM)</li>
            <li>✔ Perceptron</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =====================================
    # DATASET SECTION
    # =====================================
    st.markdown(
        '<div class="section-header">📊 Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background-color:#F8FAFC;
            padding:20px;
            border-radius:15px;
            border:1px solid #CBD5E1;
        ">

        <p style="font-size:18px; color:#334155;">
        The project uses the <b>MNIST handwritten digit dataset</b>,
        which contains grayscale images of handwritten digits.
        </p>

        <ul style="font-size:18px; color:#0F172A;">
            <li>✔ 70,000 total images</li>
            <li>✔ 28 × 28 grayscale images</li>
            <li>✔ Digits from 0 to 9</li>
            <li>✔ Standard benchmark dataset for machine learning</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================
# PREDICTION PAGE
# =====================================
elif page == "✍ Predict Digit":

    st.markdown('<div class="section-header">Draw a Digit</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=5,
            stroke_color="white",
            background_color="black",
            height=220,
            width=220,
            drawing_mode="freedraw",
            key="canvas",
        )

        predict_button = st.button("🔍 Predict Digit")

    with col2:

        if predict_button:

            if canvas_result.image_data is not None:

                img = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                )

                processed = preprocess_image(img)

                features = feature_model.predict(processed, verbose=0)

                prediction = svm_model.predict(features)[0]

                scores = np.dot(features, svm_model.W.T)[0]

                probs = np.exp(scores) / np.sum(np.exp(scores))

                top3 = np.argsort(probs)[-3:][::-1]

                st.markdown(
                    f'''
                    <div class="prediction-box">
                        <div style="font-size:22px; color:#bbb;">Predicted Digit</div>
                        <div class="prediction-number">{prediction}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

                st.markdown("### Confidence Scores")

                for idx in top3:
                    st.progress(float(probs[idx]))
                    st.write(f"Digit {idx}: {probs[idx]*100:.2f}%")

                st.markdown("---")

                st.subheader("Processed Image")

                fig, ax = plt.subplots(figsize=(3, 3))

                ax.imshow(processed[0].reshape(28, 28), cmap='gray')

                ax.axis("off")

                st.pyplot(fig)


# =====================================
# PERFORMANCE PAGE
# =====================================
elif page == "📊 Model Performance":

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Accuracy", "99%")

    with col2:
        st.metric("Feature Size", "128")

    with col3:
        st.metric("Classes", "10")

    st.markdown("---")

    st.table({
        "Model": [
            "KNN",
            "Naive Bayes",
            "Perceptron",
            "Multinomial",
            "SVM"
        ],

        "Type": [
            "Distance-Based",
            "Probabilistic",
            "Linear",
            "Regression-Based",
            "Margin-Based"
        ]
    })


# =====================================
# PIPELINE PAGE
# =====================================
elif page == "⚙ Pipeline Overview":

    st.markdown('<div class="section-header">System Pipeline</div>', unsafe_allow_html=True)

    st.code(
        '''
Input Image
    ↓
Preprocessing
    ↓
CNN Feature Extraction
    ↓
128-Dimensional Feature Vector
    ↓
Multiclass SVM
    ↓
Predicted Digit
        '''
    )

    st.markdown("---")

    st.write(
        "The CNN extracts meaningful high-level features from handwritten digits before classification using the custom multiclass SVM implementation."
    )