import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

from preprocessing import load_and_preprocess


def build_cnn_feature_extractor():
    # =========================
    # INPUT
    # =========================
    input_layer = Input(shape=(28, 28, 1))

    # =========================
    # CONV BLOCK 1
    # =========================
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)

    # =========================
    # CONV BLOCK 2
    # =========================
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # =========================
    # FLATTEN + FEATURE LAYER
    # =========================
    x = Flatten()(x)
    feature_layer = Dense(128, activation='relu')(x)  # ⭐ THIS IS YOUR FEATURES

    # =========================
    # OUTPUT LAYER (TEMPORARY FOR TRAINING CNN)
    # =========================
    output_layer = Dense(10, activation='softmax')(feature_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, feature_layer


def extract_and_save_features():
    # =========================
    # LOAD DATA
    # =========================
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # =========================
    # BUILD CNN
    # =========================
    model, feature_layer = build_cnn_feature_extractor()

    print("Training CNN...")
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

    # =========================
    # CREATE FEATURE EXTRACTOR MODEL
    # =========================
    feature_model = Model(inputs=model.input, outputs=feature_layer)

    # =========================
    # EXTRACT FEATURES
    # =========================
    print("Extracting features...")
    X_train_features = feature_model.predict(X_train)
    X_test_features = feature_model.predict(X_test)

    print("Feature shape:", X_train_features.shape)

    # =========================
    # SAVE FEATURES
    # =========================
    np.save("phase2/feature_data/X_train.npy", X_train_features)
    np.save("phase2/feature_data/X_test.npy", X_test_features)
    np.save("phase2/feature_data/y_train.npy", y_train)
    np.save("phase2/feature_data/y_test.npy", y_test)

    print("Features saved successfully!")


if __name__ == "__main__":
    extract_and_save_features()