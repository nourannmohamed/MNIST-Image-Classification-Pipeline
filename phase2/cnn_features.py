import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

from phase2.preprocessing import load_and_preprocess


def build_cnn_feature_extractor():
    #input image shape
    input_layer = Input(shape=(28, 28, 1))

    #first convolution block
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)

    #second convolution block
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    #flatten image maps to vector
    x = Flatten()(x)
    feature_layer = Dense(128, activation='relu')(x)  #features used by other models

    #output layer only for training cnn
    output_layer = Dense(10, activation='softmax')(feature_layer)

    #build full cnn model
    model = Model(inputs=input_layer, outputs=output_layer)

    #compile model for digit classification
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, feature_layer


def extract_and_save_features():
    #load preprocessed mnist data
    X_train, X_test, y_train, y_test = load_and_preprocess()

    #build cnn and get feature layer
    model, feature_layer = build_cnn_feature_extractor()

    #train cnn before extracting features
    print("Training CNN...")
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

    #save cnn weights
    os.makedirs("saved_models", exist_ok=True)
    model.save_weights("saved_models/cnn_weights.weights.h5")

    #create model that outputs features only
    feature_model = Model(inputs=model.input, outputs=feature_layer)

    #extract features from train and test images
    print("Extracting features...")
    X_train_features = feature_model.predict(X_train)
    X_test_features = feature_model.predict(X_test)

    print("Feature shape:", X_train_features.shape)

    #save features and labels for phase2 models
    np.save("phase2/feature_data/X_train.npy", X_train_features)
    np.save("phase2/feature_data/X_test.npy", X_test_features)
    np.save("phase2/feature_data/y_train.npy", y_train)
    np.save("phase2/feature_data/y_test.npy", y_test)

    print("Features saved successfully!")


if __name__ == "__main__":
    #run feature extraction script
    extract_and_save_features()

