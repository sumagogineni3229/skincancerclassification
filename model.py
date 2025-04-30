import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load Dataset
def load_data():
    skin_df = pd.read_csv('HAM10000_metadata.csv')
    data_folder_name = "static/HAM10000_images"
    ext = ".jpg"
    
    skin_df["path"] = [data_folder_name + "/" + img_id + ext for img_id in skin_df["image_id"]]
    
    # Check and process only existing files
    existing_files_mask = skin_df["path"].apply(lambda x: os.path.exists(x))
    skin_df = skin_df[existing_files_mask]  # Filtering out rows with non-existing files

    # Process images
    skin_df["image"] = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))
    skin_df["dx_idx"] = pd.Categorical(skin_df["dx"]).codes

    # Standardization - Normalization
    x_train = np.asarray(skin_df["image"].tolist())
    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)
    x_train = (x_train - x_train_mean) / x_train_std

    # One-Hot Encoding
    num_classes = skin_df["dx"].nunique()
    y_train = to_categorical(skin_df["dx_idx"], num_classes=num_classes)

    return x_train, y_train, num_classes, x_train_mean, x_train_std

# Create CNN Model
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

# Train and Save Model
def train_and_save_model(model, x_train, y_train, model_path, epochs=10, batch_size=32):
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
    model.save(model_path)

# Load Model
def load_trained_model(model_path):
    return load_model(model_path)

# Main function to train and save the models
def main():
    x_train, y_train, num_classes, x_train_mean, x_train_std = load_data()
    input_shape = x_train.shape[1:]

    # Train Model 1
    model_1 = create_model(input_shape, num_classes)
    train_and_save_model(model_1, x_train, y_train, 'static/model/my_model_1.h5', epochs=10, batch_size=300)

    # Train Model 2
    model_2 = create_model(input_shape, num_classes)
    train_and_save_model(model_2, x_train, y_train, 'static/model/my_model_2.h5', epochs=10, batch_size=300)

if __name__ == '__main__':
    main()
