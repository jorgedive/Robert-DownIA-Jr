import os
import tensorflow as tf
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint


load_dotenv()

def download_kaggle_dataset(dataset_name, files_path):
    kaggle_path = os.path.join(files_path, "PNG", "muffin-chihuahua")

    if not os.path.exists(kaggle_path):
        api_kaggle = KaggleApi()
        api_kaggle.authenticate()
        api_kaggle.dataset_download_files(dataset_name, path=kaggle_path, unzip=True)
    return kaggle_path


def get_datasets(random_state=42):
    files_path = os.getenv("FILES_LOCATION")
    images_path = os.path.join(files_path, "PNG", "muffin_chihuahua")

    train_ds = image_dataset_from_directory(
        os.path.join(images_path, "train"),
        label_mode="binary",
        batch_size=32,
        seed=random_state,
        interpolation="bicubic")

    val_ds = image_dataset_from_directory(
        os.path.join(images_path, "test"),
        label_mode="binary",
        batch_size=32,
        seed=random_state,
        interpolation="bicubic")

    return train_ds, val_ds


def train_model(epochs=50, random_state=42):
    tf.random.set_seed(random_state)
    files_path = os.getenv("FILES_LOCATION")

    try:
        train_ds, val_ds = get_datasets(random_state)
    except:
        download_kaggle_dataset(os.getenv("KAGGLE_CHIHUAHUA"), files_path)
        train_ds, val_ds = get_datasets(random_state)

    resize_and_rescale = Sequential([layers.Resizing(128, 128), layers.Rescaling(1. / 255)])
    augment_image = Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.2)])

    model = Sequential([resize_and_rescale,
                        augment_image,
                        layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal"),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal"),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal"),
                        layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal"),
                        layers.MaxPooling2D((2, 2)),
                        layers.Flatten(),
                        layers.Dropout(rate=0.3),
                        layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
                        layers.Dense(1, activation="sigmoid")])

    check_cb = ModelCheckpoint(filepath=os.path.join(os.getenv("MODELS_PATH"), "muffin_chihuahua",
                                                                        "model_{epoch:04d}.keras"),
                                                  monitor="val_accuracy",
                                                  mode="max",
                                                  save_best_only=True)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[check_cb])
    return model
