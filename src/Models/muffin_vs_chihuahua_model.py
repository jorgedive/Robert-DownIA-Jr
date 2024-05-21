import os
import tensorflow as tf
from dotenv import load_dotenv
#from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint


load_dotenv()

def download_kaggle_dataset(dataset_name, files_path):
    """ Downloads the dataset with dataset_name from kaggle using the kaggle API in case is not already downloaded.
    Args:
        dataset_name (str) - name of the dataset to be fetched
        files_path (str) - path containing the files of the project
    Returns:
        kaggle_path (str) - path where the kaggle dataset will be stored
    """

    kaggle_path = os.path.join(files_path, "PNG", "muffin-chihuahua")

    if not os.path.exists(kaggle_path):
        api_kaggle = KaggleApi()
        api_kaggle.authenticate()
        api_kaggle.dataset_download_files(dataset_name, path=kaggle_path, unzip=True)
    return kaggle_path


def get_datasets(random_state=42):
    """Fetches the stored dataset to return TensorFlow Dataset objects for the training.
    Args:
        random_state (int) - seed to set the random shuffle of the fetched data
    Returns:
        train_ds (tf.data.Dataset) - training dataset
        val_ds (tf.data.Dataset) - validation dataset
    """
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


def main():
    """Performs the training of the model. The architecture is hardcoded but may be changed in order to
    test other models. It is recommended to change the name of the Checkpoint path from the Callback if
    other architectures are going to be tested"""

    tf.random.set_seed(42)
    files_path = os.getenv("FILES_LOCATION")

    try:
        train_ds, val_ds = get_datasets()
    except:
        download_kaggle_dataset(os.getenv("KAGGLE_CHIHUAHUA"), files_path)
        train_ds, val_ds = get_datasets()

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

    check_cb = ModelCheckpoint(filepath=os.path.join(os.getenv("MODELS_PATH"), "muffin-chihuahua",
                                                                        "model_{epoch:04d}.keras"),
                                                  monitor="val_accuracy",
                                                  mode="max",
                                                  save_best_only=True)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=60, callbacks=[check_cb])


if __name__ == "__main__":
    main()
