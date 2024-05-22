import os

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import image_dataset_from_directory


load_dotenv()


def download_kaggle_dataset():
    """ Downloads the dataset with the Kaggle Chihuahua environment variable from kaggle using the kaggle API in case
    is not already downloaded.
    Returns:
        kaggle_path (str) - path where the kaggle dataset will be stored
    """

    kaggle_path = os.path.join(os.getenv("FILES_PATH"), "PNG", "muffin-chihuahua")

    if not os.path.exists(kaggle_path):
        api_kaggle = KaggleApi()
        api_kaggle.authenticate()
        api_kaggle.dataset_download_files(os.getenv("KAGGLE_CHIHUAHUA"), path=kaggle_path, unzip=True)
    return kaggle_path


def get_datasets(random_state=42):
    """Fetches the stored dataset to return TensorFlow Dataset objects for the training.
    Args:
        random_state (int) - seed to set the random shuffle of the fetched data
    Returns:
        train_ds (tf.data.Dataset) - training dataset
        val_ds (tf.data.Dataset) - validation dataset
    """

    files_path = os.getenv("FILES_PATH")
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
    """Trains the MobileNetV2 model using 2 Dense layers on top of it to perform a fine-tuning
    on the dataset"""

    saving_path = os.path.join(os.getenv("FILES_PATH"), "models", "muffin-chihuahua")
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    try:
        train_ds, val_ds = get_datasets()
    except:
        download_kaggle_dataset()
        train_ds, val_ds = get_datasets()

    resize_and_rescale = Sequential([layers.Resizing(224, 224), layers.Rescaling(1. / 255)])
    augment_image = Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.2)])

    model = Sequential([
        resize_and_rescale,
        augment_image,
        MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet"),
        layers.Flatten(),
        layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.layers[2].trainable = False
    print(model.layers[2])
    model.compile(optimizer=SGD(learning_rate=1e-2), loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, epochs=5, validation_data=val_ds, validation_steps=10)
    model.save(os.path.join(saving_path, "finetuned_mobilenetV2.keras"))


if __name__ == "__main__":
    main()
