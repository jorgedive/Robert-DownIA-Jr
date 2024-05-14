import os

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import image_dataset_from_directory

load_dotenv()


def download_kaggle_dataset(dataset_name, files_path):
    kaggle_path = os.path.join(files_path, "PNG", 'muffin-chihuahua')
    if not os.path.exists(kaggle_path):
        api_kaggle = KaggleApi()
        api_kaggle.authenticate()
        api_kaggle.dataset_download_files(dataset_name, path=kaggle_path, unzip=True)
    return kaggle_path


def create_data_set(base_dir):
    return (image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        interpolation="bicubic"), image_dataset_from_directory(
        base_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        interpolation="bicubic"))


def build_train_cnn_model(train_ds, val_ds, epochs=20, input_shape=(128, 128, 3)):
    # Resizes and reescales images. It is done to every image, both during training, evaluation and inference.
    resize_and_rescale = Sequential([layers.Resizing(128, 128), layers.Rescaling(1. / 255)])

    train_ds = resize_and_rescale(train_ds)
    val_ds = resize_and_rescale(val_ds)

    # All these layers only activate in training by default.
    data_augmentation = Sequential([layers.RandomFlip("horizontal_and_vertical"),
                                    layers.RandomRotation(0.25),
                                    layers.RandomZoom(height_factor=0.2, width_factor=0.2),
                                    layers.RandomWidth(0.1, interpolation="bicubic")])

    train_ds = data_augmentation(train_ds)

    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return model


def main():
    files_path = os.getenv("FILES_LOCATION")
    download_kaggle_dataset('samuelcortinhas/muffin-vs-chihuahua-image-classification', files_path)

    images_path = os.path.join(files_path, "PNG", "muffin-chihuahua")

    train_ds, val_ds = create_data_set(images_path)

    model = build_train_cnn_model(train_ds, val_ds)

    precision = model.evaluate(val_ds)
    print(f"Model precision = {precision[0] * 100}%. ")

    if precision[0] > 0.92:
        model.save(os.getenv("MODELS_PATH"))


if __name__ == "__main__":
    main()
