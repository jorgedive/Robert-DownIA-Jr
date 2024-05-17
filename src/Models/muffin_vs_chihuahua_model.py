import os

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, MaxPooling2D, RandomFlip, RandomRotation, RandomZoom,
                                     Rescaling, Resizing)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory

load_dotenv()


def download_kaggle_dataset(dataset_name, files_path):
    kaggle_path = os.path.join(files_path, "PNG", 'muffin-chihuahua')
    if not os.path.exists(kaggle_path):
        api_kaggle = KaggleApi()
        api_kaggle.authenticate()
        api_kaggle.dataset_download_files(dataset_name, path=kaggle_path, unzip=True)
    return kaggle_path


def main():
    files_path = os.getenv("FILES_LOCATION")
    download_kaggle_dataset(os.getenv("KAGGLE_CHIHUAHUA"), files_path)

    images_path = os.path.join(files_path, "PNG", "muffin-chihuahua")

    train_ds = image_dataset_from_directory(os.path.join(images_path, "train"),
                                            label_mode="binary",
                                            batch_size=16,
                                            seed=42,
                                            interpolation="bicubic")
    val_ds = image_dataset_from_directory(os.path.join(images_path, "test"),
                                          label_mode="binary",
                                          batch_size=16,
                                          seed=42,
                                          interpolation="bicubic")

    resize_and_rescale = Sequential([Resizing(128, 128), Rescaling(1. / 255)])
    augment_image = Sequential([RandomFlip("horizontal"), RandomRotation(0.2),
                                RandomZoom(height_factor=(0, 0.2), width_factor=(0, 0.2))])

    model = Sequential([resize_and_rescale,
                        augment_image,
                        Input((128, 128, 3)),
                        Conv2D(16, (3, 3), strides=(2, 2), activation='relu', kernel_initializer="he_normal"),
                        MaxPooling2D((2, 2)),
                        Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer="he_normal"),
                        MaxPooling2D((2, 2)),
                        Conv2D(64, (3, 3), activation='relu', kernel_initializer="he_normal"),
                        MaxPooling2D((2, 2)),
                        Flatten(),
                        Dense(128, activation='relu', kernel_initializer="he_normal"),
                        Dense(1, activation='sigmoid')])

    model.compile(optimizer=Adam(weight_decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=30)

    precision = model.evaluate(val_ds)
    print(precision)

    model.save(os.path.join(os.getenv("MODELS_PATH"), "muffin_vs_chihuahua.keras"))


if __name__ == "__main__":
    main()
