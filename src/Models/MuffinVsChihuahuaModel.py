import os

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from keras.src.layers import (Conv2D, Flatten, MaxPooling2D, RandomFlip, RandomRotation, RandomZoom, Rescaling,
                              Resizing)
from keras.src.optimizers import Adam
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
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
    download_kaggle_dataset('samuelcortinhas/muffin-vs-chihuahua-image-classification', files_path)

    images_path = os.path.join(files_path, "PNG", "muffin-chihuahua")

    train_ds = image_dataset_from_directory(images_path,
                                            label_mode="binary",
                                            batch_size=16,
                                            validation_split=0.2,
                                            subset="training",
                                            seed=42,
                                            interpolation="bicubic")
    val_ds = image_dataset_from_directory(images_path,
                                          label_mode="binary",
                                          batch_size=16,
                                          validation_split=0.2,
                                          subset="validation",
                                          seed=42,
                                          interpolation="bicubic")

    resize_and_rescale = Sequential([Resizing(128, 128), Rescaling(1. / 255)])
    augment_image = Sequential([RandomFlip("horizontal"), RandomRotation(0.2),
                                RandomZoom(height_factor=(0, 0.2), width_factor=(0, 0.2))])

    model = Sequential([resize_and_rescale,
                        augment_image,
                        Input((128, 128, 3)),
                        Conv2D(16, (3, 3), strides=(2, 2), activation='relu', kernel_initializer="he_normal"),
                        Conv2D(32, (3, 3), strides=(2, 2), activation='relu', kernel_initializer="he_normal"),
                        MaxPooling2D((2, 2)),
                        Conv2D(32, (3, 3), activation='relu', kernel_initializer="he_normal"),
                        MaxPooling2D((2, 2)),
                        Flatten(),
                        Dense(128, activation='relu', kernel_initializer="he_normal"),
                        Dense(1, activation='sigmoid')])

    # RMSprop(momentum=0.9, decay=0.01)
    model.compile(optimizer=Adam(decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    precision = model.evaluate(val_ds)
    print(f"Model precision = {precision[1] * 100}%. ")

    model.save(os.getenv("MODELS_PATH"))


if __name__ == "__main__":
    main()
