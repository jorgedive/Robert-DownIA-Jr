import pandas as pd
import os
from dotenv import load_dotenv
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential

load_dotenv()

def load_data():
    try:
        df = pd.read_csv(os.path.join(os.getenv("FILES_LOCATION"), "CSV", "cleaned_ratings.csv"), low_memory=False)
        return df
    except Exception as e:
        print(f"Could not retrieve the file. Error: {e}")
        return None


def dataset_split(random_state):
    dataframe = load_data()

    df_shuffled = dataframe.drop(columns=["timestamp"]).sample(frac=1, random_state=random_state)
    idx = int(0.9 * len(df_shuffled))

    df_train = df_shuffled[:idx]
    df_test = df_shuffled[idx:]

    return df_train, df_test


def preparare_data(train_set, test_set):
    user_mapper = {usr_id: i for i, usr_id in enumerate(train_set["userId"].unique())}
    movie_mapper = {mov_id: i for i, mov_id in enumerate(train_set["movieId"].unique())}

    user_emb = len(user_mapper)
    movie_emb = len(movie_mapper)

    user_train, user_test = train_set["userId"].map(user_mapper), test_set["userId"].map(user_mapper)
    movie_train, movie_test = train_set["movieId"].map(movie_mapper), test_set["movieId"].map(movie_mapper)

    return user_train, user_test, user_emb, movie_train, movie_test, movie_emb


def decomposition_collaborative(epochs=8, random_state=42):
    tf.random.set_seed(42)

    saving_path = os.path.join(os.getenv("MODELS_PATH"), "collaborative_filtering")
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    df_train, df_test = dataset_split(random_state)
    user_train, user_test, user_emb, movie_train, movie_test, movie_emb = preparare_data(df_train, df_test)
    embedding_dim = 10


    user_input = layers.Input(shape=(1,), name="user_in")
    movie_input = layers.Input(shape=(1,), name="movie_in")

    user_embeddings = layers.Embedding(output_dim=embedding_dim,
                                                input_dim=user_emb,
                                                input_length=1,
                                                name="user_embedding_layer")(user_input)

    movie_embeddings = layers.Embedding(output_dim=embedding_dim,
                                                 input_dim=movie_emb,
                                                 input_length=1,
                                                 name="movie_embedding_layer")(movie_input)

    user_vector = layers.Reshape([embedding_dim])(user_embeddings)
    movie_vector = layers.Reshape([embedding_dim])(movie_embeddings)

    out = tf.keras.layers.Dot(1, normalize=False)([user_vector, movie_vector])

    model = tf.keras.Model(inputs=[user_input, movie_input], outputs=out)
    model.compile(loss="mse", optimizer=Adam())

    model.fit([user_train, movie_train],
              df_train["rating"],
              batch_size=128,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True)

    model.save(os.path.join(saving_path, "collaborative_matrix_decomposition.keras"))
    return model


def deep_collaborative(epochs=8, random_state=42):
    tf.random.set_seed(42)

    saving_path = os.path.join(os.getenv("MODELS_PATH"), "collaborative_filtering")
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    df_train, df_test = dataset_split(random_state)
    user_train, user_test, user_emb, movie_train, movie_test, movie_emb = preparare_data(df_train, df_test)
    embedding_dim = 10

    user_input = tf.keras.layers.Input(shape=(1,), name="user_in")
    movie_input = tf.keras.layers.Input(shape=(1,), name="movie_in")

    user_embeddings = tf.keras.layers.Embedding(output_dim=embedding_dim,
                                                input_dim=user_emb,
                                                input_length=1,
                                                name="user_embedding")(user_input)

    movie_embeddings = tf.keras.layers.Embedding(output_dim=embedding_dim,
                                                 input_dim=movie_emb,
                                                 input_length=1,
                                                 name="movie_embedding")(movie_input)

    user_vector = tf.keras.layers.Reshape([embedding_dim])(user_embeddings)
    movie_vector = tf.keras.layers.Reshape([embedding_dim])(movie_embeddings)
    concat = tf.keras.layers.Concatenate()([user_vector, movie_vector])

    dense1 = tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal")(concat)
    dense2 = tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal")(dense1)
    y = tf.keras.layers.Dense(units=1, activation="linear")(dense2)

    model = tf.keras.Model(inputs=[user_input, movie_input], outputs=y)
    model.compile(loss="mse", optimizer="adam")

    model.fit([user_train, movie_train],
              df_train["rating"],
              batch_size=128,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True)

    model.save(os.path.join(saving_path, "collaborative_deep_learning.keras"))
    return model
