import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import pickle
import numpy as np
import pandas as pd


load_dotenv()
models_path = os.getenv("MODELS_PATH")
files_path = os.getenv("FILES_LOCATION")

RECOMMENDER_TYPE = "content_based"
MODEL = "tfidf_description_matrix.pickle"
FILE_DATA = "cleaned_content_based.csv"

default_model_path = os.path.join(models_path, RECOMMENDER_TYPE, MODEL)
default_files_path = os.path.join(files_path, "CSV", FILE_DATA)


def nltk_stopwords_punkt_download(state=False):
    if state:
        nltk.download("stopwords")
        nltk.download("punkt")


def load_data(path=default_files_path):
    try:
        df = pd.read_csv(path, low_memory=False,
                         usecols=["title", "description", "metadata", "vote_count", "vote_average"])
        return df
    except Exception as e:
        print(f"Could not retrieve the file. Error: {e}")
        return None


def save_param_matrix(param_matrix, path=default_model_path):
    """Saves parameter matrix
    Args:
        param_matrix (numpy ndarray): numpy matrix containing similarity scores between movies
        path (str): path where the model will be saved with pickle
    """
    with open(path, "wb") as f:
        pickle.dump(param_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_param_matrix(path=default_model_path):
    """Loads the model
    Args:
        path (str): path where the model is located

    Returns:
        param_matrix (numpy ndarray): numpy matrix containing the similarity for each movie
    """

    with open(path, "rb") as f:
        param_matrix = pickle.load(f)
    return param_matrix


def preprocess_data(dataframe, col="description"):
    """Preprocess data before training the recommender based on description.
    Args:
        dataframe (pandas DataFrame): pandas DataFrame containing the data
        col (str): number of the column to process. Default "description"

    Returns:
        dataframe (pandas DataFrame): processed data with nltk
    """

    nltk.download("stopwords")
    stop_w = set(stopwords.words("english"))

    remove_non_alpha_fn = lambda x: re.sub(r"[^a-zA-Z]", " ", x)
    tokenize_fn = lambda x: nltk.tokenize.word_tokenize(x)

    stemmer = SnowballStemmer("english")
    stemmer_fn = lambda desc: " ".join([stemmer.stem(w) for w in desc if w.lower() not in stop_w])

    for fn in (remove_non_alpha_fn, tokenize_fn, stemmer_fn):
        dataframe[col] = dataframe[col].transform(func=fn)
    return dataframe


def train_content_recommender(dataframe, preprocess_fn=preprocess_data, col="description", count_vec=False):
    """Recommender training
    Args:
        dataframe (pandas DataFrame): pandas DataFrame with the training data
        preprocess_fn (function): function that processes the data before training
        col (str): column where we perform the training. Default "description"
        count_vec (bool): boolean to indicate if use CountVectorizer for metadata column. Default False

    Returns:
        param_matrix (numpy ndarray): numpy matrix containing the the similarities between movies
    """

    if count_vec:
        dataframe[col] = dataframe[col].fillna("")
        count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0., stop_words="english")
        count_vec_matrix = count_vectorizer.fit_transform(dataframe[col])
        return cosine_similarity(count_vec_matrix, count_vec_matrix)

    processed_data = preprocess_fn(dataframe, col)
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), stop_words="english", min_df=0.)
    tfidf_matrix = tfidf.fit_transform(processed_data[col])

    return linear_kernel(tfidf_matrix, tfidf_matrix)


def get_recommendation(movie, top_n=10, preprocess_fn=preprocess_data, col="description", count_vec=False,
                                   file_path=default_files_path, model_path=default_model_path, state=False):
    """Recommender based on description or movie metadata
    Args:
        movie (str): movie title to take as baseline for the recommendations
        top_n (int): number of movies that will be shown. Default 10
        preprocess_fn (function): function to preprocess the data in case there is no model
        col (str): name for the column to train on in case training is needed. Default description
        count_vec (bool): boolean to indicate if use CountVectorizer. Default False
        file_path (str): path where the csv data file is located
        model_path (str): path where the model is stored
        state (bool): boolean used to download the needed nltk packages

    Returns:
        movies_list (list): list with the recommended movies
    """

    nltk_stopwords_punkt_download(state)

    df = load_data(file_path)
    if df is None:
        raise ValueError("Dataframe not provided.")

    if not (isinstance(movie, str)) and (movie not in df["title"].unique()):
        raise ValueError("Movie not found.")

    try:
        param_matrix = load_param_matrix(model_path)
    except Exception:
        param_matrix = train_content_recommender(df, preprocess_fn, col, count_vec)
        save_param_matrix(param_matrix, model_path)

    mapper = pd.Series(df.index, index=df["title"]).drop_duplicates()
    scores = np.argsort(param_matrix[mapper[movie]])[::-1]
    movie_indices = scores[1:top_n + 1]

    movies_list = df["title"].iloc[movie_indices].to_list()
    return "\n".join(movies_list)


def improved_recommendations(movie, q=0.6, top_n=25, preprocess_fn=preprocess_data, col="description", count_vec=False,
                             file_path=default_files_path, model_path=default_model_path, state=False):
    """Recommender based on description or movie metadata
    Args:
        movie (str): movie title to take as baseline for the recommendations
        q (float): quantile to perform the filtering on the top q movies. Default 0.6
        top_n (int): number of movies to filter on with the percentile. Default 25
        preprocess_fn (function): function to preprocess the data in case there is no model
        col (str): name for the column to train on in case training is needed. Default description
        count_vec (bool): boolean to indicate if use CountVectorizer. Default False
        file_path (str): path where the csv data file is located
        model_path (str): path where the model is stored
        state (bool): boolean used to download the needed nltk packages

    Returns:
        movies_list (list): list with the recommended movies
    """

    if (q <= 0) or (q >= 1):
        raise ValueError("Quantile value not valid.")

    try:
        assert top_n >= 1
    except AssertionError as e:
        print(f"Number of movies must be positive. Error: {e}")

    recommended_movies = get_recommendation(movie, top_n, preprocess_fn, col, count_vec, file_path, model_path, state)
    df = load_data(file_path)
    recommended_df = df[df["title"].isin(recommended_movies.split("\n"))]

    def wr_func(data, C, m):
        v = data["vote_count"]
        R = data["vote_average"]
        return (v / (v + m) * R) + (m / (v + m) * C)

    C = recommended_df["vote_average"].mean()
    m = recommended_df["vote_count"].quantile(q)
    recommended_df = recommended_df[recommended_df["vote_count"] > m]
    recommended_df["weighted_score"] = recommended_df.apply(wr_func, axis=1, C=C, m=m)

    return "\n".join(recommended_df.sort_values(by="weighted_score", ascending=False)["title"].to_list())
