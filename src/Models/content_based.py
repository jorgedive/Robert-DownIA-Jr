import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


load_dotenv()

tfidf_matrix_path = os.path.join(os.getenv("FILES_PATH"), "models", "content_based", "tfidf_description_matrix.pickle")
ccb_csv_path = os.path.join(os.getenv("FILES_PATH"), "CSV", "cleaned_content_based.csv")


def nltk_stopwords_punkt_download(state=False):
    if state:
        nltk.download("stopwords")
        nltk.download("punkt")


def load_data():
    """Loads cleaned_content_based.csv

    Returns:
        cleaned_content_based dataframe
    """
    try:
        df = pd.read_csv(ccb_csv_path, low_memory=False,
                         usecols=["title", "metadata", "description", "vote_count", "vote_average"])
        return df
    except Exception as e:
        print(f"Could not retrieve the file. Error: {e}")
        return None


def save_param_matrix(param_matrix):
    """Saves parameter matrix
    Args:
        param_matrix (numpy ndarray): numpy matrix containing similarity scores between movies
    """
    with open(tfidf_matrix_path, "wb") as f:
        pickle.dump(param_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_param_matrix():
    """Loads the model

    Returns:
        param_matrix (numpy ndarray): numpy matrix containing the similarity for each movie
    """

    with open(tfidf_matrix_path, "rb") as f:
        param_matrix = pickle.load(f)
    return param_matrix


def preprocess_data(dataframe):
    """Preprocess data before training the recommender based on description.
    Args:
        dataframe (pandas DataFrame): pandas DataFrame containing the data

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
        dataframe['description'] = dataframe['description'].transform(func=fn)
    return dataframe


def train_content_recommender(dataframe, count_vec=False):
    """Recommender training
    Args:
        dataframe (pandas DataFrame): pandas DataFrame with the training data
        count_vec (bool): boolean to indicate if use CountVectorizer for metadata column. Default False

    Returns:
        param_matrix (numpy ndarray): numpy matrix containing the the similarities between movies
    """

    if count_vec:
        dataframe["description"] = dataframe["description"].fillna("")
        count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0., stop_words="english")
        count_vec_matrix = count_vectorizer.fit_transform(dataframe["description"])
        return cosine_similarity(count_vec_matrix, count_vec_matrix)

    processed_data = preprocess_data(dataframe)
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), stop_words="english", min_df=0.)
    tfidf_matrix = tfidf.fit_transform(processed_data["description"])

    return linear_kernel(tfidf_matrix, tfidf_matrix)


def get_recommendation(movie, top_n=10, count_vec=False, state=False):
    """Recommender based on description or movie metadata
    Args:
        movie (str): movie title to take as baseline for the recommendations
        top_n (int): number of movies that will be shown. Default 10
        count_vec (bool): boolean to indicate if use CountVectorizer. Default False
        state (bool): boolean used to download the needed nltk packages

    Returns:
        movies_list (list): list with the recommended movies
    """

    nltk_stopwords_punkt_download(state=state)

    df = load_data()
    if df is None:
        raise ValueError("Dataframe not provided.")

    if not (isinstance(movie, str)) and (movie not in df["title"].unique()):
        raise ValueError("Movie not found.")

    try:
        param_matrix = load_param_matrix()
    except Exception:
        param_matrix = train_content_recommender(df, count_vec=count_vec)
        save_param_matrix(param_matrix)

    mapper = pd.Series(df.index, index=df["title"]).drop_duplicates()
    scores = np.argsort(param_matrix[mapper[movie]])[::-1]
    movie_indices = scores[1:top_n + 1]

    movies_list = df["title"].iloc[movie_indices].to_list()
    return "\n".join(movies_list)


def improved_recommendations(movie, q=0.6, top_n=25, count_vec=False,
                             state=False):
    """Recommender based on description or movie metadata
    Args:
        movie (str): movie title to take as baseline for the recommendations
        q (float): quantile to perform the filtering on the top q movies. Default 0.6
        top_n (int): number of movies to filter on with the percentile. Default 25
        count_vec (bool): boolean to indicate if use CountVectorizer. Default False
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

    recommended_movies = get_recommendation(movie, top_n=top_n, count_vec=count_vec, state=state)
    df = load_data()
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


load_data()
