import ast
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_data():
    try:
        df = pd.read_csv(os.path.join(os.getenv("FILES_PATH"), "CSV", "cleaned_movies.csv"), low_memory=False,
                         usecols=["title", "genres", "vote_count", "vote_average"])
        return df
    except Exception as e:
        print(f"Could not retrieve the file. Error: {e}")
        pass


def get_demographic_recommendation(genre, q=0.95, n_movies=10):
    """Simple recommender system that provides the top rated movies for a given genre.
        Args:
            genre (str): selected genre to perform the filtering
            q (float): q quantile to perform the filtering on the top q movies. Default 0.95
            n_movies (int): number of movies to show to the user

        Returns:
            movies_list (list): list with the recommended movies
        """

    df = load_data()
    if df is None:
        raise ValueError("Dataframe not provided.")

    unique_genres = set([genre for val in df["genres"].values for genre in ast.literal_eval(val)])

    if not (isinstance(genre, str)) and (genre.capitalize() in unique_genres):
        raise ValueError("Valid genre not provided.")

    if (q <= 0) or (q >= 1):
        raise ValueError("Quantile value not valid.")

    try:
        assert n_movies >= 1
    except AssertionError as e:
        print(f"Number of movies must be positive. Error: {e}")

    df["genres"] = df["genres"].apply(lambda x: ast.literal_eval(x))
    df["tmp_mask"] = df["genres"].apply(lambda x: genre in x)
    df_genre = df[df["tmp_mask"]]

    def wr_func(data, C, m):
        v = data["vote_count"]
        R = data["vote_average"]
        return (v / (v + m) * R) + (m / (v + m) * C)

    vote_avg_mean = df_genre["vote_average"].mean()
    vote_count_quantile = df_genre["vote_count"].quantile(q)
    df_genre = df_genre[df_genre["vote_count"] > vote_count_quantile]
    df_genre["weighted_score"] = df_genre.apply(wr_func, axis=1, C=vote_avg_mean, m=vote_count_quantile)

    movies_list = df_genre.sort_values("weighted_score", ascending=False)["title"].to_list()[:n_movies]
    return "\n".join(movies_list)
