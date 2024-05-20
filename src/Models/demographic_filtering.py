import os
import pandas as pd
from dotenv import load_dotenv
import ast

load_dotenv()
data_path = os.getenv("FILES_LOCATION")

# A Files/CSV/cleaned_movies.csv must exist within the Files folder
FILENAME = "cleaned_movies.csv"


def load_data(path=data_path):
    try:
        df = pd.read_csv(path, low_memory=False,
                         usecols=["title", "genres", "vote_count", "vote_average"])
        return df
    except Exception as e:
        print(f"Could not retrieve the file. Error: {e}")
        return None


def get_demographic_recommendation(genre, path=data_path, q=0.95, n_movies=10):
    """Simple recommender system that provides the top rated movies for a given genre.
        Args:
            genre (str): selected genre to perform the filtering
            path (str): path where the file to load as a dataframe is located
            q (float): q quantile to perform the filtering on the top q movies. Default 0.95
            n_movies (int): number of movies to show to the user

        Returns:
            movies_list (list): list with the recommended movies
        """

    df = load_data(path)
    if df is None:
        raise ValueError("Dataframe not provided.")

    # Gets the unique genres in the set to perform input validation
    unique_genres = set([genre for val in df["genres"].values for genre in ast.literal_eval(val)])

    if not (isinstance(genre, str)) and (genre.capitalize() in unique_genres):
        raise ValueError("Valid genre not provided.")

    if (q <= 0) or (q >= 1):
        raise ValueError("Quantile value not valid.")

    try:
        assert n_movies >= 1
    except AssertionError as e:
        print(f"Number of movies must be positive. Error: {e}")

    df["genres"] = df["genres"].apply(lambda x: ast.literal_eval(x))  # Evaluate entries as lists
    df["tmp_mask"] = df["genres"].apply(lambda x: genre in x)  # Mask to filter entries with the specified genre
    df_genre = df[df["tmp_mask"]]  # Filtered df by the mask

    # Weighted rate from IMDB
    def wr_func(data, C, m):
        v = data["vote_count"]
        R = data["vote_average"]
        return (v / (v + m) * R) + (m / (v + m) * C)

    # Compute the weighted rating for the recommender
    C = df_genre["vote_average"].mean()
    m = df_genre["vote_count"].quantile(q)
    df_genre = df_genre[df_genre["vote_count"] > m]
    df_genre["weighted_score"] = df_genre.apply(wr_func, axis=1, C=C, m=m)

    movies_list = df_genre.sort_values("weighted_score", ascending=False)["title"].to_list()[:n_movies]
    return "\n".join(movies_list)
