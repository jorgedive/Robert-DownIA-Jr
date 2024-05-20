import os
from ast import literal_eval

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def create_movies_csv():
    movies_df = pd.read_csv(os.path.join(os.getenv("FILES_LOCATION"), "CSV", "movies_metadata.csv"), low_memory=False)
    movies_df = movies_df[movies_df["adult"] == "False"].drop(columns=["adult"])
    movies_df = movies_df[movies_df["status"] != "Canceled"]
    movies_df = movies_df[movies_df["overview"].notnull()]
    movies_df["release_date"] = pd.to_datetime(movies_df["release_date"], errors="coerce")
    movies_df["popularity"] = pd.to_numeric(movies_df["popularity"], errors="coerce")
    movies_df["tagline"] = movies_df["tagline"].fillna("")
    movies_df["description"] = movies_df["overview"] + movies_df["tagline"]

    relevant_cols = ["genres", "imdb_id", "title", "overview", "description", "popularity", "vote_average",
                     "vote_count"]
    movies_df = movies_df[relevant_cols]

    list_converter = lambda series: [genre["name"] for genre in literal_eval(series)] if isinstance(
        literal_eval(series),
        list) else []

    movies_df["genres"] = movies_df["genres"].fillna("[]").transform(func=list_converter)

    movies_df.fillna(
        {"popularity": movies_df["popularity"].median(), "vote_average": movies_df["vote_average"].median(),
         "vote_count": movies_df["vote_count"].median()}, inplace=True)

    movies_df = movies_df.drop_duplicates("imdb_id")
    movies_df = movies_df.drop_duplicates("title").dropna()

    movies_df.to_csv(os.path.join(os.getenv("FILES_LOCATION"), "CSV", "cleaned_movies.csv"), encoding="UTF-8",
                     index=False)


if __name__ == "__main__":
    create_movies_csv()
